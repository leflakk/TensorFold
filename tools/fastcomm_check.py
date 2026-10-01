"""Check and time the shared-memory collectives on this host's GPUs, against NCCL.

    python tools/fastcomm_check.py --world 8            # correctness (bit-exact) + latency table
    python tools/fastcomm_check.py --world 4 --quick    # correctness only

Every rank builds every rank's inputs from fixed seeds, so the expected rank-ordered sum is known locally and the
check is bitwise. Run it once on a new machine before serving with --tp N.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.multiprocessing as mp

D = 2560


def inputs(rank: int, n: int, dtype: torch.dtype, call: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(1000 * call + rank)
    return (torch.randn((n,), generator=g, device="cuda") * (1 + rank)).to(dtype)


def expected(world: int, n: int, dtype: torch.dtype, out_dtype: torch.dtype, call: int) -> torch.Tensor:
    acc = inputs(0, n, dtype, call).float()
    for r in range(1, world):
        acc = acc + inputs(r, n, dtype, call).float()      # fp32, rank order, one rounding a step
    return acc.to(out_dtype)


def run(rank: int, world: int, port: int, quick: bool, results) -> None:
    from datetime import timedelta

    from torch.distributed import TCPStore

    from tensorfold.cuda.fastcomm import FastComm, bench

    torch.cuda.set_device(rank)
    store = TCPStore("127.0.0.1", port, world, rank == 0, timeout=timedelta(seconds=600))
    cap = 2048 * D * 4
    fc = FastComm(store, rank, world, cap)
    failures = []
    call = 0
    cases = [(rows, ti, to, shot) for rows in (1, 2, 4, 7, 16, 64, 2048)
             for ti, to in ((torch.float32, torch.bfloat16), (torch.float32, torch.float32),
                            (torch.bfloat16, torch.bfloat16))
             for shot in ("1", "2")]
    for rows, ti, to, shot in cases:
        fc.shot = shot
        call += 1
        n = rows * D
        x, out = inputs(rank, n, ti, call), torch.empty((n,), dtype=to, device="cuda")
        fc.reduce(x, out)
        want = expected(world, n, ti, to, call)
        if not torch.equal(out.view(torch.int16 if to == torch.bfloat16 else torch.int32),
                           want.view(torch.int16 if to == torch.bfloat16 else torch.int32)):
            failures.append(f"reduce rows={rows} {ti}->{to} shot={shot}")
    fc.shot = "auto"
    # all-gather of candidate-shaped rows (R x 65 fp32)
    for rows in (1, 7, 16):
        call += 1
        x = inputs(rank, rows * 65, torch.float32, call)
        got = torch.empty((world * rows * 65,), dtype=torch.float32, device="cuda")
        fc.all_gather(x, got)
        want = torch.cat([inputs(r, rows * 65, torch.float32, call) for r in range(world)])
        if not torch.equal(got, want):
            failures.append(f"gather rows={rows}")
    # inside a CUDA graph: three reduces captured, replayed with new inputs
    xs = [torch.empty((7 * D,), dtype=torch.float32, device="cuda") for _ in range(3)]
    outs = [torch.empty((7 * D,), dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    for x in xs:
        x.copy_(inputs(rank, 7 * D, torch.float32, 0))
    for x, o in zip(xs, outs):
        fc.reduce(x, o)                              # warm-up outside the capture, as Graphs does
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x, o in zip(xs, outs):
            fc.reduce(x, o)
    for rep in range(5):
        call += 1
        for i, x in enumerate(xs):
            x.copy_(inputs(rank, 7 * D, torch.float32, 100 * call + i))
        graph.replay()
        torch.cuda.synchronize()
        for i, o in enumerate(outs):
            if not torch.equal(o.view(torch.int16),
                               expected(world, 7 * D, torch.float32, torch.bfloat16, 100 * call + i).view(torch.int16)):
                failures.append(f"graph replay {rep} call {i}")
    times = {}
    if not quick:
        import torch.distributed as dist

        dist.init_process_group("nccl", store=torch.distributed.PrefixStore("nccl", store), rank=rank,
                                world_size=world)
        for rows, ti, label in ((1, torch.float32, "decode R=1 fp32"), (4, torch.float32, "decode R=4 fp32"),
                                (7, torch.float32, "decode R=7 fp32"), (2048, torch.bfloat16, "prefill 2048 bf16")):
            n = rows * D
            x = inputs(rank, n, ti, 1)
            out = torch.empty((n,), dtype=torch.bfloat16, device="cuda")
            gathered = torch.empty((world * n,), dtype=ti, device="cuda")
            row = {"shm": bench(lambda: fc.reduce(x, out))}
            for shot in ("1", "2"):
                fc.shot = shot
                row[f"shm{shot}"] = bench(lambda: fc.reduce(x, out))
            fc.shot = "auto"
            row["nccl_all_gather"] = bench(lambda: dist.all_gather_into_tensor(gathered, x))
            y = x.clone()
            row["nccl_all_reduce"] = bench(lambda: dist.all_reduce(y))
            times[label] = row
        dist.destroy_process_group()
    fc.close()
    results.put((rank, failures, times))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=torch.cuda.device_count())
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=run, args=(r, a.world, a.port, a.quick, q)) for r in range(a.world)]
    for p in procs:
        p.start()
    got = [q.get() for _ in procs]
    for p in procs:
        p.join()
    bad = [f"rank {r}: {f}" for r, fs, _ in got for f in fs]
    print(f"fastcomm on {a.world} GPUs: " + ("all checks bit-exact" if not bad else f"{len(bad)} FAILURES"))
    for line in bad[:20]:
        print("  ", line)
    times = next(t for r, _, t in got if r == 0)
    if times:
        cols = list(next(iter(times.values())))
        print(f"{'microseconds a call':22}" + "".join(f"{c:>17}" for c in cols))
        for label, row in times.items():
            print(f"{label:22}" + "".join(f"{row[c]:17.1f}" for c in cols))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
