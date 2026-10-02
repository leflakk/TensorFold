"""Check and time the shared-memory collectives on this host's GPUs, against NCCL.

    python tools/fastcomm_check.py --world 8            # correctness (bit-exact) + latency tables
    python tools/fastcomm_check.py --world 4 --quick    # correctness only

Every rank builds every rank's inputs from fixed seeds, so the expected rank-ordered sum is known locally and the
check is bitwise. Latencies are measured inside CUDA graphs (as decode runs them: no Python or launch cost), 50
calls a graph, and also eagerly for the large prompt-chunk size. Modes: 1/2 one/two-shot with flags, 3/4 one/two-shot
LL (each 8 bytes carry data and the call number).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.multiprocessing as mp

D = 2560
MODES = ("1", "2", "3", "4")


def inputs(rank: int, n: int, dtype: torch.dtype, call: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(1000 * call + rank)
    return (torch.randn((n,), generator=g, device="cuda") * (1 + rank)).to(dtype)


def expected(world: int, n: int, dtype: torch.dtype, out_dtype: torch.dtype, call: int) -> torch.Tensor:
    acc = inputs(0, n, dtype, call).float()
    for r in range(1, world):
        acc = acc + inputs(r, n, dtype, call).float()      # fp32, rank order, one rounding a step
    return acc.to(out_dtype)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def graph_us(fn, calls: int = 50, reps: int = 20) -> float:
    """Microseconds a call, ``calls`` calls captured in one graph and replayed ``reps`` times."""

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (reps * calls) * 1e6


def eager_us(fn, iters: int = 100) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def run(rank: int, world: int, port: int, quick: bool, results) -> None:
    from datetime import timedelta

    from torch.distributed import TCPStore

    from tensorfold.cuda.fastcomm import FastComm

    torch.cuda.set_device(rank)
    store = TCPStore("127.0.0.1", port, world, rank == 0, timeout=timedelta(seconds=600))
    fc = FastComm(store, rank, world, 2048 * D * 4)
    failures, times, call = [], {}, 0
    for rows in (1, 2, 4, 7, 16, 64, 2048):
        for ti, to in ((torch.float32, torch.bfloat16), (torch.float32, torch.float32),
                       (torch.bfloat16, torch.bfloat16)):
            for shot in MODES:
                n = rows * D
                if shot in ("3", "4") and 2 * n * torch.finfo(ti).bits // 8 > fc.cap:
                    continue                                   # LL takes up to half the region
                fc.shot = shot
                call += 1
                x, out = inputs(rank, n, ti, call), torch.empty((n,), dtype=to, device="cuda")
                fc.reduce(x, out)
                if not torch.equal(bits(out), bits(expected(world, n, ti, to, call))):
                    failures.append(f"reduce rows={rows} {ti}->{to} mode={shot}")
    for ll in (False, True):                                   # candidate-shaped gathers (R x 65 fp32), both forms
        fc.gather_ll = ll
        for rows in (1, 7, 16):
            call += 1
            x = inputs(rank, rows * 65, torch.float32, call)
            got = torch.empty((world * rows * 65,), dtype=torch.float32, device="cuda")
            fc.all_gather(x, got)
            if not torch.equal(got, torch.cat([inputs(r, rows * 65, torch.float32, call) for r in range(world)])):
                failures.append(f"gather rows={rows} ll={ll}")
    for shot in MODES:                                         # graph replays with new inputs, every mode
        fc.shot = shot
        xs = [torch.empty((7 * D,), dtype=torch.float32, device="cuda") for _ in range(3)]
        outs = [torch.empty((7 * D,), dtype=torch.bfloat16, device="cuda") for _ in range(3)]
        for x, o in zip(xs, outs):
            fc.reduce(x, o)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for x, o in zip(xs, outs):
                fc.reduce(x, o)
        for rep in range(4):
            call += 1
            for i, x in enumerate(xs):
                x.copy_(inputs(rank, 7 * D, torch.float32, 100 * call + i))
            graph.replay()
            torch.cuda.synchronize()
            for i, o in enumerate(outs):
                if not torch.equal(bits(o), bits(expected(world, 7 * D, torch.float32, torch.bfloat16, 100 * call + i))):
                    failures.append(f"graph mode={shot} replay {rep} call {i}")
    fc.shot, fc.gather_ll = "auto", True
    if not quick and not failures:
        import torch.distributed as dist

        dist.init_process_group("nccl", store=torch.distributed.PrefixStore("nccl", store), rank=rank,
                                world_size=world)
        for rows in (1, 2, 4, 7, 16):                          # decode windows: fp32 partials -> bf16, in graphs
            n = rows * D
            x = inputs(rank, n, torch.float32, 1)
            out = torch.empty((n,), dtype=torch.bfloat16, device="cuda")
            row = {}
            for shot in MODES:
                fc.shot = shot
                row[f"mode{shot}"] = graph_us(lambda: fc.reduce(x, out))
            fc.shot = "auto"
            row["auto"] = graph_us(lambda: fc.reduce(x, out))
            y = x.clone()
            try:
                row["nccl_ar"] = graph_us(lambda: dist.all_reduce(y))
            except Exception:                                  # noqa: BLE001 - a capture NCCL refuses: eager
                torch.cuda.synchronize()
                row["nccl_ar"] = eager_us(lambda: dist.all_reduce(y))
            times[f"R={rows} fp32 (graph)"] = row
        for rows in (1, 4, 7, 16):                             # TF_DECODE_PARTIALS=bf16 (the default): half the bytes
            n = rows * D
            x = inputs(rank, n, torch.bfloat16, 1)
            out = torch.empty((n,), dtype=torch.bfloat16, device="cuda")
            row = {}
            for shot in ("2", "4"):
                fc.shot = shot
                row[f"mode{shot}"] = graph_us(lambda: fc.reduce(x, out))
            fc.shot = "auto"
            y = x.clone()
            try:
                row["nccl_ar"] = graph_us(lambda: dist.all_reduce(y))
            except Exception:                                  # noqa: BLE001
                torch.cuda.synchronize()
                row["nccl_ar"] = eager_us(lambda: dist.all_reduce(y))
            times[f"R={rows} bf16 (graph)"] = row
        sweep = {}
        for rows in (4, 7, 16):                                # blocks for the LL two-shot and flag two-shot
            n = rows * D
            x = inputs(rank, n, torch.float32, 1)
            out = torch.empty((n,), dtype=torch.bfloat16, device="cuda")
            for shot in ("2", "4"):
                fc.shot = shot
                for blocks in (1, 2, 4, 8, 16, 32, 64):
                    fc.blocks = blocks
                    sweep[(rows, shot, blocks)] = graph_us(lambda: fc.reduce(x, out), calls=30, reps=10)
            fc.blocks = 0
        fc.shot = "auto"
        times["sweep"] = sweep
        rows = 65                                              # sampling candidates of 1 row a rank (gather)
        for ll in (False, True):
            fc.gather_ll = ll
            x = inputs(rank, rows, torch.float32, 1)
            got = torch.empty((world * rows,), dtype=torch.float32, device="cuda")
            times.setdefault("gather 65 fp32 (graph)", {})["ll" if ll else "flags"] = graph_us(
                lambda: fc.all_gather(x, got))
        n = 2048 * D                                           # a prompt chunk's bf16 partials: eager
        x = inputs(rank, n, torch.bfloat16, 1)
        out = torch.empty((n,), dtype=torch.bfloat16, device="cuda")
        fc.shot = "2"
        big = {"shm2": eager_us(lambda: fc.reduce(x, out), iters=20)}
        y = x.clone()
        big["nccl_ar"] = eager_us(lambda: dist.all_reduce(y), iters=20)
        times["prefill 2048 bf16 (eager)"] = big
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
    print(f"fastcomm on {a.world} GPUs: " + ("all checks bit-exact (modes 1-4, gathers, graph replays)" if not bad
                                            else f"{len(bad)} FAILURES"))
    for line in bad[:30]:
        print("  ", line)
    times = next(t for r, _, t in got if r == 0)
    sweep = times.pop("sweep", None)
    for label, row in times.items():
        print(f"{label:28}" + "  ".join(f"{k} {v:7.1f}" for k, v in row.items()) + "   (us a call)")
    if sweep:
        print("blocks sweep (us a call, in graphs):")
        for (rows, shot, blocks), us in sorted(sweep.items()):
            print(f"  R={rows:<3} mode{shot} blocks={blocks:<3} {us:7.1f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
