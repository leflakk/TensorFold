"""NCCL all-reduce bandwidth at prompt-chunk sizes on this host's GPUs (bf16, as Flash Next's prompt partials).

    python tools/nccl_bw.py --world 8                       # ~20 s
    NCCL_SHM_USE_CUDA_MEMCPY=1 python tools/nccl_bw.py      # then compare NCCL settings one run each

Prints, per message size, the time a call, the algorithm bandwidth (bytes / time) and the bus bandwidth
(2 (n - 1) / n x algbw, what each GPU moves through its link), with the NCCL_* variables in effect.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.multiprocessing as mp

SIZES_MB = (1.3, 2.6, 5.2, 10.5, 21.0)        # 256, 512, 1024, 2048, 4096 prompt rows x 2560 bf16


def run(rank: int, world: int, port: int, results) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    rows = []
    for mb in SIZES_MB:
        n = int(mb * 1e6 / 2) // 8 * 8
        x = torch.randn((n,), device="cuda").to(torch.bfloat16)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        s = (time.perf_counter() - t0) / iters
        alg = n * 2 / s / 1e9
        rows.append((mb, s * 1e3, alg, alg * 2 * (world - 1) / world))
    dist.destroy_process_group()
    results.put((rank, rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=torch.cuda.device_count())
    ap.add_argument("--port", type=int, default=29621)
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=run, args=(r, a.world, a.port, q)) for r in range(a.world)]
    for p in procs:
        p.start()
    got = dict(q.get() for _ in procs)
    for p in procs:
        p.join()
    print(f"NCCL all-reduce bf16 on {a.world} GPUs, env {env or '{}'}")
    for mb, ms, alg, bus in got[0]:
        print(f"  {mb:5.1f} MB  {ms:7.3f} ms  algbw {alg:5.2f} GB/s  busbw {bus:5.2f} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
