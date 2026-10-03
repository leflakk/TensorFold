"""Host <-> GPU copy bandwidth of every GPU, alone and all at once, idle and under compute (PCIe links).

    python tools/pcie_bw.py                 # ~1 min; unset CUDA_VISIBLE_DEVICES first to see every GPU

The collectives of tensor-parallel ranks on one host go through host memory (no P2P on RTX 3090s), so each GPU's
PCIe link bounds them, and the slowest link bounds everyone. GeForce cards also lower their link (down to Gen1)
when they look idle: the "busy" rows keep a matmul running on each GPU while copying. The link generation and
width are read from nvidia-smi in the middle of each measurement.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import torch
import torch.multiprocessing as mp

MB = 256


def links() -> dict[int, str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,pcie.link.gen.current,pcie.link.width.current",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    got = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            got[int(parts[0])] = f"Gen{parts[1]} x{parts[2]}"
    return got


def measure(device: int, busy: bool, start=None, results=None) -> tuple[float, float]:
    torch.cuda.set_device(device)
    host = torch.empty((MB << 20,), dtype=torch.uint8, pin_memory=True)
    dev = torch.empty((MB << 20,), dtype=torch.uint8, device="cuda")
    stop = threading.Event()
    if busy:                                   # a matmul loop on its own stream keeps the GPU out of idle states
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
        side = torch.cuda.Stream()

        def spin() -> None:
            with torch.cuda.stream(side):
                while not stop.is_set():
                    a @ a
                    side.synchronize()

        th = threading.Thread(target=spin, daemon=True)
        th.start()
        time.sleep(0.5)
    if start is not None:
        start.wait()
    rates = []
    for direction in ("h2d", "d2h"):
        dst, src = (dev, host) if direction == "h2d" else (host, dev)
        copies = torch.cuda.current_stream()     # this stream only: the busy loop's matmuls run on their own
        dst.copy_(src, non_blocking=True)
        copies.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            dst.copy_(src, non_blocking=True)
        copies.synchronize()
        rates.append(5 * MB / 1024 / (time.perf_counter() - t0))
    stop.set()
    if results is not None:
        results.put((device, rates))
    return rates[0], rates[1]


def concurrent(n: int, busy: bool) -> dict[int, tuple[float, float]]:
    ctx = mp.get_context("spawn")
    q, start = ctx.Queue(), ctx.Barrier(n)
    procs = [ctx.Process(target=measure, args=(i, busy, start, q)) for i in range(n)]
    for p in procs:
        p.start()
    got = dict(q.get() for _ in procs)
    for p in procs:
        p.join()
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    n = torch.cuda.device_count()
    print(f"{n} GPUs, {MB} MB pinned copies, GB/s (GiB/s) host->device / device->host")
    for busy in (False, True):
        label = "busy" if busy else "idle"
        seen: dict[int, str] = {}
        for i in range(n):
            mid = {}
            t = threading.Timer(0.8 if busy else 0.3, lambda: mid.update(links()))
            t.start()
            h2d, d2h = measure(i, busy)
            t.join()
            seen[i] = mid.get(i, "?")
            print(f"  {label} GPU {i} alone: {h2d:6.2f} / {d2h:6.2f}   link during copy: {seen[i]}", flush=True)
        got = concurrent(n, busy)
        total = sum(v[0] for v in got.values()), sum(v[1] for v in got.values())
        print(f"  {label} all {n} at once: " + "  ".join(f"{i}:{v[0]:.1f}/{v[1]:.1f}" for i, v in sorted(got.items()))
              + f"   total {total[0]:.1f} / {total[1]:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
