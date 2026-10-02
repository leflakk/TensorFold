"""Capture static-buffer forwards by window size, context bucket and DeltaNet buffer parity, staging new inputs before replay with the eager path's kernels and launch parameters."""

from __future__ import annotations

import torch

from .forward import compute, stage
from .mtp import mtp_compute, mtp_stage


class Graphs:
    def __init__(self, e, *, max_rows: int = 8) -> None:
        self.e = e
        self.max_rows = max_rows
        self.main: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.mtp: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mtp_out: dict[tuple[int, int], torch.Tensor] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.captures = 0

    def _capture(self, fn) -> torch.cuda.CUDAGraph:
        import gc

        torch.cuda.synchronize()
        gc.collect()
        g = torch.cuda.CUDAGraph()
        # Collecting old graphs calls cuGraphExecDestroy and invalidates an active capture.
        enabled = gc.isenabled()
        gc.disable()
        try:
            # thread-local: NCCL's helper threads (tensor parallel) may call CUDA while this thread captures
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                fn()
        finally:
            if enabled:
                gc.enable()
        torch.cuda.synchronize()
        self.captures += 1
        return g

    def _bucket(self, end: int) -> int:
        return min(self.e.st.capacity, max(8192, 1 << (end - 1).bit_length()))

    @torch.no_grad()
    def forward(self, tokens) -> torch.Tensor:
        e = self.e
        w, st, b = e.w, e.st, e.buf
        segs = stage(w, b, [(st, tokens)])
        R = segs[-1][2]
        if R > self.max_rows:
            return compute(w, segs, b)
        context = self._bucket(st.pos + R)
        key = (R, st.cur[0] if st.cur else 0, context)
        g = self.main.get(key)
        if g is None:
            compute(w, segs, b, context=context)     # eager warm-up: compiles this launch shape
            g = self._capture(lambda: compute(w, segs, b, context=context))
            self.main[key] = g
        if _kernel_profile(w, R, lambda: compute(w, segs, b, context=context), g, "verify"):
            return b.logits[:R]
        g.replay()
        return b.logits[:R]

    @torch.no_grad()
    def mtp_forward(self, next_tokens, streams: torch.Tensor) -> torch.Tensor:
        e = self.e
        w, st, b = e.w, e.st, e.mbuf
        segs = mtp_stage(w, b, [(st, next_tokens, streams)])
        n = segs[-1][2]
        if n > self.max_rows:
            return mtp_compute(w, segs, b)
        context = self._bucket(st.mtp_len + n)
        key = (n, context)
        g = self.mtp.get(key)
        if g is None:
            out = mtp_compute(w, segs, b, context=context)     # eager warm-up; its result is the view replays fill
            g = self._capture(lambda: mtp_compute(w, segs, b, context=context))
            self.mtp[key] = g
            self.mtp_out[key] = out
        if _kernel_profile(w, n, lambda: mtp_compute(w, segs, b, context=context), g, "mtp"):
            return self.mtp_out[key]
        g.replay()
        return self.mtp_out[key]

    @torch.no_grad()
    def warm(self, rows: int | None = None) -> int:
        """Capture all decode windows at both DeltaNet parities and all MTP step sizes before decoding; this dirties sequence state, so prefill afterward."""

        e = self.e
        st = e.st
        rows = rows or self.max_rows
        saved = list(st.cur)
        before = self.captures
        for parity in (0, 1):
            st.cur = [parity] * len(st.cur)
            for R in range(1, rows + 1):
                self.forward([0] * R)
        st.cur = saved
        if e.mbuf is not None:
            for n in range(1, rows + 1):
                self.mtp_forward([0] * n, e.buf.streams[:n])
        torch.cuda.synchronize()
        return self.captures - before


_SEEN: dict[str, int] = {}


def _kernel_profile(w, rows: int, eager, graph, kind: str) -> bool:
    """TF_PROFILE_KERNELS=K: rank 0 runs its K-th window of 2+ rows (and K-th MTP step) eagerly under the torch
    profiler and prints each kernel's GPU time (the graph's own time a round is TF_PROFILE_DECODE's). The other
    ranks replay their graphs meanwhile: the same kernels in the same order, so the collectives still pair up (a
    graph replayed on rank 0 alone would not). True when it ran (it computed the window)."""

    import os

    every = int(os.environ.get("TF_PROFILE_KERNELS", "0") or 0)
    if every <= 0 or int(w.meta.get("rank", 0)) != 0 or (kind == "verify" and rows < 2):
        return False
    _SEEN[kind] = _SEEN.get(kind, 0) + 1
    if _SEEN[kind] != every:
        return False
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        eager()
        torch.cuda.synchronize()
    rows_out = []
    for e in prof.key_averages():
        us = next((getattr(e, a) for a in ("self_device_time_total", "self_cuda_time_total", "device_time_total",
                                           "cuda_time_total") if getattr(e, a, 0)), 0)
        if us:
            rows_out.append((us, e.count, e.key))
    rows_out.sort(reverse=True)
    total = sum(u for u, _, _ in rows_out)
    launches = sum(c for _, c, _ in rows_out)
    print(f"[tensorfold] kernel profile ({kind}, {rows} rows): {launches} kernels, {total / 1e3:.2f} ms of GPU time "
          "(eager on rank 0; collectives include waits for the other ranks)", flush=True)
    for us, count, name in rows_out[:40]:
        print(f"[tensorfold]   {us / 1e3:8.3f} ms {count:5d}x {us / max(count, 1):8.1f} us  {name[:110]}", flush=True)
    return True
