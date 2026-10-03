"""Concurrent rounds as CUDA graphs: the verify forward of n streams, and the MTP head's steps.

A round's verify window holds n streams of W = drafts + 1 rows each: a shorter window is padded after its own rows
(padding rows come after a stream's rows in its chain, so its rows keep their bits; their outputs are dropped). Its
shape is then n, the DeltaNet scratch parity, whether last round's rows fold in, and the context bucket that bounds
the attention launches. The tables it reads (rows' streams and positions, cache, state and n-gram tail pointers,
the rows to fold) live in static device buffers refilled before each replay, so one graph serves any n streams.

A shape is captured on its first round, after an eager run of that round compiles its kernels; the DeltaNet states
that run folded last round's rows into are put back first (a fold is not idempotent; everything else a forward
writes, it writes again with the same bits). On tensor-parallel ranks every rank meets the same shapes at the same
rounds, so their captures and replays pair their collectives as the eager rounds do.
"""

from __future__ import annotations

import gc
import os

import torch

from .forward import compute
from .mtp import mtp_compute
from .static_tables import StaticTables


def enabled(tp: bool) -> bool:
    """Concurrent rounds as graphs: on tensor-parallel ranks by default (eager rounds cost 4x there: 44 ms of launches
    a round on 8 RTX 3090s); TF_MULTI_GRAPHS=1 or 0 forces them on or off."""

    value = os.environ.get("TF_MULTI_GRAPHS", "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"TF_MULTI_GRAPHS={value!r}: 0 or 1")
    return tp if value == "" else value == "1"


class RoundGraphs:
    """A concurrent decoder's captured verify forwards and MTP steps, by shape; at most ``TF_MULTI_GRAPHS_MAX``
    (default 96) graphs, the oldest dropped past it. ``capture=False``: the same padded windows and static tables,
    run eager (ranks that are threads of one process, whose collectives sync the host, cannot be captured)."""

    def __init__(self, dec, *, capture: bool = True) -> None:
        self.dec, self.capture = dec, bool(capture)
        self.width = dec.depth + 1                       # W: every stream's verify window, padded
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.outs: dict[tuple, torch.Tensor] = {}
        self.statics: dict[tuple, StaticTables] = {}
        self.captures = self.replays = 0
        self.most = int(os.environ.get("TF_MULTI_GRAPHS_MAX", "96"))
        # the up projection's per-stream terms, sized for the widest window now: a graph holds their address, so a
        # later, wider window must not reallocate them (``qmm.hc_upmix`` grows its scratch on demand)
        c = dec.w.cfg
        for b in (dec.buf, dec.mbuf):
            if b is not None:
                dev = b.mixed.device                     # with its index, as ``hc_upmix`` keys it
                b.__dict__.setdefault("hc_scratch", {})[(dev, c.streams, c.hidden)] = torch.empty(
                    (c.streams * max(b.rows, 16) * c.hidden,), dtype=torch.bfloat16, device=dev)

    def static(self, key: tuple) -> StaticTables:
        got = self.statics.get(key)
        if got is None:
            got = self.statics[key] = StaticTables(self.dec.w.device)
        return got

    def bucket(self, end: int) -> int:
        """The keys a captured step's attention covers: a power of two from 8192, at most the window."""

        return min(self.dec.capacity, max(8192, 1 << max(0, int(end) - 1).bit_length()))

    def _capture(self, fn) -> torch.cuda.CUDAGraph:
        torch.cuda.synchronize()
        gc.collect()
        g = torch.cuda.CUDAGraph()
        enabled = gc.isenabled()
        gc.disable()                             # collecting old graphs destroys their execs mid-capture
        try:
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                fn()
        finally:
            if enabled:
                gc.enable()
        torch.cuda.synchronize()
        self.captures += 1
        while len(self.graphs) >= self.most:     # the oldest shape goes (captured again if it comes back)
            old = next(iter(self.graphs))
            del self.graphs[old]
            self.outs.pop(old, None)
        return g

    def verify(self, segs, tables, step) -> torch.Tensor:
        """The round's forward (``buf.gdn_tables``/``attn_step`` set to ``tables``/``step``): logits [R, V]."""

        d = self.dec
        w, b = d.w, d.buf
        if not self.capture:
            return compute(w, segs, b)
        key = ("verify", len(segs), tables.cur, tables.folds, step.context)
        g = self.graphs.get(key)
        if g is None:
            saved = []
            if tables.folds:                     # the states the eager run folds last round's rows into
                for st, _, _ in segs:
                    for li in range(len(st.cur)):
                        state = st.rec[st.cur[li], li]
                        saved.append((state, state.clone()))
            compute(w, segs, b)                  # eager: compiles this shape's launches
            for state, copy in saved:
                state.copy_(copy)
            del saved
            g = self._capture(lambda: compute(w, segs, b))
            self.graphs[key] = g
        g.replay()
        self.replays += 1
        return b.logits[:segs[-1][2]]

    def mtp(self, segs, step) -> torch.Tensor:
        """An MTP step over ``segs`` (``mbuf.attn_step`` set to ``step``): each stream's draft-head logits."""

        d = self.dec
        w, b = d.w, d.mbuf
        if not self.capture:
            return mtp_compute(w, segs, b, pick=True)
        key = ("mtp", len(segs), segs[-1][2], step.context)
        g = self.graphs.get(key)
        if g is None:
            out = mtp_compute(w, segs, b, pick=True)     # eager (it writes the MTP cache rows a replay writes again)
            g = self._capture(lambda: mtp_compute(w, segs, b, pick=True))
            self.graphs[key] = g
            self.outs[key] = out
        g.replay()
        self.replays += 1
        return self.outs[key]
