"""Concurrent rounds as CUDA graphs: the verify forward of n streams, and the MTP head's steps.

A round's window holds n streams' own rows (each stream its token and drafts), the total rounded up to a multiple
of STEP by rows after the last stream's own (padding comes after a stream's rows in its chain, so its rows keep
their bits; their outputs are dropped). Its shape is then n, its rows, the DeltaNet scratch parity and the context
bucket that bounds the attention launches; last round's rows always fold in (none for a new stream). The tables it
reads (rows' streams and positions, cache, state and n-gram tail pointers, the rows to fold) live in static device
buffers refilled before each replay, so one graph serves any n streams holding those rows.

``warm`` captures every shape of the 8,192-key bucket before a request, on idle slots (a graph does not depend on
which streams it reads); a longer context's bucket is captured on its first round, after an eager run of that round
compiles its kernels, the DeltaNet states it folded into put back first (a fold is not idempotent; everything else
a forward writes, it writes again with the same bits). On tensor-parallel ranks every rank meets the same shapes at
the same rounds, so their captures and replays pair their collectives as the eager rounds do.
"""

from __future__ import annotations

import gc
import os

import torch

from .forward import compute
from .graphs import _kernel_profile
from .mtp import mtp_compute
from .static_tables import StaticTables

STEP = 4                 # a round's rows are a multiple of this (fewer shapes; at most STEP - 1 padding rows)


def enabled(tp: bool) -> bool:
    """Concurrent rounds as graphs: on tensor-parallel ranks by default (eager rounds cost 4x there: 44 ms of launches
    a round on 8 RTX 3090s); TF_MULTI_GRAPHS=1 or 0 forces them on or off."""

    value = os.environ.get("TF_MULTI_GRAPHS", "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"TF_MULTI_GRAPHS={value!r}: 0 or 1")
    return tp if value == "" else value == "1"


class RoundGraphs:
    """A concurrent decoder's captured verify forwards and MTP steps, by shape; at most ``TF_MULTI_GRAPHS_MAX``
    (default 160) graphs, the oldest dropped past it. ``capture=False``: the same rows and static tables, run eager
    (ranks that are threads of one process, whose collectives sync the host, cannot be captured)."""

    def __init__(self, dec, *, capture: bool = True) -> None:
        self.dec, self.capture = dec, bool(capture)
        self.width = dec.depth + 1                       # W: the most rows a stream's own window holds
        self.bound = self.width + STEP - 1               # ... and with the round's padding after them
        self.pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.outs: dict[tuple, torch.Tensor] = {}
        self.statics: dict[tuple, StaticTables] = {}
        self.captures = self.replays = 0
        self.limit = int(os.environ.get("TF_MULTI_GRAPHS_MAX", "160"))
        # the up projection's per-stream terms, sized for the widest window now: a graph holds their address, so a
        # later, wider window must not reallocate them (``qmm.hc_upmix`` grows its scratch on demand)
        c = dec.w.cfg
        for b in (dec.buf, dec.mbuf):
            if b is not None:
                dev = b.mixed.device                     # with its index, as ``hc_upmix`` keys it
                b.__dict__.setdefault("hc_scratch", {})[(dev, c.streams, c.hidden)] = torch.empty(
                    (c.streams * max(b.rows, 16) * c.hidden,), dtype=torch.bfloat16, device=dev)

    def rows(self, n: int, own: int) -> int:
        """The rows of a window of ``n`` streams holding ``own`` rows of their own: the next multiple of STEP, at
        most n windows of W."""

        return max(own, min(n * self.width, -(-own // STEP) * STEP))

    def pad(self, windows: list) -> list:
        """``windows`` [(state, tokens)] with the last stream's tokens repeated after its own up to the round's rows
        (as many as its cache holds past them: a round short of its multiple is a shape of its own)."""

        own = sum(len(tokens) for _, tokens in windows)
        st, tokens = windows[-1]
        extra = min(self.rows(len(windows), own) - own, st.capacity - st.pos - len(tokens))
        if extra <= 0:
            return windows
        return [*windows[:-1], (st, list(tokens) + [tokens[-1]] * extra)]

    def static(self, key: tuple) -> StaticTables:
        got = self.statics.get(key)
        if got is None:
            got = self.statics[key] = StaticTables(self.dec.w.device)
        return got

    def bucket(self, end: int) -> int:
        """The keys a captured step's attention covers: a power of two from 8192, at most the window."""

        return min(self.dec.capacity, max(8192, 1 << max(0, int(end) - 1).bit_length()))

    def tables(self, segs, held) -> tuple:
        """A captured round's DeltaNet tables and attention step for ``segs`` (``held``: each stream's rows to fold),
        in the static buffers of its shape (streams, rows): the warm-up's capture and every later round of that shape
        read the same buffers, refilled here."""

        from . import attn_multi, gdn_multi

        d = self.dec
        n, R = len(segs), segs[-1][2]
        tables = gdn_multi.Tables(d.w, d.gdn, segs, held, static=self.static(("gdn", n, R)), width=self.width,
                                  folds=True)
        step = attn_multi.Step(d.w, segs, mtp=False, static=self.static(("attn", n, R)), most=self.bound,
                               context=self.bucket(max(st.pos + a1 - a0 for st, a0, a1 in segs)))
        return tables, step

    def mtp_step(self, segs):
        """A captured MTP step's attention step for ``segs``, in the static buffers of its shape (streams, rows)."""

        from . import attn_multi

        n, R = len(segs), segs[-1][2]
        return attn_multi.Step(self.dec.w, segs, mtp=True, static=self.static(("mtp", n, R)), most=self.bound,
                               context=self.bucket(max(st.mtp_len + a1 - a0 for st, a0, a1 in segs)))

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
        while len(self.graphs) >= self.limit:    # the oldest shape goes (captured again if it comes back)
            old = next(iter(self.graphs))
            del self.graphs[old]
            self.outs.pop(old, None)
        return g

    def verify(self, segs, tables, step) -> torch.Tensor:
        """The round's forward (``buf.gdn_tables``/``attn_step`` set to ``tables``/``step``): logits [R, V]."""

        d = self.dec
        w, b = d.w, d.buf
        R = segs[-1][2]
        if not self.capture:
            return compute(w, segs, b)
        key = ("verify", len(segs), R, tables.cur, tables.folds, step.context)
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
        # TF_PROFILE_KERNELS=K: rank 0 runs its K-th round of each stream count eagerly under the profiler (the other
        # ranks replay the same kernels in the same order, so the collectives pair up)
        if _kernel_profile(w, R, lambda: compute(w, segs, b), g, f"concurrent verify, {len(segs)} streams"):
            return b.logits[:R]
        g.replay()
        self.replays += 1
        return b.logits[:R]

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
        if _kernel_profile(w, segs[-1][2], lambda: mtp_compute(w, segs, b, pick=True), g,
                           f"concurrent MTP step, {len(segs)} streams"):
            return self.outs[key]
        g.replay()
        self.replays += 1
        return self.outs[key]

    def warm(self, states: list) -> None:
        """Capture every verify shape (1 to ``len(states)`` streams, each row count, both DeltaNet parities) and
        every MTP step shape of the 8,192-key bucket, on ``states`` (idle slots, emptied afterwards): a graph reads
        its streams by table, so any streams of a shape replay it."""

        if not self.capture:
            return
        from .forward import stage
        from .mtp import mtp_stage

        d = self.dec
        w = d.w
        saved_parity = d.gdn.parity
        try:
            for n in range(1, len(states) + 1):
                shapes = sorted({self.rows(n, own) for own in range(n, n * self.width + 1)})
                for R in shapes:
                    for parity in (0, 1):
                        d.gdn.parity = parity
                        segs = stage(w, d.buf, _split(states[:n], R))
                        tables, step = self.tables(segs, [[] for _ in segs])
                        d.buf.gdn_tables, d.buf.attn_step = tables, step
                        try:
                            self.verify(segs, tables, step)
                        finally:
                            d.buf.gdn_tables = d.buf.attn_step = None
                if d.mbuf is None:
                    continue
                for R in sorted(set(shapes) | {self.rows(n, n)}):      # absorbs of R rows; a draft step's rows
                    windows, a0 = [], 0
                    for st, tokens in _split(states[:n], R):
                        windows.append((st, tokens, d.mbuf.mtp_in[a0:a0 + len(tokens)]))
                        a0 += len(tokens)
                    segs = mtp_stage(w, d.mbuf, windows)
                    step = d.mbuf.attn_step = self.mtp_step(segs)
                    try:
                        self.mtp(segs, step)
                    finally:
                        d.mbuf.attn_step = None
        finally:
            d.gdn.parity = saved_parity
            for st in states:
                st.reset(w)
            torch.cuda.synchronize()


def _split(states: list, rows: int) -> list:
    """``rows`` rows of token 1 over ``states``, as evenly as they go (a synthetic window)."""

    n = len(states)
    return [(st, [1] * (rows // n + (i < rows % n))) for i, st in enumerate(states)]
