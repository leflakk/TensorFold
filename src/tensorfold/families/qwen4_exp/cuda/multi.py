"""Flash Next's concurrent rounds on one GPU or tensor-parallel ranks: every stream keeps exactly its own accepted
prefix."""

from __future__ import annotations

import time

import numpy as np
import torch

from tensorfold.cuda.logprobs import capture

from tensorfold.cuda.capacity import available_bytes
from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.memory_gate import AgreedGate, MemoryGate, NoRoom, torch_live
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from .decode import (PAIR_MIN, PREFILL_ROWS, WARM_TAIL, Engine, _Profile, _gathered_fits, choose_host_rows, draft,
                     entry_end, gathered_host, pair_prefill, prefill_begin, prefill_chunk, tp_sample_rows)
from . import attn_multi, gdn_multi, image_rows, prefixes, round_graphs
from .forward import Cut, Overlap, commit, compute, compute_mixed, converges, cut_snapshot, stage
from .mtp import mtp_compute, mtp_stage
from .prompt_plan import pass_limit
from .state import ENDS, Buffers, State
from ..cuda import CONFIDENCE, DEPTH

FIRST, STEP = 256, 8192          # rows an idle slot keeps; rows a stream's caches grow by at a time
GIB = 1024**3
SHARE = 0.0                      # --decode-share: a round alone takes this share of its pass's time (0: whole passes)
PASS_MIN = 128                   # the fewest prompt rows a round's pass takes
FILL_GUARD = 8                   # a prompt passed over this many passes takes the next one (no starvation), as on Macs


def _agree(w):
    """Tensor parallel: every rank's verdict -> True only if each rank's is (one small gather on every rank)."""

    world, device = int(w.meta["world"]), w.device

    def agree(ok: bool) -> bool:
        mine = torch.tensor([int(bool(ok))], dtype=torch.int64, device=device)
        every = torch.empty((world,), dtype=torch.int64, device=device)
        w.comm.all_gather(mine, every)
        return bool(int(every.min()))

    return agree


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int, prefill_rows: int,
          pbuf2: Buffers | None = None, overlap=None) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs); ``pbuf2`` and
    ``overlap``: a lone prompt's pass in two overlapped halves, as the one-stream engine's chunks run."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, prefill_rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    e.pbuf2, e.overlap = pbuf2, overlap
    e.stops = ()
    return e


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context.

    Tensor parallel (``w.comm``): every rank runs this decoder on the same calls in the same order (rank 0 decides
    them, ``multi_tp``), so each rank takes the same rounds, passes and growth: rows sample from every rank's
    gathered candidates, the memory gate's answers are agreed by all ranks, and a call fills at most one prompt
    pass (a request waiting to be admitted is rank 0's to see, between calls)."""

    tp = False                                       # tensor parallel (``w.comm``)
    planned: int | None = None                       # tensor parallel: the pass rows rank 0 chose for this round
    clock = False                                    # rank 0 of tensor-parallel ranks: times passes alone (share)

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH,
                 confidence: float | tuple[float, ...] = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", prefill_rows: int = PREFILL_ROWS,
                 share: float = SHARE, points=None, vision=None, workspace_bytes: int = 0,
                 graphs: bool | str | None = None) -> None:
        self.tp = w.comm is not None
        if self.tp and vision is not None:
            raise ValueError("image input on Flash Next runs on one GPU")
        self.w, self.depth, self.capacity = w, depth, capacity
        # where a draft chain stops: one cut, or one by the streams of the round that verifies the drafts (the n-th
        # for n streams, the last for more: a round's rows wait for each other, so a doubtful draft is worth less
        # the more streams share it); every rank counts the same streams
        self.cuts = tuple(map(float, confidence)) if isinstance(confidence, (tuple, list)) else (float(confidence),)
        self.confidence = self.cuts[0]
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.vision = vision
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows, self.slots = slots * (depth + 1), slots
        # a round's window and a prompt pass share each layer's expert launch: the pass's buffers hold both
        self.converged, self.prefill_rows = converges(w), prefill_rows
        # rounds beside a filling prompt size its pass so decoding keeps ``share`` of the pass's time (0: whole passes)
        self.share, self.round_s, self.row_s = share, None, None
        self.buf = Buffers(w, rows, capacity, moe_prefill=True)
        # tensor parallel: the MTP head's experts take the split prefill form, as the one-stream engine's do
        self.mbuf = Buffers(w, rows, capacity, moe_prefill=True if self.tp else None) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows + (rows if self.converged else 0), capacity, prefill=True)
        # ranks summing through shared memory: a pass of one prompt as two halves whose sums overlap the other's
        # compute (the one-stream engine's ``prefill_chunk``; +33-39% prompt speed on 8 RTX 3090s)
        self.pbuf2 = self.overlap = None
        if pair_prefill(w) and not self.converged:
            self.pbuf2 = Buffers(w, prefill_rows // 2, capacity, prefill=True)
            self.overlap = Overlap(w.comm)
        self.gdn = gdn_multi.Scratch(w, rows)            # every stream's DeltaNet rows, one launch a step
        # rounds and MTP steps as CUDA graphs (``round_graphs``): windows padded to depth + 1 rows a stream
        # (``graphs="pad"``: the padded windows and static tables, run eager)
        use = round_graphs.enabled(self.tp) if graphs is None else bool(graphs)
        capturable = vision is None and getattr(w, "x3", None) is None and all(
            getattr(getattr(getattr(layer, "moe", None), "experts", None), "capturable", True) is not False
            for layer in w.layers)
        self.rounds = (round_graphs.RoundGraphs(self, capture=graphs != "pad")
                       if use and capturable and torch.cuda.is_available() else None)
        self.held: dict[int, list[int]] = {}             # stream id -> last round's kept rows, folded in next round
        self.drafted_at: float | None = None             # TF_PROFILE_DECODE: when the last round's drafts were made
        # slots start small and grow with their stream's context, up to the window, while the gate has room
        self.free = [State(w, min(capacity, FIRST), depth + 1, kv_dtype, limit=capacity) for _ in range(slots)]
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.window_bytes = self.free[0].cache_bytes(capacity)          # one stream's caches at the full window
        free = torch_live(torch, available_bytes) if torch.cuda.is_available() else None
        # the mapped n-gram tables are not held back (they barely fit on a Spark); lookups page from disk instead
        live = free
        room, reserve = live() if live is not None else 1 << 62, max(2 * GIB, workspace_bytes)
        self.memory_gate = (AgreedGate(room, reserve, live, agree=_agree(w)) if self.tp else
                            MemoryGate(room, reserve=reserve, live=live))
        self.streams: dict[int, Stream] = {}
        self.filling: list[Stream] = []                  # admitted, prompts still prefilling (oldest first)
        self.fills: dict[int, list] = {}                 # stream id -> [its engine, drafts?, next row, kept state]
        self.passed: dict[int, int] = {}                 # stream id -> prompt passes since its last rows
        self.arrived = lambda: False                     # a request waits to be admitted (the scheduler sets this)
        self.next_id = 0
        self.draft_host = w.draft_ids.cpu().numpy() if w.draft_ids is not None else None
        self.kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []   # (ids, slot, snapshot, tail)
        self.keep = keep

    def _busy(self) -> set[int]:
        return {id(s.st) for s in [*self.streams.values(), *self.filling]}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st: State, rows: int, *, alone: bool = False, protect: State | None = None) -> bool:
        """Grow caches to hold ``rows`` while the gate has room, kept ends first; ``alone`` grows anyway."""

        if rows <= st.capacity or st.capacity >= st.limit:       # admission's count keeps a stream within its window
            return True
        size = min(st.limit, -(-rows // STEP) * STEP)
        before = st.cache_bytes()
        grow = st.cache_bytes(size) - before
        while not self.memory_gate.fits(grow + st.layer_bytes(size)):     # a layer's old buffers stay until its copy
            if not self._evict_kept(st, protect=protect):
                if alone:
                    break
                return False
        try:
            added = st.resize(size)
        except Exception:
            self.memory_gate.take(st.cache_bytes() - before)
            raise
        self.memory_gate.take(added)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()             # the old buffers back to the system: MemAvailable stays true
        return True

    def _shrink(self, st: State, *, force: bool = False) -> None:
        """An idle slot back to its first rows: its caches' memory returns to the gate."""

        st.reset(self.w)
        if force or st.capacity > FIRST:
            self.memory_gate.give(-st.resize(min(FIRST, st.limit)))

    def _evict_kept(self, keep: State, *, protect: State | None = None) -> bool:
        """Free the oldest idle kept prompt end (never ``keep``); False when none is left."""

        busy = self._busy()
        for ids, st, _, _ in self.kept:
            if st is not keep and st is not protect and id(st) not in busy:
                self._drop_kept(st)
                self._shrink(st)
                if all(f is not st for f in self.free):
                    self.free.append(st)
                return True
        return False

    def _make_room(self) -> list[Stream]:
        """Before a round: grow each live window oldest-first; a stream that can't grow makes the newest end."""

        live = sorted((s for s in self.streams.values() if not s.done), key=lambda s: s.sid)
        blocked = False
        for s in live:
            rows = max(s.st.pos, s.st.mtp_len) + len(s.drafts) + self.depth + 2
            s.waiting = rows > s.st.capacity if blocked else not self._grow(s.st, rows, alone=len(live) == 1)
            blocked = blocked or s.waiting
        if live and live[0].waiting and len(live) > 1:        # even the oldest can't grow: the newest ends
            newest = live[-1]
            newest.error = RuntimeError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, after "
                f"{len(newest.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt or "
                "max_tokens, or start the server with a smaller --parallel.")
            newest.done, newest.waiting = True, False
            self.memory_gate.ends += 1
            self.streams.pop(newest.sid, None)
            self.held.pop(newest.sid, None)
            self._drop_kept(newest.st)
            self._shrink(newest.st)
            self.free.append(newest.st)
            return [newest, *self._make_room()]
        self.memory_gate.waits += any(s.waiting for s in live)
        return []

    def _slot_for(self, prompt: list[int], reuse: bool):
        """Reuse the longest kept point, copying a fork into a free slot when the memory gate permits it."""

        return prefixes.slot_for(self, prompt, reuse)

    def _remember(self, ids: list[int], st: State, snap: dict, tail) -> None:
        prefixes.remember(self, ids, st, snap, tail)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    @torch.no_grad()
    def warm(self) -> None:
        """A synthetic greedy request through prefill (a full chunk, then a partial one cut at the kept point), its drafts and one round, then forgotten, so no request compiles or loads a kernel."""

        self.planned = self.prefill_rows                # whole passes on every rank (no clock decides them here)
        try:
            self._warm()
        finally:
            self.planned = None

    def _warm(self) -> None:
        s = Stream([0] * min(self.prefill_rows + WARM_TAIL + 1, self.capacity - self.depth - 2), 2)
        self.admit(s)
        for _ in range(64):                              # the whole prompt (nothing else decodes), then a round
            if not any(x is s for x in self.filling) and (s.done or s.sid not in self.streams):
                break
            self.round()                                 # tensor parallel: a pass a call
        self.streams.pop(s.sid, None)
        self.held.pop(s.sid, None)
        self._drop_kept(s.st)
        self._shrink(s.st)
        if all(f is not s.st for f in self.free):
            self.free.append(s.st)
        if self.rounds is not None:                      # every shape of rounds and MTP steps, on idle slots
            self.rounds.warm(list(self.free[:self.slots]))

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request in a free slot (a kept prompt end it extends, if any); rounds prefill its prompt."""

        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        if any(x.waiting for x in self.streams.values()):
            raise NoRoom("streams already wait for memory; a new request waits until one finishes")
        t0 = time.perf_counter()
        st, resume, s.cached = self._slot_for(list(s.prompt), s.draft and s.vision is None)
        if not self._grow(st, len(s.prompt) + self.depth + 2, alone=not self.streams and not self.filling):
            if resume is None:
                self.free.append(st)
            else:                                        # the kept prompt end stays kept
                self._remember(list(s.prompt[:s.cached]), st, resume["state"], resume["tail"])
            raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes")
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows, self.pbuf2,
                  self.overlap)
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        try:
            begin = prefill_begin(e, s.prompt, mtp=mtp, resume=resume)
            image_rows.begin(e, s, self.vision)
        except Exception:
            self._drop_kept(st)
            self.free.append(st)
            raise
        e.stops = sorted({p for p in self.points(s.prompt) if begin + MIN_GAP <= p < entry_end(s.prompt)}) \
            if s.draft and st.image_positions is None and self.points is not None else []
        s.sid, s.st = self.next_id, st
        self.next_id += 1
        s.prefill_s = time.perf_counter() - t0
        same = resume is not None and self._keep_at(s) == begin         # the same prompt again: its own point
        self.fills[s.sid] = [e, mtp, begin, (resume["state"], resume["tail"]) if same else None]
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        """Fill ordered prompt pieces until a stream decodes or a waiting request needs admission."""

        ended: list[Stream] = []
        while self.filling:
            ended += self._pass()
            if self.tp or any(not x.done and not x.waiting for x in self.streams.values()) or self.arrived():
                break                                    # tensor parallel: rank 0 sees arrivals between calls
        return ended

    def _pass_rows(self) -> int:
        """A round's prompt rows: its decode (a round alone) takes ``share`` of the pass's time, by the last rounds."""

        if self.planned is not None:                     # tensor parallel: rank 0's choice, on every rank
            return self.planned
        live = any(not s.done for s in self.streams.values())
        return pass_limit(self.prefill_rows, live, self.share, self.round_s, self.row_s, PASS_MIN)

    def _timed(self, seconds: float, rows: int) -> None:
        """A round's wall time: a round alone updates its estimate, a round with a pass the seconds a row adds."""

        if rows:
            extra = max(0.0, seconds - (self.round_s or 0.0)) / rows
            self.row_s = extra if self.row_s is None else 0.7 * self.row_s + 0.3 * extra
        else:
            self.round_s = seconds if self.round_s is None else 0.7 * self.round_s + 0.3 * seconds

    def _order(self) -> list[Stream]:
        """Order overdue prompts first, then foreground, fewest rows left and stable arrival order."""

        def key(s: Stream):
            passed = self.passed.get(s.sid, 0)
            due = passed >= FILL_GUARD
            return (not due, -passed if due else 0, s.background, len(s.prompt) - self.fills[s.sid][2])

        return sorted(self.filling, key=key)                         # stable: ties keep arrival order

    def _pieces(self, rows: int | None = None) -> list[tuple[Stream, int, int]]:
        """The next pass: rows from the filling prompts in ``_order``, up to ``rows`` and ENDS ending prompts."""

        pieces, room = [], self._pass_rows() if rows is None else rows
        for s in self._order():
            e, mtp, start, _ = self.fills[s.sid]
            n = min(next((p for p in e.stops if p > start), len(s.prompt)) - start, room)
            ends = sum(1 for x, a, k in pieces if a + k == len(x.prompt))
            if n == 0 or (start + n == len(s.prompt) and ends == ENDS):
                break
            pieces.append((s, start, n))
            room -= n
        return pieces

    def _pass(self) -> list[Stream]:
        """One prompt pass alone; prompts that end sample their first token, draft and join the rounds."""

        pieces = self._pieces()
        self._note_passed(pieces)
        t0 = time.perf_counter()
        try:
            if self.pbuf2 is not None and len(pieces) == 1 and pieces[0][2] // 2 >= PAIR_MIN \
                    and pieces[0][0].st.image_positions is None:
                heads, lasts = self._pass_pair(pieces[0])
            else:
                segs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                ends, cuts = self._end_rows(pieces, segs), self._cuts(pieces, segs)
                logits = compute(self.w, segs, self.pbuf, logits=bool(ends), ends=ends, cuts=cuts)
                heads = logits[:len(ends)].clone() if ends else None
                lasts = self._absorb(pieces, segs, cuts)
        except Exception as exc:                         # noqa: BLE001  (these requests fail, the others go on)
            if self.tp:                                  # a rank's own failure: the ranks are no longer in step
                raise
            return self._failed(pieces, exc)
        rows = sum(n for _, _, n in pieces)
        if self.clock and rows and self.share > 0:       # a pass alone: what a prompt row costs (--decode-share)
            torch.cuda.synchronize()
            per = (time.perf_counter() - t0) / rows
            self.row_s = per if self.row_s is None else 0.7 * self.row_s + 0.3 * per
        staying = sum(1 for s in self.streams.values() if not s.done and not s.waiting)
        return self._joined(pieces, heads, lasts, (time.perf_counter() - t0) / len(pieces),
                            self._cut(staying, pieces))

    def _pass_pair(self, piece) -> tuple:
        """A pass of one prompt piece through the one-stream engine's ``prefill_chunk`` (two overlapped halves when
        its kept point allows, the same bits as one chunk): its head (if it ends the prompt) and its last row."""

        s, a, n = piece
        e, mtp = self.fills[s.sid][0], self.fills[s.sid][1]
        k = self._point(s, a)
        keep_at = k if k is not None and a < k <= a + n else None
        e.kept = None
        head = prefill_chunk(e, s.prompt, a, mtp=mtp, keep_at=keep_at, end=a + n)
        if keep_at is not None:                          # the point's state, as ``_absorb`` keeps it
            self.fills[s.sid][3] = (e.kept["state"], e.kept["tail"])
        return head, [e.last_streams]

    def _note_passed(self, pieces) -> None:
        """Count a pass against every filling prompt it left out; one it took starts over."""

        took = {s.sid for s, _, _ in pieces}
        for s in self.filling:
            self.passed[s.sid] = 0 if s.sid in took else self.passed.get(s.sid, 0) + 1
        for sid in [k for k in self.passed if k not in {s.sid for s in self.filling}]:
            del self.passed[sid]

    @staticmethod
    def _end_rows(pieces, segs) -> list[int]:
        """The pass rows that end a prompt (each gets the head)."""

        return [a1 - 1 for (s, a, n), (_, _, a1) in zip(pieces, segs) if a + n == len(s.prompt)]

    @staticmethod
    def _keep_at(s: Stream) -> int | None:
        """Where a drafting stream's prompt state is kept: one token before its end, which a next turn extends."""

        return entry_end(s.prompt) if s.draft and s.st.image_positions is None else None

    def _point(self, s: Stream, start: int) -> int | None:
        """The next message-start or prompt-end snapshot this prompt piece can reach."""

        if not s.draft or s.st.image_positions is not None:
            return None
        return next((p for p in self.fills[s.sid][0].stops if p > start), self._keep_at(s))

    def _cuts(self, pieces, segs) -> list[Cut]:
        """The kept points strictly inside the pass's pieces, where their DeltaNet chains split."""

        return [Cut(k - a, at=a0) for (s, a, n), (_, a0, _) in zip(pieces, segs)
                if (k := self._point(s, a)) is not None and a < k < a + n]

    def _absorb(self, pieces, segs, cuts=()) -> list[torch.Tensor]:
        """After a pass's forward: each prompt's last row and kept point, the MTP head's absorb, the commits."""

        lasts = [self.pbuf.streams[a1 - 1:a1].clone() for _, _, a1 in segs]
        at, points = {cut.at: cut for cut in cuts}, []
        for (s, a, n), (st, a0, _) in zip(pieces, segs):      # before the MTP head writes the pass's streams
            k = self._point(s, a)
            if k is None or not a < k <= a + n:
                continue
            row, mtp = k - a, self.fills[s.sid][1]
            mtp_len = st.mtp_len + row - 1 if mtp else st.mtp_len       # every row but the point's last
            tail = self.pbuf.streams[a0 + row - 1:a0 + row].clone() if mtp else None
            cut = at.get(a0)
            snap = None if cut is None else cut_snapshot(self.w, st, self.pbuf, cut, mtp_len)
            points.append((s, mtp_len, tail, snap))
        absorb = [(s.st, s.prompt[a + 1:a + n + 1], self.pbuf.streams[a0:a0 + n])
                  for (s, a, n), (_, a0, _) in zip(pieces, segs) if self.fills[s.sid][1] and a + 1 < len(s.prompt)]
        if absorb:                   # the MTP head absorbs each prompt's rows (its cache in position order)
            absorb = [(st, nxt, streams[:len(nxt)]) for st, nxt, streams in absorb]
            mtp_compute(self.w, mtp_stage(self.w, self.pbuf, absorb), self.pbuf)
            for st, nxt, _ in absorb:
                st.set_mtp_len(st.mtp_len + len(nxt))
        for (s, a, n), (st, a0, _) in zip(pieces, segs):
            commit(self.w, st, self.pbuf, n, n, at=a0)
        for s, mtp_len, tail, snap in points:            # a point that ends its piece: the state as committed
            self.fills[s.sid][3] = (snap if snap is not None else {**s.st.snapshot(), "mtp_len": mtp_len}, tail)
        return lasts

    def _failed(self, pieces, exc: Exception) -> list[Stream]:
        failed = [s for s, _, _ in pieces]
        for s in failed:
            s.error, s.done = exc, True
            self.filling.remove(s)
            self.fills.pop(s.sid)
            self._drop_kept(s.st)
        return failed                                    # finish() frees their slots

    def _cut(self, staying: int, pieces=()) -> float:
        """The cut for drafts the next round verifies: ``staying`` streams go on into it, with the prompts that end
        in ``pieces``."""

        n = staying + sum(1 for s, a, k in pieces if a + k == len(s.prompt))
        return self.cuts[min(max(n, 1), len(self.cuts)) - 1]

    def _joined(self, pieces, heads, lasts, spent: float, cut: float) -> list[Stream]:
        """Prompts that ended sample their first token, draft (``cut``: where a chain stops) and join the rounds;
        returns those already done."""

        joined, head = [], 0
        for (s, a, n), last in zip(pieces, lasts):
            s.prefill_s += spent
            e, mtp, _, kept = self.fills[s.sid]
            self.fills[s.sid][2] = a + n
            if kept is not None and a < kept[0]["pos"] <= a + n:
                self._remember(list(s.prompt[:kept[0]["pos"]]), s.st, *kept)
            self.fills[s.sid][3] = None                    # only the bounded cache owns a stored snapshot
            if a + n < len(s.prompt):
                continue
            self.filling.remove(s)
            self.fills.pop(s.sid)
            st, e.last_streams = s.st, last
            logits = heads[head:head + 1]
            if s.constraint is not None:                 # a reply's grammar: the first token too
                logits = s.constraint.mask(logits, None, self.w.meta.get("vocab_offset", 0))
            first = e.sample(logits, [len(s.prompt)], s.sampling)[0]
            if s.probabilities is not None:
                capture(logits, [first], [len(s.prompt)], s.probabilities)
            if s.constraint is not None:
                s.constraint.advance([first])
            head += 1
            image_rows.finish(st)
            s.context = list(s.prompt)
            s.drafts = draft(e, last, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                             cut) if mtp and s.count > 1 else []
            s.started = time.perf_counter()
            self.streams[s.sid] = s
            s.take([first], self._ends(s))
            if s.done:
                joined.append(s)
        return joined

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams, with the next prompt pass in the same forward while prompts fill."""

        ended = self._make_room()                      # every stream's caches hold this round, or the newest wait
        live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if self.filling and (not live or not self.converged):
            ended += self._fill()                      # passes alone; a prompt that ends here joins this round
            live = [s for s in self.streams.values() if not s.done and not s.waiting]
            self.drafted_at = None                     # (the time since the last round is a pass's, not overhead)
        if not live:
            return ended
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None:
                tokens = [s.out[-1]] + list(s.drafts)
                try:
                    grammars[s.sid] = s.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                except GrammarError as exc:              # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    self.held.pop(s.sid, None)
                    failed.append(s)
                    continue
                s.drafts = grammars[s.sid].tokens[1:]
        live = [s for s in live if not s.done]
        if not live:
            return failed + ended
        prof = _Profile.make(self.w, len(live))          # TF_PROFILE_DECODE=N: where rounds' time goes (rank 0)
        if prof:
            prof.since("outside", self.drafted_at)       # the scheduler, the ring and the replies since last round
        t0 = time.perf_counter()
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        # captured rounds: the round's rows to its shape's (``RoundGraphs.rows``), padding after the last stream's own
        segs = stage(self.w, self.buf, windows if self.rounds is None else self.rounds.pad(windows))
        ranges = [(a0, a0 + len(tokens)) for (_, a0, _), (_, tokens) in zip(segs, windows)]   # each stream's rows
        # a pass shares the round's forward only where their experts share a launch; else _fill ran it between rounds
        pieces, psegs = (self._pieces(self._pass_rows()) if self.filling and self.converged else []), None
        cuts = []
        if pieces:
            self._note_passed(pieces)
            try:
                psegs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                cuts = self._cuts(pieces, psegs)
            except Exception as exc:                     # noqa: BLE001  (the pass's requests fail, the round goes on)
                ended += self._failed(pieces, exc)
                pieces = []
        held = [self.held.pop(s.sid, []) for s in live]
        graphed = self.rounds is not None and not pieces and all(st.image_positions is None for st, _, _ in segs)
        if graphed:                                      # this shape's device tables, refilled
            tables, step = self.rounds.tables(segs, held)
        else:
            tables = gdn_multi.Tables(self.w, self.gdn, segs, held)
            step = attn_multi.Step(self.w, segs, mtp=False)
        self.buf.gdn_tables, self.buf.attn_step = tables, step
        if prof:
            prof.mark("host")
        try:
            if pieces:                                 # the window and the pass: each layer's experts once for both
                pends = self._end_rows(pieces, psegs)
                logits, heads = compute_mixed(self.w, segs, self.buf, psegs, self.pbuf, ends=pends, cuts=cuts)
                heads = heads[:len(pends)].clone() if pends else None
            elif graphed:                              # a captured shape (a new stream's round folds nothing)
                logits = self.rounds.verify(segs, tables, step)
            else:
                logits = compute(self.w, segs, self.buf)
        finally:
            self.buf.gdn_tables = self.buf.attn_step = None
        if prof:
            prof.mark("verify", segs[-1][2])
            prof.count("streams", len(live))
            prof.count("padding", segs[-1][2] - ranges[-1][1])          # rows past the last stream's own
            if self.rounds is not None and self.rounds.capture:
                prof.gauge("graphs captured", self.rounds.captures)
        lasts = self._absorb(pieces, psegs, cuts) if pieces else None
        offset = int(self.w.meta.get("vocab_offset", 0))                # tensor parallel: this rank's columns
        for s, (a0, a1) in zip(live, ranges):
            if s.sid in grammars:
                s.constraint.mask(logits[a0:a1], grammars[s.sid], offset)
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for (st, _, _), (a0, a1) in zip(segs, ranges)]
        if self.tp:
            sampled = self._tp_sample(logits, ranges, positions, live, grammars, segs[-1][2])
        else:
            sampled = self._sample(logits, ranges, positions, [s.sampling for s in live])
        paths = [accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self._ends(s))
                 for s, (_, tokens), rows in zip(live, windows, sampled)]
        if prof:
            prof.mark("sample")
        for s, (_, tokens), (_, a0, _), (path, end), pos in zip(live, windows, segs, paths, positions):
            if s.probabilities is not None:
                capture(logits, [tokens[r] for r in path[1:]] + [end], [pos[r] for r in path],
                        s.probabilities, rows=[a0 + r for r in path])
        for s, rows in zip(live, gdn_multi.keep(tables, [len(path) for path, _ in paths])):
            self.held[s.sid] = rows                      # the next round's trees fold these rows in first
        kept = []
        for s, (_, tokens), (st, a0, a1), rows, (path, end) in zip(live, windows, segs, sampled, paths):
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0, states=False)
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = [tokens[r] for r in path[1:]] + [end]
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            last = s.error is not None or len(s.out) + len(new) >= s.count or end in self._ends(s)
            kept.append((s, a0, rows[:len(path)], new, last))
        if prof:
            prof.mark("commit", sum(len(path) for path, _ in paths))
        cut = self._cut(sum(1 for *_, last in kept if not last), pieces)      # by the next round's streams
        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last], cut)
        if prof:
            prof.mark("draft", sum(len(s.drafts) for s, _, _, _, last in kept if s.draft and not last))
            prof.round()
            self.drafted_at = None if pieces else time.perf_counter()
        for s, _, _, new, _ in kept:
            if s.error is not None:
                s.done = True
                continue
            s.take(new, self._ends(s))
        spent = time.perf_counter() - t0
        self._timed(spent, sum(n for _, _, n in pieces))
        if pieces:                                     # prompts that ended in this round's pass join the next
            ended += self._joined(pieces, heads, lasts, spent / len(pieces), cut)
        done = [s for s in live if s.done]
        for s in done:                                   # a finished stream's state is never read again
            self.held.pop(s.sid, None)
        return failed + done + ended

    def _draft_all(self, streams: list, cut: float) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one step a depth; a chain
        stops before a later draft under ``cut``."""

        for s, _, _ in streams:
            s.drafts = []
        room = {s.sid: min(self.depth, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
        todo = [(s, a0, keep) for s, a0, keep in streams if room[s.sid] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        if self.rounds is None:
            windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
            segs = mtp_stage(self.w, self.mbuf, windows)
            lasts = [a1 - 1 for _, _, a1 in segs]
        else:                                            # a captured step: its rows to its shape's
            segs, lasts = self._mtp_stage([(s.st, keep, self.buf.streams[a0:a0 + len(keep)])
                                           for s, a0, keep in todo])
        logits = self._mtp(segs)
        for s, _, keep in todo:
            s.st.set_mtp_len(s.st.mtp_len + len(keep))
        active = [(s, row) for (s, _, _), row in zip(todo, lasts)]
        for j in range(self.depth):
            picks = self._picks(logits, [s.st.pos + 1 + j for s, _ in active], [s.sampling for s, _ in active])
            nxt = []
            for (s, row), (d, p) in zip(active, picks):
                low = cut > 0 and p < cut
                if low and j > 0:
                    continue
                s.drafts.append(d)
                if not low and j + 1 < room[s.sid]:
                    nxt.append((s, row, d))
            if not nxt:
                return
            windows = [(s.st, [d], self.mbuf.streams[row:row + 1]) for s, row, d in nxt]
            if self.rounds is None:
                segs = mtp_stage(self.w, self.mbuf, windows)
                lasts = [a0 for _, a0, _ in segs]
            else:
                segs, lasts = self._mtp_stage(windows)
            logits = self._mtp(segs)
            for s, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, row) for (s, _, _), row in zip(nxt, lasts)]

    def _mtp_stage(self, windows: list) -> tuple[list, list[int]]:
        """A captured MTP step's ``mtp_stage``: the streams' rows written into the head's input rows in place, the
        last stream's repeated after its own up to the step's rows (``RoundGraphs.rows``); and each stream's last
        own row."""

        own = sum(len(tokens) for _, tokens, _ in windows)
        last = windows[-1][0]                            # padding no further than its MTP cache holds
        extra = min(self.rounds.rows(len(windows), own) - own, last.capacity - last.mtp_len - len(windows[-1][1]))
        staged, lasts, a0 = [], [], 0
        mtp_in = self.mbuf.mtp_in
        for i, (st, tokens, streams) in enumerate(windows):
            n = len(tokens)
            mtp_in[a0:a0 + n].copy_(streams, non_blocking=True)
            if i == len(windows) - 1 and extra > 0:     # padding rows: the last own row again
                mtp_in[a0 + n:a0 + n + extra].copy_(streams[n - 1:n].expand(extra, -1), non_blocking=True)
                tokens, n = list(tokens) + [tokens[-1]] * extra, n + extra
            staged.append((st, tokens, mtp_in[a0:a0 + n]))
            lasts.append(a0 + len(windows[i][1]) - 1)
            a0 += n
        return mtp_stage(self.w, self.mbuf, staged, lasts=lasts), lasts

    @staticmethod
    def _sample(logits: torch.Tensor, ranges: list, positions: list, samplings: list) -> list[list[int]]:
        """Each stream's rows [a0, a1) sampled at its positions; padded windows' rows go first into one block."""

        starts = [a0 for a0, _ in ranges] + [ranges[-1][1]]
        if any(a1 != b0 for (_, a1), b0 in zip(ranges, starts[1:])):        # padding between streams' rows
            logits = torch.cat([logits[a0:a1] for a0, a1 in ranges])
            starts = [0]
            for a0, a1 in ranges:
                starts.append(starts[-1] + a1 - a0)
        return sample_streams(logits, starts, positions, samplings)

    def _tp_sample(self, logits: torch.Tensor, ranges: list, positions: list, live: list, masked,
                   rows: int) -> list[list[int]]:
        """Tensor parallel: each stream's rows from every rank's candidates, gathered in the forward (one read-back
        of the window's ``rows``), every stream's rows of one sampling rule in one draw; a grammar's masked rows, or a
        sampler they don't cover, draw over the masked shards with a gather of their own (every rank takes the same
        ones, in order)."""

        w, out = self.w, [None] * len(live)
        at, spans, pos, smps = [], [], [], []           # the gathered rows: their streams' spans, positions, samplers
        for i, s in enumerate(live):
            a0, a1 = ranges[i]
            if s.sid in masked or not _gathered_fits(s.sampling):
                out[i] = tp_sample_rows(w, logits[a0:a1], positions[i], s.sampling, offset=int(w.meta["vocab_offset"]))
                continue
            spans.append((i, len(at), a1 - a0))
            at += range(a0, a1)
            pos += positions[i]
            smps += [s.sampling] * (a1 - a0)
        if at:
            got = choose_host_rows(gathered_host(w, self.buf.cand_all, rows), at, pos, smps)
            for i, j, n in spans:
                out[i] = got[j:j + n]
        return out

    def _mtp(self, segs: list) -> torch.Tensor:
        """An MTP step over every drafting stream, its attention one launch a kernel for all of them (a captured
        step of that shape with ``round_graphs``)."""

        if self.rounds is None:
            self.mbuf.attn_step = attn_multi.Step(self.w, segs, mtp=True)
        else:
            self.mbuf.attn_step = self.rounds.mtp_step(segs)
        try:
            if self.rounds is not None:
                return self.rounds.mtp(segs, self.mbuf.attn_step)
            return mtp_compute(self.w, segs, self.mbuf, pick=True)
        finally:
            self.mbuf.attn_step = None

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Each row's keyed draft and its probability at temperature 1, one read-back (drafts change speed only)."""

        if self.tp:
            return self._tp_picks(logits, positions, samplings)
        row = logits.float()
        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])
        k = min(k, row.shape[1])
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        top, col = row.max(dim=-1, keepdim=True)
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        got = torch.cat([vals, idx.float(), top, col.float(), lse], dim=1).cpu().numpy()
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            g = got[i]
            lse_i = float(g[2 * k + 2])
            if smp is None or smp.temperature <= 0:
                c = int(g[2 * k + 1])
                out.append((int(self.draft_host[c]) if self.draft_host is not None else c,
                            float(np.exp(float(g[2 * k]) - lse_i))))
                continue
            cols = g[k:2 * k].astype(np.int64)
            ids = self.draft_host[cols] if self.draft_host is not None else cols
            tok = choose_rows(g[None, :k].astype(np.float32), ids[None, :], [pos], smp)[0]
            hit = np.nonzero(ids == tok)[0]
            out.append((int(tok), float(np.exp(float(g[hit[0]]) - lse_i)) if len(hit) else 0.0))
        return out

    def _tp_picks(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """``_picks`` from every rank's draft candidates, gathered in the MTP step (``mbuf.cand_all``, one read-back),
        the streams of one sampling rule in one draw; a sampler they don't cover draws over the shards with its own
        gather."""

        w, out, fits = self.w, [None] * len(positions), []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            if _gathered_fits(smp):
                fits.append(i)
                continue
            toks, probs = tp_sample_rows(w, logits[i:i + 1], [pos], smp, offset=int(w.meta["vocab_offset"]),
                                         id_map=w.draft_ids, with_prob=True)
            out[i] = (int(toks[0]), float(probs[0]))
        if fits:
            toks, probs = choose_host_rows(gathered_host(w, self.mbuf.cand_all, len(positions)), fits,
                                           [positions[i] for i in fits], [samplings[i] for i in fits], with_prob=True)
            for i, t, p in zip(fits, toks, probs):
                out[i] = (int(t), float(p))
        return out

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt state is kept stays with it, the rest are free again."""

        for s in done:
            self.streams.pop(s.sid, None)
            self.held.pop(s.sid, None)                   # a stream that ended outside a round (a yield, a stop)
            if not any(k[1] is s.st for k in self.kept) and all(f is not s.st for f in self.free):
                self._shrink(s.st)
                self.free.append(s.st)

    def drop(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.filling, self.fills = [], {}
        for s in live:
            self.streams.pop(s.sid, None)
            self.held.pop(s.sid, None)
            self._drop_kept(s.st)
            self._shrink(s.st)
            self.free.append(s.st)
        return live


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))
