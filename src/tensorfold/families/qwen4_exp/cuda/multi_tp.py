"""Flash Next's concurrent decoder on tensor-parallel ranks: rank 0 decides each call, every rank runs it.

Rank 0's ``Scheduler`` drives a ``Leader``. Before each decoder call that runs on the GPU (an admission, a round)
the leader sends the call to the other ranks with what only rank 0 knows: the request, the prompt pass's rows and
the streams that ended since the last message (finished replies, clients that left). Each follower replays the
calls on its own ``MultiDecoder`` in the same order, so every rank takes the same rounds, passes and cache growth
(their collectives pair up) and samples the same tokens from the gathered candidates. Calls without GPU work
(``finish``, ``drop``) ride with the next message.

A client that leaves ends its stream at the next call, not inside the round that emitted to it: a round's later
work (a prompt that ends, then the decode window) must see the same streams on every rank. The stream leaves the
decoder with that call's finished streams, once, on every rank.

Messages go through a ring in shared host memory (the ranks share this host): a few microseconds a round. One that
outgrows a slot (an admission of a prompt of several hundred thousand tokens) goes through the TCP store.
"""

from __future__ import annotations

import json
import mmap
import os
import secrets
import struct
import time
import zlib
from datetime import timedelta
from typing import Any, Callable

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling

SLOTS = 8                    # messages in flight at most (rank 0 waits for the slowest follower past that)
SLOT = 2 << 20               # bytes a slot holds: a 183k-token prompt's admission is about 1.3 MB of JSON
HEAD = 4096                  # the ring's header: each rank's last message read (int64 at 8 x rank)
FIELDS = 32                  # a slot's own header: sequence, length, CRC-32, spilled (int64 each)
SPIN = 20000                 # polls before a wait starts sleeping (a round's next message is usually this close)
KEY = "tensorfold/flashnext/multi"


def _poll(ready: Callable[[], bool], what: str, timeout: float | None) -> None:
    """Wait until ``ready()``: busy polls first, then sleeps that lengthen with the wait (an idle server)."""

    spins, start = 0, None
    while not ready():
        spins += 1
        if spins < SPIN:
            continue
        now = time.monotonic()
        start = now if start is None else start
        waited = now - start
        if timeout is not None and waited > timeout:
            raise TimeoutError(f"waited {waited:.0f}s for {what}")
        time.sleep(0 if waited < 0.01 else 1e-4 if waited < 0.5 else 2e-3)


class Ring:
    """Rank 0 -> ranks 1 .. world-1: message n in slot n % SLOTS, read once by every follower.

    Rank 0 writes a slot's body, then its length, CRC and spill flag, then its sequence number; a follower reads
    the sequence, then the rest, and takes the body once its CRC matches (on a weakly ordered host a read can see
    the new sequence before the body: it reads again). Each follower's last read sequence, in the header, tells
    rank 0 when a slot is free again."""

    def __init__(self, store, rank: int, world: int, *, tag: str = "ring", timeout: float = 600.0) -> None:
        self.store, self.rank, self.world, self.timeout = store, int(rank), int(world), float(timeout)
        if not 1 <= self.world <= HEAD // 8:
            raise ValueError(f"a ring for {self.world} ranks")
        self.size = HEAD + SLOTS * SLOT
        self.key, self.seq = f"{KEY}/{tag}", 0
        if self.rank == 0:
            path = f"/dev/shm/tensorfold_multi_{os.getpid()}_{secrets.token_hex(4)}"
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.ftruncate(fd, self.size)              # zeros: no message yet, every rank has read none
                self.mm = mmap.mmap(fd, self.size)
            finally:
                os.close(fd)
            store.set(f"{self.key}/name", path)
        else:
            path = store.get(f"{self.key}/name").decode()
            fd = os.open(path, os.O_RDWR)
            try:
                self.mm = mmap.mmap(fd, self.size)
            finally:
                os.close(fd)
        store.set(f"{self.key}/opened/{self.rank}", "1")
        if self.rank == 0:                               # every rank mapped it: the name can go (the mappings stay)
            store.wait([f"{self.key}/opened/{r}" for r in range(self.world)], timedelta(seconds=self.timeout))
            os.unlink(path)

    def _read(self, r: int) -> int:
        return struct.unpack_from("<q", self.mm, 8 * r)[0]

    def send(self, data: bytes) -> None:
        """Rank 0: the next message."""

        seq = self.seq + 1
        free = seq - SLOTS                               # the message this slot held last
        if free > 0:
            _poll(lambda: min(self._read(r) for r in range(1, self.world)) >= free,
                  f"every rank to read message {free}", self.timeout)
        at = HEAD + (seq % SLOTS) * SLOT
        spilled = len(data) > SLOT - FIELDS
        body = b"" if spilled else bytes(data)
        if spilled:                                      # one key a follower: each deletes its own
            for r in range(1, self.world):
                self.store.set(f"{self.key}/spill/{seq}/{r}", bytes(data))
        self.mm[at + FIELDS:at + FIELDS + len(body)] = body
        struct.pack_into("<qqq", self.mm, at + 8, len(body), zlib.crc32(body), int(spilled))
        struct.pack_into("<q", self.mm, at, seq)        # last: a follower takes the slot once this lands
        self.seq = seq

    def recv(self) -> bytes:
        """Ranks 1 ..: the next message (waits for it)."""

        seq = self.seq + 1
        at = HEAD + (seq % SLOTS) * SLOT
        while True:
            _poll(lambda: struct.unpack_from("<q", self.mm, at)[0] == seq, f"message {seq} from rank 0", None)
            length, crc, spilled = struct.unpack_from("<qqq", self.mm, at + 8)
            body = bytes(self.mm[at + FIELDS:at + FIELDS + length]) if 0 <= length <= SLOT - FIELDS else b""
            if zlib.crc32(body) == crc and struct.unpack_from("<q", self.mm, at)[0] == seq:
                break
        if spilled:
            key = f"{self.key}/spill/{seq}/{self.rank}"
            body = bytes(self.store.get(key))
            self.store.delete_key(key)
        self.seq = seq
        struct.pack_into("<q", self.mm, 8 * self.rank, seq)
        return body

    def close(self) -> None:
        mm, self.mm = getattr(self, "mm", None), None
        if mm is not None:
            mm.close()


def pack(s: Stream) -> dict:
    """What a follower needs to admit the same request: prompt, count, sampling, rule, priority, grammar."""

    from tensorfold.engine.grammar import pack as grammar

    smp = s.sampling
    return {"prompt": [int(t) for t in s.prompt], "count": int(s.count), "draft": bool(s.draft),
            "stop_eos": bool(s.stop_eos), "background": bool(s.background), "grammar": grammar(s.constraint),
            "sampling": None if smp is None else [int(smp.seed), float(smp.temperature), int(smp.top_k),
                                                  float(smp.top_p), float(smp.min_p)]}


def unpack(body: dict, grammar: Callable[[list[int]], Any] | None = None) -> Stream:
    smp = body["sampling"]
    s = Stream(list(body["prompt"]), int(body["count"]), None if smp is None else Sampling(*smp),
               draft=bool(body["draft"]), stop_eos=bool(body["stop_eos"]), background=bool(body["background"]))
    if body.get("grammar"):
        if grammar is None:
            raise RuntimeError("rank 0 sent a request with a grammar, and this rank compiles none")
        s.constraint = grammar(body["grammar"])                 # compiled here as on rank 0
    return s


def _encode(ops: list) -> bytes:
    return json.dumps(ops, separators=(",", ":")).encode()


class _Deferred:
    """A stream's ``emit`` on rank 0: the tokens go out at once, a stop the client asks for waits for the next call."""

    def __init__(self, leader: "Leader", s: Stream, emit: Callable[[list[int]], bool | None]) -> None:
        self.leader, self.s, self.emit = leader, s, emit

    def __call__(self, new: list[int]) -> bool:
        if self.emit(new):
            self.leader.stops.append(self.s)
        return False


class Leader:
    """The ``Scheduler``'s decoder on rank 0: each call reaches the other ranks before it runs here."""

    # background streams keep their lane (a replay prefills again, and a resumed prompt may round differently
    # from a fresh one: its tokens could then differ from those it sent, which ends it on rank 0 alone)
    yields = False

    def __init__(self, decoder, ring: Ring) -> None:
        self.d, self.ring = decoder, ring
        decoder.clock = True                     # its passes' times size the next ones (--decode-share)
        self.pending: list = []                  # calls without GPU work, sent with the next message
        self.stops: list[Stream] = []            # streams whose client left, ended at the next call
        self.ended: list[Stream] = []            # those ended, returned (to be finished) by the next round
        self.broken: BaseException | None = None
        self.arrived = lambda: False             # the Scheduler sets it; the ranks' decoder fills a pass a call

    @property
    def streams(self) -> dict:
        return self.d.streams

    @property
    def memory_gate(self):
        return self.d.memory_gate

    def live(self) -> int:
        return self.d.live()

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the tensor-parallel ranks are out of step after an error in a round "
                               f"({self.broken!r}); restart the server") from self.broken

    def _send(self, op: list) -> None:
        self.ring.send(_encode([*self.pending, op]))
        self.pending = []

    def _stop_left(self) -> None:
        """End the streams whose client left, at this call on every rank; the next round returns them, and their
        finish (with that round's finished streams) frees their slots."""

        gone = []
        for s in self.stops:
            if not s.done and self.d.streams.get(s.sid) is s and all(g is not s for g in gone):
                s.done, s.finished = True, time.perf_counter()
                gone.append(s)
        self.stops = []
        if gone:
            self.pending.append(["end", [int(s.sid) for s in gone]])
            self.ended += gone

    def admit(self, s: Stream) -> None:
        self._check()
        if s.vision is not None:
            raise ValueError("image input on Flash Next runs on one GPU")
        if s.probabilities is not None:
            raise ValueError("logprobs are supported on one GPU only")
        if s.emit is not None:
            s.emit = _Deferred(self, s, s.emit.emit if isinstance(s.emit, _Deferred) else s.emit)
        self._stop_left()
        self._send(["admit", pack(s)])
        try:
            self.d.admit(s)
        except (NoRoom, ValueError):                     # the same on every rank: the request waits or fails
            raise
        except BaseException as exc:
            self.broken = exc
            raise

    def round(self) -> list[Stream]:
        self._check()
        self._stop_left()
        rows = int(self.d._pass_rows())                  # timed on rank 0: every rank fills the same rows
        self._send(["round", rows])
        self.d.planned = rows
        try:
            done = self.d.round()
        except BaseException as exc:
            self.broken = exc
            raise
        finally:
            self.d.planned = None
        ended, self.ended = self.ended, []
        return ended + done

    def finish(self, done: list[Stream]) -> None:
        if done:
            self.pending.append(["finish", [int(s.sid) for s in done]])
        self.d.finish(done)

    def drop(self) -> list[Stream]:
        self.pending.append(["drop"])
        ended, self.stops, self.ended = self.ended, [], []
        return ended + self.d.drop()

    def stop(self) -> None:
        """The followers leave ``follow`` (rank 0 is closing)."""

        if self.ring is not None:
            self._send(["close"])
            self.ring = None


def follow(decoder, ring: Ring, grammar: Callable[[list[int]], Any] | None = None) -> None:
    """Ranks 1 ..: replay rank 0's calls on this rank's decoder, in order, until rank 0 stops."""

    known: dict[int, Stream] = {}
    while True:
        for op in json.loads(ring.recv()):
            kind = op[0]
            if kind == "close":
                return
            if kind == "end":                            # clients that left: done from this call on
                for sid in op[1]:
                    if sid in known:
                        known[sid].done = True
            elif kind == "finish":
                decoder.finish([known.pop(sid) for sid in op[1] if sid in known])
            elif kind == "drop":
                decoder.drop()
                known.clear()
            elif kind == "admit":
                s = unpack(op[1], grammar)
                try:
                    decoder.admit(s)
                except (NoRoom, ValueError):             # rank 0's admission raised the same
                    continue
                known[s.sid] = s
            elif kind == "round":
                decoder.planned = int(op[1])
                try:
                    decoder.round()
                finally:
                    decoder.planned = None
            else:
                raise RuntimeError(f"rank 0 sent an unknown call {kind!r}")
