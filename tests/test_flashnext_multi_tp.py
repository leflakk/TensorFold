"""Rank 0's calls to Flash Next's concurrent decoder reach the other ranks in order (``multi_tp``), on the CPU.

The shared-memory ring carries every message once to each follower (wrapping its slots, spilling one larger than a
slot to the store), and followers replaying a ``Leader``'s messages make the same decoder calls rank 0 made: the
same admissions, rounds (with rank 0's pass rows) and finishes, a client that left ending at the next call.
"""

import threading

import pytest

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda import multi_tp


class _Store:
    """The TCPStore calls the ring makes, for ranks that are threads."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.cv = threading.Condition()

    def set(self, key, value) -> None:
        with self.cv:
            self.data[key] = value.encode() if isinstance(value, str) else bytes(value)
            self.cv.notify_all()

    def get(self, key) -> bytes:
        with self.cv:
            if not self.cv.wait_for(lambda: key in self.data, timeout=60):
                raise TimeoutError(key)
            return self.data[key]

    def wait(self, keys, timeout=None) -> None:
        with self.cv:
            if not self.cv.wait_for(lambda: all(k in self.data for k in keys), timeout=60):
                raise TimeoutError(keys)

    def delete_key(self, key) -> bool:
        with self.cv:
            return self.data.pop(key, None) is not None


def _rings(world: int, store: _Store, tag: str) -> list:
    rings: list = [None] * world
    threads = [threading.Thread(target=lambda r=r: rings.__setitem__(r, multi_tp.Ring(store, r, world, tag=tag,
                                                                                      timeout=60)))
               for r in range(world)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert all(rings)
    return rings


def _run(fns) -> list:
    results, errors = [None] * len(fns), []

    def body(i):
        try:
            results[i] = fns[i]()
        except BaseException as exc:            # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=body, args=(i,)) for i in range(len(fns))]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
    if errors:
        raise errors[0]
    return results


def test_every_follower_reads_every_message_once_in_order():
    store = _Store()
    rings = _rings(3, store, "order")
    big = bytes(range(256)) * (multi_tp.SLOT // 256 + 3)          # past a slot: through the store
    messages = [f"m{i}".encode() * (i + 1) for i in range(3 * multi_tp.SLOTS)]
    messages[5] = big

    def lead():
        for m in messages:
            rings[0].send(m)

    def follow(r):
        return [rings[r].recv() for _ in messages]

    _, got1, got2 = _run([lead, lambda: follow(1), lambda: follow(2)])
    assert got1 == messages and got2 == messages
    assert not [k for k in store.data if "/spill/" in k]          # each follower deleted its spilled copy
    for ring in rings:
        ring.close()


class _Decoder:
    """The calls a Leader forwards, logged; a round gives every live stream its next token."""

    def __init__(self) -> None:
        self.streams: dict[int, Stream] = {}
        self.filling: list[Stream] = []
        self.calls: list = []
        self.planned = None
        self.next_id = 0
        self.memory_gate = None
        self.rows = 128

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def _pass_rows(self) -> int:
        self.rows += 64                       # rank 0's own clock: the followers must take its value
        return self.rows if self.planned is None else self.planned

    def admit(self, s: Stream) -> None:
        if len(s.prompt) > 8:
            raise ValueError("a prompt past this decoder's window")
        if s.prompt[0] == 99:
            raise NoRoom("waits for memory")
        s.sid, self.next_id = self.next_id, self.next_id + 1
        self.streams[s.sid] = s
        smp = None if s.sampling is None else (s.sampling.seed, s.sampling.top_k)
        self.calls.append(("admit", s.sid, list(s.prompt), s.count, smp, s.draft, s.background))

    def round(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done]
        self.calls.append(("round", self.planned, [s.sid for s in live]))
        done = []
        for s in live:
            s.take([100 + len(s.out)], ())
            if s.done:
                done.append(s)
        return done

    def finish(self, done: list[Stream]) -> None:
        for s in done:
            self.streams.pop(s.sid, None)
        if done:
            self.calls.append(("finish", sorted(s.sid for s in done)))

    def drop(self) -> list[Stream]:
        live = list(self.streams.values())
        self.streams = {}
        self.calls.append(("drop", sorted(s.sid for s in live)))
        return live


def test_followers_replay_rank_zeros_calls_and_a_client_that_left_ends_at_the_next_call():
    store = _Store()
    rings = _rings(3, store, "calls")
    decoders = [_Decoder() for _ in range(3)]
    heard: dict[str, list[int]] = {"a": [], "b": []}

    def lead():
        leader = multi_tp.Leader(decoders[0], rings[0])
        a = Stream([1, 2, 3], 6, Sampling(seed=5, top_k=20), emit=lambda new: (heard["a"].extend(new),
                                                                              len(heard["a"]) >= 2)[1])
        b = Stream([4, 5], 3, None, draft=False, background=True, emit=lambda new: heard["b"].extend(new))
        leader.admit(a)
        leader.admit(b)
        with pytest.raises(ValueError):
            leader.admit(Stream(list(range(20)), 4))                # refused alike on every rank
        with pytest.raises(NoRoom):
            leader.admit(Stream([99], 4))
        rounds = 0
        while leader.live():
            done = leader.round()
            leader.finish(done)
            rounds += 1
            assert rounds < 10
        c = Stream([7], 2, None, emit=lambda new: None)
        leader.admit(c)                                             # the earlier finishes ride with this admission
        assert leader.drop() == [c]                                 # (after an error: rides with the close)
        leader.stop()
        return a, b

    def follow(r):
        multi_tp.follow(decoders[r], rings[r])
        return decoders[r].calls

    (a, b), calls1, calls2 = _run([lead, lambda: follow(1), lambda: follow(2)])
    assert calls1 == decoders[0].calls and calls2 == decoders[0].calls
    rounds = [c for c in decoders[0].calls if c[0] == "round"]
    assert [c[1] for c in rounds] == [192, 256, 320]                # rank 0's pass rows, on every rank
    assert [c[2] for c in rounds] == [[0, 1], [0, 1], [1]]          # a's client left in round 2: gone in round 3
    assert heard["a"] == [100, 101] and a.done and len(a.out) == 2
    assert heard["b"] == [100, 101, 102] and b.done
    assert ("finish", [0, 1]) in decoders[0].calls                  # a (ended) with b (finished), once each
    for ring in rings:
        ring.close()
