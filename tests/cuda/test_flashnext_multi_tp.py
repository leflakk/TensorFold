"""Flash Next's concurrent decoder on tensor-parallel ranks, simulated as threads on one GPU (one RTX 3090 is enough).

``test_flashnext_tp_wide``'s checkpoint (the real expert width and heads) on 4 ranks. Streams decoded together emit
each one's solo run on the one-stream engine, on every rank; then rank 0 serves them through a ``Scheduler`` and a
``Leader`` while the other ranks replay its calls (``multi_tp.follow``), with a client that leaves mid-reply. Rounds
run eager, with windows padded as captured rounds pad them (``graphs="pad"``) or not: these ranks' collectives sync
the host, which a capture forbids (``test_flashnext_multi_graphs`` captures on one GPU).
"""

import queue
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import test_flashnext_pair_prefill as pair  # noqa: E402
import test_flashnext_tp as base  # noqa: E402
import test_flashnext_tp_wide as wide  # noqa: E402

from tensorfold.cuda.scheduler import Scheduler  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import multi_tp  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import load  # noqa: E402

checkpoint = wide.checkpoint          # the module fixture (640-wide experts)
WORLD = 4
COUNT = 20
PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]


def _long(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, base.V, (n,), generator=g).tolist()


def _ranks(checkpoint, comm: str, vocab: bool):
    hub = base._Hub(WORLD)
    kw = {"draft_vocab": str(checkpoint / "draft_ids.txt")} if vocab else {}
    ws = []
    for r in range(WORLD):
        w = load(checkpoint, tp=(r, WORLD), **kw)
        w.comm = (pair._FastThreadComm if comm == "fast" else base._ThreadComm)(hub, r)   # before any buffers
        ws.append(w)
    return hub, ws


def _threads(hub, fns) -> list:
    """fns[r]() on its own thread for every rank; results in rank order (the first error re-raised)."""

    results, errors = [None] * len(fns), []

    def body(r):
        try:
            with torch.no_grad():
                results[r] = fns[r]()
        except BaseException as exc:                # noqa: BLE001  (reported below; unblock the other ranks)
            errors.append(exc)
            hub.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(len(fns))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results


def _solo(w, prompts, samplings) -> list[list[int]]:
    """Each prompt alone on the one-stream engine: its first token, then serial decoding."""

    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    out = []
    for prompt, sampling in zip(prompts, samplings):
        out.append(serial_decode(e, prefill(e, prompt, sampling), COUNT, sampling).tokens)
    return out


# (collectives, draft vocabulary, expert K splits, drafts a round, padded windows): 4 streams of 6 rows take the
# split gate/up past 16 rows (the 1024-wide test model takes none unless TF_EXPERT_SPLITS asks)
@pytest.mark.parametrize("comm,vocab,splits,depth,graphs", [("fast", False, "", 3, "pad"),
                                                            ("fast", True, "4", 5, "pad"),
                                                            ("fast", False, "", 3, False),
                                                            ("gather", False, "", 3, "pad")])
def test_streams_decoded_together_on_ranks_equal_each_alone(checkpoint, monkeypatch, comm, vocab, splits, depth,
                                                            graphs):
    monkeypatch.setenv("TF_EXPERT_SPLITS", splits)
    hub, ws = _ranks(checkpoint, comm, vocab)
    prompts = PROMPTS[:3] + [_long(70, 11)]                  # the last fills over five 16-row passes
    refs = _threads(hub, [lambda w=w: _solo(w, prompts, SAMPLINGS) for w in ws])
    assert all(ref == refs[0] for ref in refs)               # every rank samples the same tokens

    def body(w):
        dec = MultiDecoder(w, slots=4, capacity=1024, depth=depth, confidence=0.3, prefill_rows=16, graphs=graphs)
        assert dec.tp and not dec.converged and dec.buf.moe.plan.split == (int(splits) if splits else 0)
        assert (dec.rounds is not None) == bool(graphs) and (dec.rounds is None or not dec.rounds.capture)
        streams = [Stream(p, COUNT, smp, draft=i != 3, stop_eos=False)           # as serial_decode: past an end
                   for i, (p, smp) in enumerate(zip(prompts, SAMPLINGS))]
        dec.admit(streams[0])
        dec.finish(dec.round())                              # alone: its prompt, then a round
        for s in streams[1:]:
            dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return [s.out for s in streams], [s.min_rows for s in streams]

    got = _threads(hub, [lambda w=w: body(w) for w in ws])
    for r, (outs, rows) in enumerate(got):
        assert outs == refs[r], r
        assert all(m >= 2 for m in rows[:3]), (r, rows)    # drafting streams verified drafts
    assert all(g[0] == got[0][0] for g in got)
    ws.clear()
    torch.cuda.empty_cache()


class _Queues:
    """``multi_tp.Ring``'s contract for ranks that are threads: one queue a follower."""

    def __init__(self, world: int) -> None:
        self.queues = [queue.Queue() for _ in range(world)]

    def end(self, rank: int):
        outer = self

        class End:
            def send(self, data: bytes) -> None:
                for q in outer.queues[1:]:
                    q.put(bytes(data))

            def recv(self) -> bytes:
                return outer.queues[rank].get(timeout=600)

        return End()


def test_rank_zero_serves_and_the_others_replay_its_calls(checkpoint):
    hub, ws = _ranks(checkpoint, "fast", False)
    prompts = [PROMPTS[0], _long(70, 11), PROMPTS[1], PROMPTS[0] + [7, 8], _long(40, 5)]
    samplings = [Sampling(seed=3, top_k=20, top_p=0.95), None, Sampling(seed=4, top_k=20, top_p=0.95), None,
                 Sampling(seed=9, top_k=20, top_p=0.95)]
    refs = _threads(hub, [lambda w=w: _solo(w, prompts, samplings) for w in ws])
    assert all(ref == refs[0] for ref in refs)
    ring = _Queues(WORLD)
    decoders = [MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, prefill_rows=16, graphs="pad")
                for w in ws]
    replies: dict[int, list[int]] = {}

    def serve():
        leader = multi_tp.Leader(decoders[0], ring.end(0))
        sched = Scheduler(leader, max_streams=3)
        assert sched.decoder is leader and not leader.yields

        def client(i):
            got = replies.setdefault(i, [])

            def emit(new):
                got.extend(new)
                return i == 4 and len(got) >= 6              # this client leaves after six tokens

            sched.submit(prompts[i], COUNT, samplings[i], True, emit, stop_eos=False, background=i == 2)

        clients = [threading.Thread(target=client, args=(i,)) for i in range(len(prompts))]
        for th in clients:
            th.start()
        for th in clients:
            th.join()
        sched.close()
        leader.stop()
        return decoders[0].live()

    def replay(r):
        multi_tp.follow(decoders[r], ring.end(r))
        return decoders[r].live()

    left = _threads(hub, [serve] + [lambda r=r: replay(r) for r in range(1, WORLD)])
    assert left == [0] * WORLD                                # every rank finished every stream
    for i, ref in enumerate(refs[0]):
        if i == 4:                                            # the client that left: its first tokens, then none
            assert 6 <= len(replies[i]) < COUNT and replies[i] == ref[:len(replies[i])], replies[i]
        else:
            assert replies[i] == ref, i
    for dec in decoders:
        assert len(dec.free) + len({id(k[1]) for k in dec.kept}) == 3
    decoders.clear()
    ws.clear()
    torch.cuda.empty_cache()
