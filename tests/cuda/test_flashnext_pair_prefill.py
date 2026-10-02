"""Prompt chunks as two overlapped halves (``compute_pair``) give the whole chunk's bits, on 4 and 8 ranks.

Ranks are threads on one GPU (``test_flashnext_tp``'s checkpoint, real expert width and heads) with an exact,
size-independent rank-ordered sum standing in for the shared-memory collectives, so the only difference between a
paired and a whole prefill is the pairing itself: the first token, the kept state one token before the end, and the
drafted continuation must be the same on every rank.
"""

import random

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import test_flashnext_tp as base  # noqa: E402
import test_flashnext_tp_wide as wide  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, entry_end, mtp_decode, prefill  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import load  # noqa: E402

checkpoint = wide.checkpoint          # the module fixture (640-wide experts)


class _FastThreadComm(base._ThreadComm):
    """The thread comm with ``reduce``: every rank's partial summed in rank order in fp32, any size."""

    fast = True

    def reduce(self, part: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.cuda.current_stream().synchronize()
        self.hub.slots[self.rank] = part
        self.hub.barrier.wait()
        acc = self.hub.slots[0].float().clone()
        for r in range(1, self.world):
            acc = acc + self.hub.slots[r].float()
        out.copy_(acc.to(out.dtype))
        torch.cuda.current_stream().synchronize()
        self.hub.barrier.wait()
        return out


def _engines(checkpoint, world: int, hub, paired: bool):
    engines = []
    for r in range(world):
        w = load(checkpoint, tp=(r, world))
        w.comm = _FastThreadComm(hub, r)            # before the engine: its buffers take the reduce path
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=1024)
        if not paired:
            e.pbuf2 = e.overlap = None
        engines.append(e)
    return engines


def _run(engines, hub, fn):
    results, errors = [None] * len(engines), []

    def body(r):
        try:
            with torch.no_grad():
                results[r] = fn(r, engines[r])
        except BaseException as exc:                # noqa: BLE001
            errors.append(exc)
            hub.barrier.abort()

    import threading

    threads = [threading.Thread(target=body, args=(r,)) for r in range(len(engines))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=99, top_k=20, top_p=0.95)])
def test_paired_prompt_chunks_give_the_whole_chunks_bits(checkpoint, world, sampling):
    rng = random.Random(7)
    prompt = [rng.randrange(1, base.V) for _ in range(1500)]       # a full 1024-row chunk, then 476 rows: both pair

    def body(r, e):
        assert (e.pbuf2 is not None) == paired
        first = prefill(e, prompt, sampling, keep_at=entry_end(prompt))
        kept = e.kept
        streams = e.last_streams.clone()
        out = mtp_decode(e, first, 24, sampling, depth=3, confidence=0.0).tokens
        state = kept["state"]
        return first, out, streams, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}

    results = {}
    for paired in (True, False):
        hub = base._Hub(world)
        engines = _engines(checkpoint, world, hub, paired)
        results[paired] = _run(engines, hub, body)
        del engines
        torch.cuda.empty_cache()
    for r in range(world):
        (f1, o1, s1, k1), (f2, o2, s2, k2) = results[True][r], results[False][r]
        assert f1 == f2 and o1 == o2, (r, f1, f2)
        assert torch.equal(s1, s2), r
        assert k1["pos"] == k2["pos"] and k1["mtp_len"] == k2["mtp_len"]
        for name in ("rec", "conv", "ple_tail"):
            assert torch.equal(k1[name], k2[name]), (r, name)
    assert all(res[1] == results[True][0][1] for res in results[True])   # every rank emits the same tokens
