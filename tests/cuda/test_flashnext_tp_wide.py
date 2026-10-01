"""Flash Next on 4 and 8 tensor-parallel ranks, simulated as threads on one GPU (one RTX 3090 is enough).

The small random checkpoint of ``test_flashnext_tp`` with the real expert width (640, so 8 ranks hold 96 or 64 of
it in whole 32-groups) and the real heads (24 query / 2 KV: 4 or 8 ranks share a KV head). Checks: every rank emits
the same tokens; MTP-drafted decoding gives serial decoding's tokens; window rows give serial steps' bits; the
ranks' logits agree with the one-GPU model's to rounding.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import test_flashnext_tp as base  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import tp as plan  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import load  # noqa: E402

WIDTH = 640
PROMPT = base.PROMPT


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("flashnext_tp_wide")
    was = base.W
    base.W = WIDTH
    try:
        base._checkpoint(root)
    finally:
        base.W = was
    return root


@pytest.fixture(scope="module")
def single(checkpoint):
    return load(checkpoint)


def _ranks(checkpoint, world, **kw):
    return [load(checkpoint, tp=(r, world), **kw) for r in range(world)]


@pytest.mark.parametrize("world", [4, 8])
def test_ranks_split_heads_kv_groups_experts_and_vocabulary(checkpoint, world):
    ranks = _ranks(checkpoint, world)
    widths = plan.widths(WIDTH, world)
    assert sum(widths) == WIDTH and all(x % 32 == 0 for x in widths)
    for r, w in enumerate(ranks):
        assert w.meta["world"] == world and w.meta["rank"] == r
        assert (w.cfg.heads, w.cfg.kv_heads, w.cfg.nk, w.cfg.nv) == (base.HEADS // world, 1, base.NK // world,
                                                                    base.NV // world)
        assert w.cfg.moe_width == widths[r] and w.head.n == base.V // world
    del ranks
    torch.cuda.empty_cache()


@pytest.mark.parametrize("world", [4, 8])
def test_logits_agree_with_one_gpu(checkpoint, single, world):
    e1 = Engine(single, capacity=1024, max_rows=16, prefill_rows=16)
    ref = forward(single, e1.st, e1.buf, PROMPT)[:len(PROMPT)].float().clone()
    engines = [Engine(w, capacity=1024, max_rows=16, prefill_rows=16) for w in _ranks(checkpoint, world)]
    parts = base._run_ranks(lambda r, e: forward(e.w, e.st, e.buf, PROMPT)[:len(PROMPT)].float().clone(), engines)
    got = torch.cat(parts, dim=1)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=1)
    assert float(cos.min()) > 0.999, cos
    assert float((got - ref).abs().max()) < 0.05 * float(ref.abs().max())


@pytest.mark.parametrize("world", [4, 8])
def test_windows_match_serial_steps(checkpoint, world):
    engines = [Engine(w, capacity=1024, max_rows=16, prefill_rows=16) for w in _ranks(checkpoint, world)]
    nxt = [401, 33, 2048, 5, 77, 1500, 9, 10, 11]

    def body(r, e):
        w = e.w
        forward(w, e.st, e.buf, PROMPT)
        commit(w, e.st, e.buf, len(PROMPT), len(PROMPT))
        serial = e.st.clone()
        steps = []
        for t in nxt:
            steps.append(forward(w, serial, e.buf, [t])[0].clone())
            commit(w, serial, e.buf, 1, 1)
        bad = []
        for R in (2, 4, 8):
            st = e.st.clone()
            lg = forward(w, st, e.buf, nxt[:R])
            bad += [(R, row) for row in range(R) if not torch.equal(lg[row], steps[row])]
        return bad

    for r, bad in enumerate(base._run_ranks(body, engines)):
        assert not bad, (r, bad)


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=1234, top_k=20, top_p=0.95)])
def test_mtp_drafts_give_serial_tokens_on_every_rank(checkpoint, world, sampling):
    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=16) for w in _ranks(checkpoint, world)]

    def body(r, e):
        first = prefill(e, PROMPT, sampling)
        out = {"serial": serial_decode(e, first, 24, sampling).tokens}
        for name, depth in (("d3", 3), ("d6", 6)):
            prefill(e, PROMPT, sampling)
            out[name] = mtp_decode(e, first, 24, sampling, depth=depth, confidence=0.0).tokens
        return out

    got = base._run_ranks(body, engines)
    assert all(g == got[0] for g in got)
    assert got[0]["d3"] == got[0]["serial"] and got[0]["d6"] == got[0]["serial"]
