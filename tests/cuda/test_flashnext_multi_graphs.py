"""Flash Next's concurrent rounds as CUDA graphs (``round_graphs``), on one GPU.

Windows are padded to depth + 1 rows a stream, a stream count's verify forward and MTP steps are captured once a
shape and replayed with each round's tables: every stream still emits its solo run (dense and sparse attention,
quantized caches, an n-gram layer, caches that grow or move), as eager rounds do. The one-launch kernels the
captured rounds read by table give each row the bits of its stream's own launch.
"""

import tempfile
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _bf16_table, _cfg, _model, _ple, _Rand  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attention as attn_mod, attn_multi, glue, multi  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

DEV = "cuda"
PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]


def _solo(w, prompts, samplings, count, *, capacity=1024, prefill_rows=16, kv_dtype="bf16"):
    out = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=prefill_rows, kv_dtype=kv_dtype)
        out.append(serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens)
    return out


def _decode(dec, streams, first: int = 1):
    """The first streams alone a round, then the rest join: stream counts 1 up to all, then down as they end."""

    for s in streams[:first]:
        dec.admit(s)
    dec.finish(dec.round())
    for s in streams[first:]:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    return [s.out for s in streams]


@pytest.fixture
def apart(monkeypatch):
    """Prompt passes between rounds (as on tensor-parallel ranks): every round a pure decode window."""

    monkeypatch.setattr(multi, "converges", lambda w: False)


@pytest.mark.parametrize("graphs", [True, "pad", False])
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_rounds_in_graphs_equal_each_alone(apart, graphs, kv_dtype):
    w = _model()
    refs = _solo(w, PROMPTS, SAMPLINGS, 20, kv_dtype=kv_dtype)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=16,
                       graphs=graphs)
    assert (dec.rounds is not None) == bool(graphs)
    streams = [Stream(p, 20, smp, draft=i != 3, stop_eos=False) for i, (p, smp) in enumerate(zip(PROMPTS, SAMPLINGS))]
    assert _decode(dec, streams) == refs
    if graphs is True:
        kinds = {key[:2] for key in dec.rounds.graphs}
        assert any(k[0] == "verify" for k in kinds) and any(k[0] == "mtp" for k in kinds), kinds
        # the first round alone (it folds nothing) and four streams: a lone request's first round replays too
        assert {k[1] for k in kinds if k[0] == "verify"} >= {1, 4}
        assert any(k[0] == "verify" and k[1] == 1 and not k[3] for k in dec.rounds.graphs)
    assert len(dec.free) + len({id(k[1]) for k in dec.kept}) == 4 and not dec.live()


def test_a_warmed_decoder_replays_captured_rounds(apart):
    """``warm`` captures every stream count's rounds before a request: the requests' rounds replay them."""

    w = _model()
    refs = _solo(w, PROMPTS, SAMPLINGS, 24)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, prefill_rows=16, graphs=True)
    dec.warm()
    warmed = dec.rounds.captures
    assert warmed > 0 and not dec.live() and not dec.kept and len(dec.free) == 4
    streams = [Stream(p, 24, smp, draft=i != 3, stop_eos=False) for i, (p, smp) in enumerate(zip(PROMPTS, SAMPLINGS))]
    replays = dec.rounds.replays
    assert _decode(dec, streams, first=4) == refs
    assert dec.rounds.replays > replays and dec.rounds.captures - warmed <= 8     # mostly the warm-up's graphs


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_sparse_streams_in_graphs_decode_as_alone(apart, kv_dtype):
    """Past the attention budget a long stream selects blocks (one launch for every row, by table) beside dense ones."""

    w = _model(7)
    g = torch.Generator().manual_seed(9)
    prompts = [torch.randint(1, V, (2400,), generator=g).tolist(), PROMPTS[0], PROMPTS[1]]
    samplings = [Sampling(seed=5, top_k=20, top_p=0.95), None, Sampling(seed=6, top_k=20, top_p=0.95)]
    refs = _solo(w, prompts, samplings, 16, capacity=4096, prefill_rows=256, kv_dtype=kv_dtype)
    dec = MultiDecoder(w, slots=3, capacity=4096, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=256,
                       graphs=True)
    streams = [Stream(p, 16, smp, stop_eos=False) for p, smp in zip(prompts, samplings)]
    assert _decode(dec, streams, first=3) == refs


def test_an_ngram_layer_in_graphs_decodes_as_alone(apart):
    """The n-gram layer's conv over each stream's tail: one launch by table, each row its stream's own bits."""

    c = _cfg(ple=True)
    with tempfile.TemporaryDirectory() as tmp:
        table = _bf16_table(Path(tmp) / "shard_0.safetensors", c.ngram(0).rows, c.ngram(0).dims)
        w = _model(ple=_ple(c, table, _Rand(3)))
        refs = _solo(w, PROMPTS, SAMPLINGS, 20)
        for graphs in (True, False):
            dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, prefill_rows=16, graphs=graphs)
            streams = [Stream(p, 20, smp, draft=i != 3, stop_eos=False)
                       for i, (p, smp) in enumerate(zip(PROMPTS, SAMPLINGS))]
            assert _decode(dec, streams) == refs, graphs
            del dec


def test_streams_in_graphs_keep_their_bits_when_their_caches_grow_or_move(apart):
    """A captured round reads its streams' caches by table, refilled each round: growth and moves change nothing."""

    w = _model()
    samplings = [None, Sampling(seed=5, top_k=20, top_p=0.95)]
    prompts = [[(7 * i + 3) % (V - 1) + 1 for i in range(240)], [(11 * i + 5) % (V - 1) + 1 for i in range(250)]]
    refs = _solo(w, prompts, samplings, 40)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, prefill_rows=16, graphs=True)
    streams = [Stream(list(p), 40, smp, stop_eos=False) for p, smp in zip(prompts, samplings)]
    for s in streams:
        dec.admit(s)
    rounds = 0
    while dec.live():
        dec.finish(dec.round())
        rounds += 1
        if rounds % 3 == 0:                          # move every live stream's caches and states to new storage
            for s in streams:
                st = s.st
                for kc in st.kc + [st.mtp_kc]:
                    kc.k, kc.v, kc.ks, kc.vs = kc.k.clone(), kc.v.clone(), kc.ks.clone(), kc.vs.clone()
                st.ikc = [t.clone() for t in st.ikc]
                st.pooled = [t.clone() for t in st.pooled]
                st.mtp_ikc, st.mtp_pooled = st.mtp_ikc.clone(), st.mtp_pooled.clone()
                st.conv, st.rec = st.conv.clone(), st.rec.clone()
    for s, ref in zip(streams, refs):
        assert s.out == ref and s.st.version >= 1


def test_the_window_ngram_conv_gives_each_streams_own_bits():
    S, D, TAPS, DIL = 4, 512, 4, 3
    W, NT = S * D, (TAPS - 1) * DIL
    lens = [3, 1, 7, 2]
    R = sum(lens)
    g = torch.Generator(device=DEV).manual_seed(4)
    gated = torch.randn((R, W), generator=g, device=DEV).to(torch.bfloat16)
    pss = (torch.rand((R, S), generator=g, device=DEV) * 50 + 1).float()
    nc = (1 + 0.1 * torch.randn(W, generator=g, device=DEV)).to(torch.bfloat16)
    cw = (torch.randn((W, TAPS), generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    h = torch.randn((R, W), generator=g, device=DEV).to(torch.bfloat16)
    tails = [torch.randn((NT, W), generator=g, device=DEV).to(torch.bfloat16) for _ in lens]
    ref_h, ref_n, a0 = h.clone(), torch.empty((R, W), dtype=torch.bfloat16, device=DEV), 0
    for t, n in zip(tails, lens):
        glue.ple_conv(gated[a0:a0 + n], pss[a0:a0 + n], nc, t, cw, ref_h[a0:a0 + n], ref_h[a0:a0 + n],
                      ref_n[a0:a0 + n], 1e-6, S, DIL)
        a0 += n
    sid = torch.tensor([s for s, n in enumerate(lens) for _ in range(n)], dtype=torch.int32, device=DEV)
    first = torch.tensor([100 * (s + 1) for s in range(len(lens))], dtype=torch.int32, device=DEV)
    posr = torch.tensor([100 * (s + 1) + j for s, n in enumerate(lens) for j in range(n)], dtype=torch.int32,
                        device=DEV)
    ptrs = torch.tensor([t.data_ptr() for t in tails], dtype=torch.int64, device=DEV)
    got_h, got_n = h.clone(), torch.empty((R, W), dtype=torch.bfloat16, device=DEV)
    glue.ple_conv_multi(gated, pss, nc, ptrs, sid, posr, first, cw, got_h, got_h, got_n, 1e-6, S, DIL)
    assert torch.equal(got_h, ref_h) and torch.equal(got_n, ref_n)


def test_the_window_sparse_select_gives_each_streams_own_lists():
    """``_scores_multi`` and the selects by row position list the blocks each stream's own launches list."""

    HI, DI, RATIO, BUDGET = 4, 128, 4, 64
    TOP, NB = BUDGET // RATIO, 512
    sc = attn_mod.AttnScratch(32, 4, 64, NB * RATIO, DEV, budget=BUDGET, ratio=RATIO)
    ref = attn_mod.AttnScratch(32, 4, 64, NB * RATIO, DEV, budget=BUDGET, ratio=RATIO)
    pos0, lens = [300, 20, 1500, 61], [3, 2, 7, 1]
    R, n = sum(lens), len(lens)
    g = torch.Generator(device=DEV).manual_seed(6)
    pooled = [torch.randn((NB, DI), generator=g, device=DEV).to(torch.bfloat16) for _ in lens]
    iq = torch.randn((R, HI, DI), generator=g, device=DEV).to(torch.bfloat16)
    a0 = 0
    for p0, k, pool in zip(pos0, lens, pooled):
        view = attn_mod.AttnScratch.__new__(attn_mod.AttnScratch)
        view.__dict__.update(ref.__dict__)
        view.scores, view.ids, view.nk, view.sparse = ref.scores[a0:], ref.ids[a0:], ref.nk[a0:], ref.sparse[a0:]
        attn_mod.qsa_rows(iq[a0:a0 + k], pool, torch.tensor([p0], dtype=torch.int32, device=DEV), view, k,
                          context=p0 + k)
        a0 += k
    posr = torch.tensor([p0 + j for p0, k in zip(pos0, lens) for j in range(k)], dtype=torch.int32, device=DEV)
    sid = torch.tensor([s for s, k in enumerate(lens) for _ in range(k)], dtype=torch.int32, device=DEV)
    cp = torch.zeros((attn_multi.PTRS * n,), dtype=torch.int64, device=DEV)
    cp[5 * n:6 * n] = torch.tensor([t.data_ptr() for t in pooled], dtype=torch.int64, device=DEV)
    blocks = NB
    attn_multi._scores_multi[(R, -(-blocks // 64))](iq, cp, posr, sid, sc.scores, sc.nb, n, HI=HI, DI=DI, RATIO=RATIO,
                                                    TOP=TOP, BB=64, num_warps=4)
    attn_mod._launch_select(sc, posr, R, blocks, rowpos=True, decode=True)
    sparse = ref.sparse[:R].bool()
    assert sparse.any() and not sparse.all()
    assert torch.equal(sc.sparse[:R], ref.sparse[:R]) and torch.equal(sc.nk[:R], ref.nk[:R])
    assert torch.equal(sc.ids[:R][sparse], ref.ids[:R][sparse])
