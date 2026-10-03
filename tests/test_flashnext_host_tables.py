"""A concurrent round's host work (any machine): the n-gram rows of every stream in one lookup and one gather, the
DeltaNet tables' state pointers by stride and their chain plans by row counts. Each gives the values it gave stream
by stream and layer by layer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import gdn_multi
from tensorfold.families.qwen4_exp.cuda.ngram import NGram
from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.cuda.static_tables import StaticTables
from tensorfold.families.qwen4_exp.cuda.weight_types import Config

EOS = 0


def _ngram() -> NGram:
    return NGram(vocab=1000, ngram_size=3, heads_per_ngram=4, vocab_base=5000, divisor=128, shards=1, seed=7,
                 eos=EOS, embed_dim=256)


def _pairs(rng, sizes):
    """(history, tokens) pairs with EOS in some histories and tokens."""

    pairs = []
    for i, k in enumerate(sizes):
        history = rng.integers(1, 1000, size=2)
        if i % 2:
            history[i % 2] = EOS
        tokens = rng.integers(1, 1000, size=k)
        if k > 2 and i % 3 == 0:
            tokens[k // 2] = EOS
        pairs.append((history.astype(np.int64), tokens.astype(np.int64)))
    return pairs


def test_one_lookup_of_several_sequences_gives_each_its_own_rows():
    ng, rng = _ngram(), np.random.default_rng(3)
    for sizes in ([1], [3, 7], [1, 1, 1, 1], [7, 2, 5, 1], [40, 3]):
        pairs = _pairs(rng, sizes)
        alone = np.concatenate([ng.ids(h, t) for h, t in pairs])
        got = ng.ids_many(pairs)
        assert got.shape == (sum(sizes), ng.heads) and np.array_equal(got, alone), sizes
    first = (ng.initial_history(), np.array([5, 6, 7]))                       # all-EOS history, a fresh stream
    assert np.array_equal(ng.ids_many([first, first]), np.concatenate([ng.ids(*first)] * 2))
    short = [(np.array([9]), np.array([4, 5])), (np.array([1, 2]), np.array([3]))]     # (a short history: alone)
    assert np.array_equal(ng.ids_many(short), np.concatenate([ng.ids(h, t) for h, t in short]))


@pytest.mark.torch
def test_a_rounds_ngram_rows_stage_in_one_lookup_and_one_gather():
    from tensorfold.families.qwen4_exp.cuda import forward

    ng, rng, words = _ngram(), np.random.default_rng(5), 2
    gathers = []

    class Table:
        bits = 4

        def gather(self, ids):
            gathers.append(ids.shape)
            flat = ids.reshape(-1)
            w = np.stack([flat, flat + 1], axis=1).astype(np.uint32)
            return w, (flat[:, None] % 7).astype(np.uint16), (flat[:, None] % 5).astype(np.uint16)

    rows, nrow = 16, 16 * ng.heads
    b = SimpleNamespace(rows=rows, staged=SimpleNamespace(synchronize=lambda: None, record=lambda: None),
                        ids_host=torch.zeros(rows, dtype=torch.int32), ids=torch.zeros(rows, dtype=torch.int32),
                        ple_hw=torch.zeros((nrow, words), dtype=torch.int32),
                        ple_w=torch.zeros((nrow, words), dtype=torch.int32),
                        ple_hs=torch.zeros((nrow, 1), dtype=torch.int16), ple_s=torch.zeros((nrow, 1), dtype=torch.bfloat16),
                        ple_hb=torch.zeros((nrow, 1), dtype=torch.int16), ple_b=torch.zeros((nrow, 1), dtype=torch.bfloat16))
    w = SimpleNamespace(layers=[SimpleNamespace(ple=None), SimpleNamespace(ple=SimpleNamespace(ngram=ng, table=Table()))],
                        x3=None)
    pairs = _pairs(rng, [3, 1, 4])
    states = [SimpleNamespace(pos=10, capacity=64, ple_history=h, ple_last=None) for h, _ in pairs]
    segs = forward.stage(w, b, [(st, list(t)) for st, (_, t) in zip(states, pairs)])
    assert [(a0, a1) for _, a0, a1 in segs] == [(0, 3), (3, 4), (4, 8)]
    assert gathers == [(8, ng.heads)]                                         # one gather for the three streams
    ids = np.concatenate([ng.ids(h, t) for h, t in pairs]).reshape(-1)
    assert b.ple_w[:len(ids), 0].tolist() == ids.astype(np.int32).tolist()  # each row where it went stream by stream
    for st, (h, t) in zip(states, pairs):
        assert st.ple_last[0] is h and np.array_equal(st.ple_last[1], t)


def _cfg() -> Config:
    return Config(hidden=64, layers=8, layer_types=["linear"] * 8, vocab=512, eps=1e-6, heads=4, kv_heads=1,
                  head_dim=64, rope_theta=1e7, rotary_dim=32, nk=2, nv=2, dk=16, dv=16, conv_kernel=4,
                  experts=8, top_k=2, moe_width=64, shared_width=64, streams=4, low=64, index_heads=4,
                  index_dim=32, index_budget=2048, index_ratio=4, ple_layers=[], ple_dim=256, ple_kernel=4,
                  ngram_size=3, heads_per_ngram=8, ngram_base=1000, ngram_divisor=128, ngram_shards=1, seed=1,
                  ple_eos=0, eos=(0,), group_size=32, bits=4)


def test_a_rounds_deltanet_tables_point_at_each_layers_states_and_chain_its_rows():
    from tensorfold.cuda.kernels import gdn as shared

    w = SimpleNamespace(cfg=_cfg(), device=torch.device("cpu"),
                        layers=[SimpleNamespace(linear=i % 4 != 3, index=i) for i in range(8)], mtp=None)
    scratch = gdn_multi.Scratch(w, 32)
    states = [State(w, 64, 7, limit=4096) for _ in range(3)]
    for k, st in enumerate(states):                       # each stream's layers on their own current state
        st.cur = [(li * (k + 1) + k) % 2 for li in range(len(st.cur))]
    for counts in ([1], [3, 7], [2, 1, 9], [7, 7, 7]):
        segs, a0 = [], 0
        for st, k in zip(states, counts):
            segs.append((st, a0, a0 + k))
            a0 += k
        n, lin, static = len(segs), scratch.lin, StaticTables(w.device)
        t = gdn_multi.Tables(w, scratch, segs, [[0] * min(k, 2) for k in counts], static=static, width=7, folds=True)
        ptrs = static.host["ptrs"].view(2, lin, n)
        for s, (st, _, _) in enumerate(segs):
            for li in range(lin):
                assert int(ptrs[0, li, s]) == st.conv[li].data_ptr() and t.conv[li, s] == st.conv[li].data_ptr()
                assert int(ptrs[1, li, s]) == st.rec[st.cur[li], li].data_ptr()
        entries, starts, slots, most = shared.plan_host([list(range(-1, k - 1)) for k in counts])
        R = sum(counts)
        assert static.host["ints"][5 * R:8 * R + n + 1].tolist() == entries + starts, counts
        assert (t.plan.slots, t.plan.max_rows) == (slots, most)
