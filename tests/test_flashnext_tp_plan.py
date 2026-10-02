"""How Flash Next splits over 1, 2, 4 and 8 ranks (``families/qwen4_exp/cuda/tp.py``), on the real shapes."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda.geometry import share
from tensorfold.families.qwen4_exp.cuda import tp

CFG = SimpleNamespace(heads=24, kv_heads=2, nk=16, nv=48, vocab=248320, moe_width=640, shared_width=640)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_spans_cover_the_width_in_whole_groups(world):
    tp.check(CFG, world)
    spans = [tp.span(640, world, r, 32) for r in range(world)]
    assert spans[0][0] == 0 and spans[-1][1] == 640
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all((hi - lo) % 32 == 0 and hi > lo for lo, hi in spans)
    assert max(hi - lo for lo, hi in spans) == share(640, world)


def test_eight_ranks_take_96_or_64_columns():
    assert tp.widths(640, 8) == [96, 96, 96, 96, 64, 64, 64, 64]
    assert tp.widths(640, 4) == [160] * 4


@pytest.mark.parametrize("world, want", [(1, [(0, 2)]), (2, [(0, 1), (1, 1)]),
                                         (4, [(0, 1), (0, 1), (1, 1), (1, 1)]),
                                         (8, [(0, 1)] * 4 + [(1, 1)] * 4)])
def test_kv_heads_follow_their_query_groups(world, want):
    got = [tp.kv_span(2, world, r) for r in range(world)]
    assert got == want
    q = 24 // world
    for r, (kv, _) in enumerate(got):              # every local query head reads the KV head this rank holds
        assert all(h // 12 == kv for h in range(r * q, (r + 1) * q))


@pytest.mark.parametrize("world", [3, 16])
def test_other_rank_counts_are_refused(world):
    with pytest.raises(ValueError, match="--tp"):
        tp.check(CFG, world)
