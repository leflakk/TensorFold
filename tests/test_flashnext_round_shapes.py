"""Captured concurrent rounds' shapes and tables, on the CPU (``round_graphs``).

A graph captured on the warm-up's synthetic window of a shape (streams, rows) replays any later round of that
shape: both must build their tables in the same static buffers, of the same sizes, whichever way the rows split
between the streams. A round's rows go up to a multiple of 4 by padding after the last stream's own rows, no
further than its cache holds.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import gdn_multi, round_graphs
from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.cuda.weight_types import Config

DEPTH, SLOTS = 6, 4


def _cfg() -> Config:
    return Config(hidden=256, layers=2, layer_types=["linear", "attention"], vocab=512, eps=1e-6, heads=24, kv_heads=2,
                  head_dim=256, rope_theta=1e7, rotary_dim=64, nk=16, nv=48, dk=128, dv=128, conv_kernel=4,
                  experts=8, top_k=2, moe_width=64, shared_width=64, streams=4, low=64, index_heads=4,
                  index_dim=128, index_budget=2048, index_ratio=4, ple_layers=[], ple_dim=256, ple_kernel=4,
                  ngram_size=3, heads_per_ngram=8, ngram_base=1000, ngram_divisor=128, ngram_shards=1, seed=1,
                  ple_eos=0, eos=(0,), group_size=32, bits=4)


@pytest.fixture(scope="module")
def decoder():
    w = SimpleNamespace(cfg=_cfg(), device=torch.device("cpu"),
                        layers=[SimpleNamespace(linear=True, index=0), SimpleNamespace(linear=False, index=1)])
    w.mtp = SimpleNamespace(layer=SimpleNamespace(linear=False, index=-1))
    rows = SLOTS * (DEPTH + 1)
    dec = SimpleNamespace(w=w, depth=DEPTH, capacity=4096, buf=None, mbuf=None, gdn=gdn_multi.Scratch(w, rows))
    dec.rounds = round_graphs.RoundGraphs(dec)
    dec.states = [State(w, 64, DEPTH + 1, limit=4096) for _ in range(SLOTS)]
    return dec


def _segs(states, counts):
    segs, a0 = [], 0
    for st, k in zip(states, counts):
        segs.append((st, a0, a0 + k))
        a0 += k
    return segs


def _compositions(n: int, R: int, W: int):
    """A few ways ``R`` rows split over n streams (each 1..W rows but the last, which the padding may lengthen)."""

    even = [R // n + (i < R % n) for i in range(n)]
    ones = [1] * (n - 1) + [R - (n - 1)]
    full = [min(W, R - (n - 1 - i)) for i in range(n)]
    full[-1] = R - sum(full[:-1])
    return [c for c in (even, ones, full) if all(k >= 1 for k in c)]


def test_every_shape_the_warm_up_captures_is_the_buffers_its_rounds_refill(decoder):
    rg, W = decoder.rounds, DEPTH + 1
    for n in range(1, SLOTS + 1):
        shapes = sorted({rg.rows(n, own) for own in range(n, n * W + 1)})
        for R in shapes:
            buffers = None
            for counts in _compositions(n, R, W):
                segs = _segs(decoder.states[:n], counts)
                tables, step = rg.tables(segs, [list(range(min(k, W))) for k in counts])
                got = (id(rg.statics[("gdn", n, R)]), id(rg.statics[("attn", n, R)]))
                buffers = buffers or got
                assert got == buffers, (n, R, counts)            # the same static buffers, refilled in place
                assert tables.folds and step.most == rg.bound and step.n == n and step.rows == R
            for counts in _compositions(n, R, W):            # the MTP head's step of that shape
                rg.mtp_step(_segs(decoder.states[:n], counts))
                assert ("mtp", n, R) in rg.statics


def test_a_rounds_rows_go_to_a_multiple_of_4_after_the_last_streams_own(decoder):
    rg, W = decoder.rounds, DEPTH + 1
    a, b = decoder.states[:2]
    a.set_pos(10), b.set_pos(10)
    for counts in ([3, 2], [7, 7], [1, 1], [6, 3]):
        windows = [(a, list(range(counts[0]))), (b, list(range(100, 100 + counts[1])))]
        padded = rg.pad(windows)
        own, rows = sum(counts), sum(len(t) for _, t in padded)
        assert rows == min(2 * W, -(-own // 4) * 4) and rows >= own, counts
        assert padded[0][1] == windows[0][1] and padded[1][1][:counts[1]] == windows[1][1]
        assert set(padded[1][1][counts[1]:]) <= {windows[1][1][-1]}          # the last own token again
    tight = State(decoder.w, 16, W, limit=16)
    tight.set_pos(12)
    padded = rg.pad([(a, [1, 2, 3]), (tight, [4, 5])])       # 5 rows -> 8, but its cache holds 2 rows past them
    assert [len(t) for _, t in padded] == [3, 4]
    a.set_pos(0), b.set_pos(0)


def _decoder():
    w = SimpleNamespace(cfg=_cfg(), device=torch.device("cpu"),
                        layers=[SimpleNamespace(linear=True, index=0), SimpleNamespace(linear=False, index=1)])
    w.mtp = SimpleNamespace(layer=SimpleNamespace(linear=False, index=-1))
    rows = SLOTS * (DEPTH + 1)
    dec = SimpleNamespace(w=w, depth=DEPTH, capacity=4096, buf=None, mbuf=None, gdn=gdn_multi.Scratch(w, rows))
    dec.rounds = round_graphs.RoundGraphs(dec)
    dec.states = [State(w, 64, DEPTH + 1, limit=4096) for _ in range(SLOTS)]
    return dec


def test_the_warm_up_walks_every_shape_once_a_parity_and_empties_its_slots(monkeypatch):
    """``warm``'s loops on the CPU (staging, captures and replays stubbed): each verify shape at both DeltaNet
    parities, each MTP step shape once, every slot empty afterwards."""

    from tensorfold.families.qwen4_exp.cuda import forward, mtp

    dec = _decoder()
    rg, calls = dec.rounds, []
    monkeypatch.setattr(forward, "stage", lambda w, b, windows: _segs([st for st, _ in windows],
                                                                       [len(t) for _, t in windows]))
    monkeypatch.setattr(mtp, "mtp_stage", lambda w, b, windows, lasts=None: _segs(
        [st for st, _, _ in windows], [len(t) for _, t, _ in windows]))
    monkeypatch.setattr(rg, "verify", lambda segs, tables, step: calls.append(
        ("verify", len(segs), segs[-1][2], tables.cur, dec.buf.attn_step is step)))
    monkeypatch.setattr(rg, "mtp", lambda segs, step: calls.append(
        ("mtp", len(segs), segs[-1][2], dec.mbuf.attn_step is step)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    dec.buf = SimpleNamespace(gdn_tables=None, attn_step=None)
    dec.mbuf = SimpleNamespace(mtp_in=torch.zeros((SLOTS * (DEPTH + 1), 8)), attn_step=None)
    for st in dec.states:
        st.set_pos(5)
    rg.warm(dec.states)
    W = DEPTH + 1
    shapes = {n: sorted({rg.rows(n, own) for own in range(n, n * W + 1)}) for n in range(1, SLOTS + 1)}
    verify = [c for c in calls if c[0] == "verify"]
    assert sorted(c[1:4] for c in verify) == sorted((n, R, p) for n in shapes for R in shapes[n] for p in (0, 1))
    assert len(verify) == 2 * 19 and all(c[4] for c in verify)          # 19 shapes at --parallel 4, 6 drafts
    steps = [c for c in calls if c[0] == "mtp"]
    assert sorted(c[1:3] for c in steps) == sorted((n, R) for n in shapes for R in set(shapes[n]) | {rg.rows(n, n)})
    assert all(c[3] for c in steps)
    assert all(st.pos == 0 for st in dec.states) and dec.gdn.parity == 0


def test_a_captured_mtp_steps_rows_are_written_in_place_and_padded_with_the_last_own_row():
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    dec = _decoder()
    rows, wide = SLOTS * (DEPTH + 1), 8
    dec.mbuf = SimpleNamespace(mtp_in=torch.zeros((rows, wide)), ids=torch.zeros(rows, dtype=torch.int32),
                               ids_host=torch.zeros(rows, dtype=torch.int32), last=torch.zeros(rows, dtype=torch.int32),
                               last_host=torch.zeros(rows, dtype=torch.int32),
                               staged=SimpleNamespace(synchronize=lambda: None, record=lambda: None))
    source = torch.arange(10 * wide, dtype=torch.float32).view(10, wide)
    windows = [(dec.states[0], [5, 6, 7], source[0:3]), (dec.states[1], [8, 9], source[5:7])]
    segs, lasts = MultiDecoder._mtp_stage(dec, windows)
    assert [(a0, a1) for _, a0, a1 in segs] == [(0, 3), (3, 8)]          # 5 own rows -> 8
    assert lasts == [2, 4]                                              # each stream's last own row
    assert torch.equal(dec.mbuf.mtp_in[:5], torch.cat([source[0:3], source[5:7]]))
    assert torch.equal(dec.mbuf.mtp_in[5:8], source[6:7].expand(3, -1))  # the last own row again
    assert dec.mbuf.ids_host[:8].tolist() == [5, 6, 7, 8, 9, 9, 9, 9] and dec.mbuf.last_host[:2].tolist() == [2, 4]
