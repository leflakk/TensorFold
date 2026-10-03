"""--parallel's MTP cut by the streams of the round that verifies the drafts (any machine).

A round's rows all wait for each other, so the more streams share it, the less a doubtful draft is worth: on
tensor-parallel ranks a chain stops at a cut that rises with the streams (drafts change speed only, never tokens).
"""

import json
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import CONFIDENCE, TP_STREAM_CONFIDENCE, stream_confidence


def test_the_cuts_come_from_the_default_or_the_environment(monkeypatch):
    monkeypatch.delenv("TF_MULTI_CONFIDENCE", raising=False)
    assert stream_confidence(TP_STREAM_CONFIDENCE) == (0.5, 0.6, 0.7)
    assert stream_confidence((CONFIDENCE,)) == 0.7                    # one cut: a float, as --mtp-confidence gives
    monkeypatch.setenv("TF_MULTI_CONFIDENCE", " 0.4, 0.8 ")
    assert stream_confidence(TP_STREAM_CONFIDENCE) == (0.4, 0.8)
    monkeypatch.setenv("TF_MULTI_CONFIDENCE", "0.6")
    assert stream_confidence(TP_STREAM_CONFIDENCE) == 0.6
    for bad in ("1.5", "0.5,,0.7", "high", "-0.1,0.5"):
        monkeypatch.setenv("TF_MULTI_CONFIDENCE", bad)
        with pytest.raises(ValueError, match="TF_MULTI_CONFIDENCE"):
            stream_confidence(TP_STREAM_CONFIDENCE)


@pytest.mark.torch
def test_parallel_ranks_cut_by_stream_count_unless_one_cut_is_asked(tmp_path, monkeypatch):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine as fn_engine

    monkeypatch.delenv("TF_MULTI_CONFIDENCE", raising=False)
    monkeypatch.setattr(fn_engine, "FlashNextEngine", lambda *a, **k: SimpleNamespace(**k))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc.weight": "model.safetensors"}}))
    ranks = dict(tp=8, master="127.0.0.1")
    assert qwen4_exp.cuda_engine(tmp_path).confidence == 0.7                       # one GPU, one stream
    assert qwen4_exp.cuda_engine(tmp_path, parallel=4).confidence == 0.7           # one GPU: one cut for any count
    assert qwen4_exp.cuda_engine(tmp_path, **ranks).confidence == 0.3              # the one-stream engine on 8 ranks
    assert qwen4_exp.cuda_engine(tmp_path, parallel=4, **ranks).confidence == (0.5, 0.6, 0.7)
    assert qwen4_exp.cuda_engine(tmp_path, parallel=4, mtp_confidence=0.4, **ranks).confidence == 0.4
    monkeypatch.setenv("TF_MULTI_CONFIDENCE", "0.4,0.8")
    assert qwen4_exp.cuda_engine(tmp_path, parallel=4, **ranks).confidence == (0.4, 0.8)
    assert qwen4_exp.cuda_engine(tmp_path, parallel=4, mtp_confidence=0.6, **ranks).confidence == 0.6
    assert qwen4_exp.cuda_engine(tmp_path, **ranks).confidence == 0.3              # --parallel's only


@pytest.mark.torch
def test_a_round_drafts_for_the_streams_of_the_next_round():
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    dec = object.__new__(MultiDecoder)
    dec.cuts = (0.5, 0.6, 0.7)
    s = SimpleNamespace(prompt=[1] * 10)
    assert [dec._cut(n) for n in range(6)] == [0.5, 0.5, 0.6, 0.7, 0.7, 0.7]      # the last cut for more streams
    assert dec._cut(1, [(s, 4, 6)]) == 0.6              # a prompt that ends in this round's pass joins the next round
    assert dec._cut(1, [(s, 0, 6)]) == 0.5              # one still filling does not
    dec.cuts = (0.3,)
    assert dec._cut(4, [(s, 4, 6)]) == 0.3


@pytest.mark.torch
def test_the_ranks_compare_every_cut_and_the_banner_names_them():
    from tensorfold.families.qwen4_exp.cuda.engine import _cuts_key, _cuts_text

    assert _cuts_text((0.7,)) == "70%"
    assert _cuts_text((0.5, 0.6, 0.7)) == "50%/60%/70% for rounds of 1/2/3+ streams"
    assert _cuts_key((0.3,)) == 300000                                   # one cut: its millionths, as before
    keys = {_cuts_key(c) for c in [(0.5, 0.6, 0.7), (0.5, 0.6, 0.8), (0.5, 0.7), (0.5,), (0.7,)]}
    assert len(keys) == 5 and _cuts_key((0.5, 0.6, 0.7)) < 0
