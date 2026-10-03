"""TF_PROFILE_DECODE's round profiles (any machine): concurrent rounds by their stream count, with the time since
the round before; each print covers the last N rounds."""

import itertools
from types import SimpleNamespace

import pytest


@pytest.mark.torch
def test_rounds_are_profiled_by_stream_count_and_each_print_covers_the_last_rounds(monkeypatch, capsys):
    from tensorfold.families.qwen4_exp.cuda import decode

    w = SimpleNamespace(meta={"rank": 0})
    monkeypatch.delenv("TF_PROFILE_DECODE", raising=False)
    assert decode._Profile.make(w, 4) is None
    monkeypatch.setenv("TF_PROFILE_DECODE", "2")
    assert decode._Profile.make(SimpleNamespace(meta={"rank": 3}), 4) is None          # rank 0 prints alone
    monkeypatch.setattr(decode, "_PROFILES", {})
    monkeypatch.setattr(decode.torch.cuda, "synchronize", lambda: None)
    clock = itertools.count(10.0, 0.001)
    monkeypatch.setattr(decode, "time", SimpleNamespace(perf_counter=lambda: next(clock)))

    four = decode._Profile.make(w, 4)
    assert decode._Profile.make(w, 4) is four and decode._Profile.make(w, 1) is not four
    assert decode._Profile.make(w) is not four                                           # the one-stream engine's
    for _ in range(2):
        prof = decode._Profile.make(w, 4)
        prof.since("outside", prof.last - 0.004)
        prof.since("outside", None)                                                      # (not counted)
        prof.mark("verify", 20)
        prof.count("padding", 1)
        prof.gauge("graphs captured", 57)
        prof.round()
    out = capsys.readouterr().out
    assert "decode profile (4 streams) over 2 rounds" in out and "outside 4.00 ms" in out
    assert "verify rows 20.00 (padding 1.00)" in out and out.rstrip().endswith("a round; graphs captured 57")
    assert (four.rounds, four.seconds, four.counts) == (0, {}, {})                       # the next print starts over
    assert four.gauges == {"graphs captured": 57}
    one = decode._Profile.make(w, 1)
    one.round()
    one.round()
    assert "decode profile (1 stream) over 2 rounds" in capsys.readouterr().out
