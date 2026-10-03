"""The one-stream engine's serial reference ("draft": false), any machine: it decodes in the drafting state, whose
kept prompt states then go (its prefill overwrites their rows), so the window holds one set of caches; with
TF_SERIAL_TWIN=1 it decodes in a state of its own, made once, and the kept states stay."""

from types import SimpleNamespace

import pytest


@pytest.mark.torch
@pytest.mark.parametrize("twin", [False, True])
def test_the_serial_reference_decodes_in_the_drafting_state_unless_it_has_its_own(monkeypatch, twin):
    import torch

    from tensorfold.families.qwen4_exp.cuda import decode
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    seen = []
    monkeypatch.setattr(decode, "prefill", lambda e, prompt, sampling, **kw: seen.append(("prefill", e, kw["mtp"])) or 7)
    monkeypatch.setattr(decode, "serial_decode", lambda e, first, count, sampling, **kw: seen.append(("decode", e))
                        or SimpleNamespace(seconds=0.1, rounds=3, tokens_per_second=30.0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    eng = FlashNextEngine.__new__(FlashNextEngine)
    own = SimpleNamespace(name="twin")
    eng.e = SimpleNamespace(name="drafting", twin=lambda: own)
    eng.serial, eng.serial_twin, eng.eos = None, twin, (0,)
    kept = [([1, 2, 3], {"state": "kept"})]
    for _ in range(2):
        eng.cache = list(kept)
        stats = eng._serial([1, 2, 3, 4], 8, None, None)
        assert stats["drafts"] is False and stats["rounds"] == 3
        assert eng.cache == (kept if twin else [])            # the drafting state's kept prompt states go with it
    used = own if twin else eng.e
    assert seen == [("prefill", used, False), ("decode", used)] * 2
    assert eng.serial is (own if twin else None)              # the twin is made once, on first use
