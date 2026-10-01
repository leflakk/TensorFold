"""The CUDA server refuses a GPU below the checkpoint's kernels' compute capability at startup, before any weights."""

import json

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import build, capacity  # noqa: E402

GPUS = [((7, 5), "NVIDIA GeForce RTX 2080 Ti"), ((8, 6), "NVIDIA GeForce RTX 3090"), ((8, 9), "NVIDIA GeForce RTX 4090"), ((9, 0), "NVIDIA H100"),
        ((12, 1), "NVIDIA GB10")]


def _gpu(monkeypatch, capability, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)


@pytest.mark.parametrize("capability,name", GPUS)
@pytest.mark.parametrize("need", [build.MIN_CAPABILITY, build.FP8, build.CLUSTERS])
def test_the_startup_check_names_the_gpu(monkeypatch, capability, name, need):
    _gpu(monkeypatch, capability, name)
    if capability < need:
        want = f"compute capability {need[0]}.{need[1]} or newer.*{name}.*{capability[0]}.{capability[1]}"
        with pytest.raises(ValueError, match=want):
            build.refuse_old_gpu(need)
    else:
        build.refuse_old_gpu(need)


def test_no_gpu_leaves_it_to_the_engine(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    build.refuse_old_gpu()


@pytest.mark.parametrize("quantization", [{"group_size": 64, "bits": 4}, {"quant_method": "compressed-tensors"},
                                          {"quant_method": "modelopt", "quant_algo": "NVFP4"}])
def test_checkpoints_run_from_ampere_fp8_ones_from_ada(tmp_path, quantization):
    key = "quantization" if "bits" in quantization else "quantization_config"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", key: quantization}))
    want = build.FP8 if quantization.get("quant_method") == "modelopt" else build.MIN_CAPABILITY
    assert capacity.floor(tmp_path) == want
    assert build.MIN_CAPABILITY == (8, 6) and build.FP8 == (8, 9)


def test_an_nvfp4_checkpoint_below_ada_is_refused_before_any_weight_loads(monkeypatch, tmp_path):
    _gpu(monkeypatch, (8, 6), "NVIDIA GeForce RTX 3090")
    config = {"model_type": "qwen3_5", "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    fail = lambda *a, **k: pytest.fail("admission read the checkpoint on a GPU it refuses")   # noqa: E731
    monkeypatch.setattr(capacity, "estimate_weights", fail)
    with pytest.raises(ValueError, match=r"compute capability 8\.9 or newer.*RTX 3090.*is 8\.6"):
        capacity.admit(tmp_path, None, None, torch, fail, fail)
