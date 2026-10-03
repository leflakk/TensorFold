"""The two-pass Triton top-k (``topk.candidates``) against torch.topk and logsumexp, on the GPU."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import topk  # noqa: E402

K = 32


@pytest.mark.parametrize("rows,n", [(1, 9949), (7, 9949), (4, 31040), (16, 31040), (2, 1000)])
@pytest.mark.parametrize("mapped", [False, True])
def test_candidates_hold_each_rows_top_k(rows, n, mapped):
    g = torch.Generator(device="cuda").manual_seed(rows * n)
    logits = (torch.randn((rows, n), generator=g, device="cuda") * 3).to(torch.bfloat16)
    id_map = (torch.randperm(250000, device="cuda")[:n].sort().values if mapped else None)
    out = torch.empty((rows, 2 * K + 1), dtype=torch.float32, device="cuda")
    topk.candidates(logits, out, K, offset=1000, id_map=id_map, scratch={})
    lf = logits.float()
    vals, idx = torch.topk(lf, K, dim=-1)
    for r in range(rows):
        got_v, got_i = out[r, :K], out[r, K:2 * K].contiguous().view(torch.int32).long()
        assert torch.equal(torch.sort(got_v, descending=True).values, vals[r]), r      # the same K values
        cols = torch.searchsorted(id_map, got_i) if mapped else got_i - 1000
        assert torch.equal(lf[r, cols], got_v), r                                      # each id holds its value
        assert len(set(got_i.tolist())) == K, r                                        # no id twice
        lse = torch.logsumexp(lf[r], 0)
        assert abs(float(out[r, 2 * K]) - float(lse)) <= 1e-5 * max(1.0, abs(float(lse))), r


def test_equal_logits_go_lowest_id_first():
    logits = torch.zeros((1, 5000), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((1, 2 * K + 1), dtype=torch.float32, device="cuda")
    topk.candidates(logits, out, K, scratch={})
    assert out[0, K:2 * K].contiguous().view(torch.int32).tolist() == list(range(K))
