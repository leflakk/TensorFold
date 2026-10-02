"""Where hc_writeback's rank-partials mode (3) and its rounded-branch mode (1) disagree on this GPU.

    python tools/diag_writeback.py
"""

import torch

from tensorfold.families.qwen4_exp.cuda import glue

torch.manual_seed(12)
rows, d, s = 3, 2560, 4
h = torch.randn((rows, s * d), device="cuda").to(torch.bfloat16)
parts = torch.randn((2, rows, d), device="cuda")
inj = (torch.rand((rows, s), device="cuda") * 2).to(torch.bfloat16)
pa, pb = torch.empty((rows, d // 256, s), device="cuda"), torch.empty((rows, d // 256, s), device="cuda")
ha, hb = h.clone(), h.clone()
glue.hc_writeback(ha, ha, pa, s, 3, branch=parts.contiguous(), inject=inj)
branch = (parts[0] + parts[1]).to(torch.bfloat16)
glue.hc_writeback(hb, hb, pb, s, 1, branch=branch, inject=inj)
diff = (ha.view(torch.int16) != hb.view(torch.int16)).nonzero()
print(f"{torch.cuda.get_device_name()}: {len(diff)} of {ha.numel()} stream values differ; pss equal: "
      f"{torch.equal(pa, pb)}")
for r, j in diff[:8].tolist():
    st, col = divmod(j, d)
    b = branch[r, col].float()
    exact = h[r, j].float() + b * inj[r, st].float()                   # fused: one rounding
    stepwise = h[r, j].float() + (b * inj[r, st].float()).to(torch.bfloat16).float()
    print(f"  row {r} stream {st} col {col}: mode3 {ha[r, j].item():+.6g} mode1 {hb[r, j].item():+.6g} | "
          f"h {h[r, j].item():+.6g} branch {b.item():+.6g} (fp32 sum {(parts[0, r, col] + parts[1, r, col]).item():+.8g}) "
          f"inj {inj[r, st].item():.4g} | bf16(fused) {exact.to(torch.bfloat16).item():+.6g} "
          f"bf16(stepwise) {stepwise.to(torch.bfloat16).item():+.6g}")
