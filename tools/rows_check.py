"""Does a decode row's arithmetic depend on its window's width? Flash Next's decode kernels at the real shapes.

    CUDA_VISIBLE_DEVICES=0 python tools/rows_check.py          # one GPU, ~1 min, random weights

Concurrent rounds put up to 4 streams of 7 rows in one window: every kernel a window runs must give each row the
bits it gets alone (1 row) at 2, 7, 16, 17, 28, 32 and 64 rows, or a stream decoded beside others would differ
from its reply alone. Prints one line a check: ok, or the first width and row that differ.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

ROWS = [2, 3, 7, 16, 17, 21, 28, 32, 48, 64]
S, D, LOW = 4, 2560, 320                 # hyper-connection streams, hidden, low rank
DEV = "cuda"


def q4(n: int, k: int, seed: int, lead=()):
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (*lead, n, k // 8), generator=g, device=DEV,
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((*lead, n, k // 32), generator=g, device=DEV) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((*lead, n, k // 32), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    return words, scales, biases


def report(name: str, alone, run, outputs: int = 1) -> bool:
    """``alone(r)``: row r's outputs computed alone; ``run(m)``: the first m rows' outputs together."""

    single = [alone(r) for r in range(max(ROWS))]
    for m in ROWS:
        got = run(m)
        for r in range(m):
            for o in range(outputs):
                a, b = single[r][o], got[o][r]
                if not torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)):
                    diff = (a.float() - b.float()).abs().max().item()
                    print(f"  DIFF {name}: width {m}, row {r}, output {o}, max |diff| {diff:.3g}", flush=True)
                    return False
    print(f"  ok   {name}", flush=True)
    return True


def main() -> int:
    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda import moe
    from tensorfold.families.qwen4_exp.cuda import forward as fwd
    from tensorfold.families.qwen4_exp.cuda import glue, qmm
    from tensorfold.families.qwen4_exp.cuda.weights import HC

    torch.manual_seed(0)
    print(f"GPU {torch.cuda.get_device_name()} (sm_{''.join(map(str, torch.cuda.get_device_capability()))})")
    qmm.device_defaults()
    moe.device_defaults()
    big = max(ROWS)
    ok = True

    # the tiled matmuls of the hyper-connections (down + inject, a mixer's down, up), as the plain read-out runs them
    for n, k in ((324, S * D), (320, S * D), (S * D, LOW)):
        q = qmm.make_q4(*q4(n, k, n + k), "tiled")
        x = torch.randn((big, k), device=DEV).to(torch.bfloat16)
        ok &= report(f"tiled matmul {n}x{k}", lambda r: [qmm.matmul(x[r:r + 1], q, f32=True)],
                     lambda m: [qmm.matmul(x[:m], q, f32=True)])

    # the frag matmuls of a rank at 8 ranks (DeltaNet in, attention in/out, the head's 1/8)
    for n, k in ((2048, D), (2560, 768), (1824, D), (31040, D)):
        q = qmm.make_q4(*q4(n, k, 3 * n + k))
        x = torch.randn((big, k), device=DEV).to(torch.bfloat16)
        ok &= report(f"frag matmul {n}x{k}", lambda r: [qmm.matmul(x[r:r + 1], q, f32=True)],
                     lambda m: [qmm.matmul(x[:m], q, f32=True)])

    # a hyper-connection read-out (with inject gates) as a decode window runs it
    down = qmm.stack_q4([q4(LOW, S * D, 5), q4(S, S * D, 6)], "tiled")
    hc = HC(down, qmm.make_q4(*q4(S * D, LOW, 7), "tiled"),
            1 + 0.05 * torch.randn((S * D,), device=DEV), True)
    h = (torch.randn((big, S * D), device=DEV) * 3).to(torch.bfloat16)

    def readout(rows: torch.Tensor):
        m = rows.shape[0]
        f32, bf = torch.float32, torch.bfloat16
        b = SimpleNamespace(pss=torch.empty((m, D // 256, S), dtype=f32, device=DEV),
                            normed=torch.empty((m, S * D), dtype=bf, device=DEV),
                            xs_normed=torch.empty((m, S * D // 32), dtype=f32, device=DEV),
                            dn=torch.empty((m, LOW + S), dtype=bf, device=DEV),
                            dn_mix=torch.empty((m, LOW), dtype=bf, device=DEV),
                            act=torch.empty((m, LOW), dtype=bf, device=DEV),
                            xs_act=torch.empty((m, LOW // 32), dtype=f32, device=DEV),
                            up=torch.empty((m, S * D), dtype=bf, device=DEV),
                            mixed=torch.empty((m, D), dtype=bf, device=DEV),
                            xs_mixed=torch.empty((m, D // 32), dtype=f32, device=DEV),
                            part=torch.empty((64 * max(m, 4) * 2560,), dtype=f32, device=DEV),
                            inj=torch.empty((m, S), dtype=bf, device=DEV), prefill=False)
        hh = rows.clone()
        glue.hc_writeback(hh, hh, b.pss, S, 0)
        fwd._readout(hc, b, hh, m, 1e-6, S, LOW, b.inj)
        return [b.mixed, b.xs_mixed, b.inj, b.normed]

    ok &= report("hyper-connection read-out (_readout)", lambda r: readout(h[r:r + 1]),
                 lambda m: readout(h[:m]), outputs=4)

    # the router and the routed experts (split gate/up, as tensor-parallel decode windows run them)
    experts, top = 512, 10
    for width in (96, 64):
        gate, up_w, down_w = (q4(width, D, 11, (experts + 1,)), q4(width, D, 12, (experts + 1,)),
                              q4(D, width, 13, (experts + 1,)))
        ex = grouped.make([gate, up_w], down_w, 32)
        cfg = SimpleNamespace(num_experts_per_tok=top, num_experts=experts, moe_intermediate_size=width, hidden_size=D)
        router = (torch.randn((experts + 1, D), device=DEV) * 0.02).to(torch.bfloat16)
        x = torch.randn((big, D), device=DEV).to(torch.bfloat16)
        ok &= report("router 513x2560" if width == 96 else "router again",
                     lambda r: [moe.router(x[r:r + 1], router)], lambda m: [moe.router(x[:m], router)])
        for split in (8, 0):
            buf = moe.MoEBuffers(big, cfg, DEV, prefill=True, split=split)

            def run(m, buf=buf):
                moe.moe(x[:m], router, ex, buf, top, experts)
                return [buf.pick[:m].clone(), buf.wts[:m].clone(), buf.y[:m].clone()]

            ok &= report(f"experts width {width}, {'split ' + str(split) if split else 'unsplit'}",
                         lambda r, run=run: [t[0] for t in run_one(run, r, x)], run, outputs=3)
    print("all rows keep their bits" if ok else "SOME ROWS DEPEND ON THEIR WINDOW", flush=True)
    return 0 if ok else 1


def run_one(run, r: int, x: torch.Tensor):
    """Row r alone through ``run`` (which reads the first m rows of x): r moved to row 0, then back."""

    saved = x[0].clone()
    x[0].copy_(x[r])
    try:
        return run(1)
    finally:
        x[0].copy_(saved)


if __name__ == "__main__":
    sys.exit(main())
