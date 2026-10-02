"""Time Flash Next's decode-window kernels on one GPU at a TP 8 rank's shapes, over their launch shapes.

    CUDA_VISIBLE_DEVICES=0 python tools/tune_decode_kernels.py            # ~5 min, random weights, no checkpoint

Each launch is captured 50 times in a CUDA graph (as decode runs it) and the replay timed. Kernels:
- hyper-connection down projection [324, 10240] (TF_HC_DOWN=sk,gpi,warps,stages,bn; sk changes bits),
- hyper-connection up projection + mix [10240, 320] (TF_HC_UPMIX=db,gpi,warps,stages; db 32 or 64),
- router [513, 2560] bf16 (TF_ROUTER=block_e,bk,stages,warps),
- routed + shared experts' SwiGLU gate/up at widths 96 and 64 (TF_EXPERT_SPLITS=n; 1: the unsplit kernel),
  and the down projection for reference.
It prints, per kernel and row count, the default's time, the best shape and the variable to set.
"""

from __future__ import annotations

import itertools
import time

import torch

D, S, LOW, E, TOPK = 2560, 4, 320, 512, 10
ROWS = (1, 4, 7)


def graph_us(fn, calls: int = 50, reps: int = 20) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (reps * calls) * 1e6


def q4_arrays(n: int, k: int, lead=(), seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    words = torch.randint(-(2 ** 31), 2 ** 31 - 1, (*lead, n, k // 8), generator=g, device="cuda",
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((*lead, n, k // 32), generator=g, device="cuda") * 0.01 + 0.001).to(torch.bfloat16)
    biases = (-scales.float() * 7.5).to(torch.bfloat16)
    return words, scales, biases


def report(title: str, results: dict, default, env: str) -> None:
    for rows, by_cfg in results.items():
        best = min(by_cfg, key=by_cfg.get)
        print(f"{title:28} R={rows:<2} default {by_cfg[default]:7.1f} us   best {by_cfg[best]:7.1f} us   "
              f"{env}={','.join(map(str, best)) if isinstance(best, tuple) else best}", flush=True)


def tune_hc(results_out: list) -> None:
    from tensorfold.families.qwen4_exp.cuda import glue, qmm

    qmm.device_defaults()                      # this GPU's defaults first, so the sweep's settings stay put
    down_w = qmm.make_q4(*q4_arrays(LOW + S, S * D, seed=1), "tiled")
    up_w = qmm.make_q4(*q4_arrays(S * D, LOW, seed=2), "tiled")
    scale = torch.rand((S * D,), device="cuda") + 0.5
    part = torch.empty((64 * 16 * (LOW + S),), dtype=torch.float32, device="cuda")
    downs, ups = {}, {}
    default_down = qmm.SHAPES16[(LOW + S, S * D)]
    default_up = qmm.UPMIX
    for rows in ROWS:
        h = torch.randn((rows, S * D), device="cuda").to(torch.bfloat16)
        pss = torch.empty((rows, D // 256, S), device="cuda")
        glue.hc_writeback(h, h, pss, S, 0)                 # the streams' partial sums of squares, as decode has them
        normed = torch.empty((rows, S * D), dtype=torch.bfloat16, device="cuda")
        out = torch.empty((rows, LOW + S), dtype=torch.bfloat16, device="cuda")
        downs[rows] = {}
        for sk, gpi, warps, stages, bn in itertools.product((16, 32, 64), (1, 2, 4), (2, 4), (2, 3, 4), (32, 64)):
            if (S * D // 32) % sk or (S * D // 32 // sk) % gpi:
                continue
            qmm.SHAPES16[(LOW + S, S * D)] = (sk, gpi, warps, stages, bn)
            try:
                downs[rows][(sk, gpi, warps, stages, bn)] = graph_us(
                    lambda: qmm.hc_down(h, pss, scale, normed, down_w, 1e-6, S, out=out, part=part))
            except Exception:                              # noqa: BLE001 - a shape the kernel refuses
                torch.cuda.synchronize()
        qmm.SHAPES16[(LOW + S, S * D)] = default_down
        act = torch.randn((rows, LOW), device="cuda").to(torch.bfloat16)
        xs_act = qmm.group_sums(act)
        mixed = torch.empty((rows, D), dtype=torch.bfloat16, device="cuda")
        xs_mixed = torch.empty((rows, D // 32), dtype=torch.float32, device="cuda")
        ups[rows] = {}
        for db, gpi, warps, stages in itertools.product((32, 64), (1, 2, 5), (2, 4, 8), (2, 3, 4)):   # db <= a 64-row tile
            qmm.UPMIX = (db, gpi, warps, stages)
            try:
                ups[rows][(db, gpi, warps, stages)] = graph_us(
                    lambda: qmm.hc_upmix(act, xs_act, up_w, normed, mixed, xs_mixed, S))
            except Exception:                              # noqa: BLE001
                torch.cuda.synchronize()
        qmm.UPMIX = default_up
    report("HC down [324 x 10240]", downs, default_down, "TF_HC_DOWN")
    report("HC up + mix [10240 x 320]", ups, default_up, "TF_HC_UPMIX")
    results_out += [("hc_down", downs), ("hc_upmix", ups)]


def tune_router(results_out: list) -> None:
    from tensorfold.cuda import moe

    moe.device_defaults()
    rows_w = (torch.randn((E + 1, D), device="cuda") * 0.02).to(torch.bfloat16)
    found = {}
    default = moe.ROUTER16
    for rows in ROWS:
        x = torch.randn((rows, D), device="cuda").to(torch.bfloat16)
        out = torch.empty((rows, E + 1), dtype=torch.float32, device="cuda")
        found[rows] = {}
        for be, bk, stages, warps in itertools.product((16, 32, 64), (64, 128, 256), (2, 3, 4), (2, 4, 8)):
            moe.ROUTER16 = (be, bk, stages, warps)
            try:
                found[rows][(be, bk, stages, warps)] = graph_us(lambda: moe.router(x, rows_w, out))
            except Exception:                              # noqa: BLE001
                torch.cuda.synchronize()
        moe.ROUTER16 = default
    report("router [513 x 2560]", found, default, "TF_ROUTER")
    results_out.append(("router", found))


def tune_experts(results_out: list) -> None:
    from types import SimpleNamespace

    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda import moe

    for width in (96, 64):
        gate = q4_arrays(width, D, (E + 1,), seed=3)
        up = q4_arrays(width, D, (E + 1,), seed=4)
        down = q4_arrays(D, width, (E + 1,), seed=5)
        ex = grouped.make([gate, up], down, 32)
        del gate, up, down
        cfg = SimpleNamespace(num_experts_per_tok=TOPK, num_experts=E, moe_intermediate_size=width, hidden_size=D)
        found, downs = {}, {}
        for rows in ROWS:
            g = torch.Generator(device="cuda").manual_seed(rows)
            picks = torch.stack([torch.randperm(E, generator=g, device="cuda")[:TOPK] for _ in range(rows)])
            picks = torch.cat([picks, torch.full((rows, 1), E, device="cuda")], dim=1).to(torch.int32).contiguous()
            x = torch.randn((rows, D), device="cuda").to(torch.bfloat16)
            found[rows] = {}
            for sk in (1, 2, 4, 8, 16):
                buf = moe.MoEBuffers(16, cfg, "cuda", prefill=True, split=sk)
                buf.pick[:rows].copy_(picks)
                grouped.route(buf.pick[:rows], buf.plan)
                act = buf.act.view(-1, width)
                found[rows][sk] = graph_us(lambda: grouped.gate_up(x, ex, buf.plan, act, rows))
                if sk == 8:
                    downs[rows] = graph_us(lambda: grouped.down(act, ex, buf.plan, buf.y.view(-1, D), rows))
        report(f"experts gate/up width {width}", found, 8, "TF_EXPERT_SPLITS")
        for rows, us in downs.items():
            print(f"{'experts down width ' + str(width):28} R={rows:<2} {us:7.1f} us (reference)", flush=True)
        results_out.append((f"experts{width}", found))
        del ex
        torch.cuda.empty_cache()


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"{torch.cuda.get_device_name()}: decode-window kernels at a TP 8 rank's shapes (us a launch, in graphs)",
          flush=True)
    results: list = []
    tune_experts(results)
    tune_hc(results)
    tune_router(results)


if __name__ == "__main__":
    main()
