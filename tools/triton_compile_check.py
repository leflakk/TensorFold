"""Compile the rtx3090 branch's Triton kernels for sm_86 without a GPU (the real compiler, not the interpreter)."""
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from tensorfold.families.qwen4_exp.cuda import attention, attn_multi, glue, qmm, topk

CASES = [
    ("topk._block_top", topk._block_top,
     {"L": "*bf16", "ld": "i32", "n": "i32", "VALS": "*fp32", "IDS": "*i32", "MAXS": "*fp32", "SUMS": "*fp32",
      "NB": "i32", "K": "constexpr", "BLOCK": "constexpr"}, {"K": 32, "BLOCK": 1024}),
    ("topk._merge_top", topk._merge_top,
     {"VALS": "*fp32", "IDS": "*i32", "MAXS": "*fp32", "SUMS": "*fp32", "NB": "i32", "OUT": "*fp32", "MAP": "*i64",
      "offset": "i32", "K": "constexpr", "NBP": "constexpr", "HAS_MAP": "constexpr"},
     {"K": 32, "NBP": 32, "HAS_MAP": True}),
    ("topk._merge_top(no map)", topk._merge_top,
     {"VALS": "*fp32", "IDS": "*i32", "MAXS": "*fp32", "SUMS": "*fp32", "NB": "i32", "OUT": "*fp32", "MAP": "*i32",
      "offset": "i32", "K": "constexpr", "NBP": "constexpr", "HAS_MAP": "constexpr"},
     {"K": 32, "NBP": 32, "HAS_MAP": False}),
    ("qmm._qmm_upterm", qmm._qmm_upterm,
     {"X": "*bf16", "XS": "*fp32", "W": "*i32", "S": "*bf16", "B": "*bf16", "NORMED": "*bf16", "TERMS": "*bf16",
      "M": "i32", "N": "constexpr", "K": "constexpr", "D": "constexpr", "BM": "constexpr", "DB": "constexpr",
      "GPI": "constexpr", "SBN": "constexpr"},
     {"N": 10240, "K": 320, "D": 2560, "BM": 16, "DB": 64, "GPI": 2, "SBN": 64}),
    ("qmm._mix_terms", qmm._mix_terms,
     {"TERMS": "*bf16", "MIXED": "*bf16", "XSM": "*fp32", "M": "i32", "D": "constexpr", "SS": "constexpr",
      "BM": "constexpr", "DB": "constexpr"}, {"D": 2560, "SS": 4, "BM": 16, "DB": 64}),
    ("qmm._qmm_upmix", qmm._qmm_upmix,
     {"X": "*bf16", "XS": "*fp32", "W": "*i32", "S": "*bf16", "B": "*bf16", "NORMED": "*bf16", "MIXED": "*bf16",
      "XSM": "*fp32", "M": "i32", "N": "constexpr", "K": "constexpr", "D": "constexpr", "SS": "constexpr",
      "BM": "constexpr", "DB": "constexpr", "GPI": "constexpr", "SBN": "constexpr"},
     {"N": 10240, "K": 320, "D": 2560, "SS": 4, "BM": 16, "DB": 32, "GPI": 2, "SBN": 64}),
    ("glue._hc_writeback", glue._hc_writeback,
     {"H": "*bf16", "HOUT": "*bf16", "PSS": "*fp32", "BR": "*bf16", "INJ": "*bf16", "Y": "*bf16", "WTS": "*fp32",
      "RS": "i32", "D": "constexpr", "S": "constexpr", "MODE": "constexpr", "TOPK": "constexpr", "SLOTS": "constexpr",
      "BLOCK": "constexpr", "WORLD": "constexpr"},
     {"D": 2560, "S": 4, "MODE": 1, "TOPK": 1, "SLOTS": 1, "BLOCK": 256, "WORLD": 1}),
    ("glue._ple_conv_multi", glue._ple_conv_multi,
     {"GATED": "*bf16", "PSS": "*fp32", "NC": "*bf16", "TAILS": "*i64", "SID": "*i32", "POSR": "*i32",
      "FIRST": "*i32", "CW": "*bf16", "H": "*bf16", "HOUT": "*bf16", "NROW": "*bf16", "eps": "fp32", "R": "i32",
      "D": "constexpr", "S": "constexpr", "TAPS": "constexpr", "DIL": "constexpr", "BLOCK": "constexpr"},
     {"D": 2560, "S": 4, "TAPS": 4, "DIL": 3, "BLOCK": 512}),
    ("attn_multi._scores_multi", attn_multi._scores_multi,
     {"IQ": "*bf16", "CP": "*i64", "POSR": "*i32", "SID": "*i32", "SC": "*fp32", "NB": "i32", "N": "i32",
      "HI": "constexpr", "DI": "constexpr", "RATIO": "constexpr", "TOP": "constexpr", "BB": "constexpr"},
     {"HI": 4, "DI": 128, "RATIO": 4, "TOP": 512, "BB": 64}),
    ("attention._select(rowpos)", attention._select,
     {"SC": "*fp32", "POS0": "*i32", "IDS": "*i32", "NKR": "*i32", "SPR": "*i32", "NB": "i32", "RATIO": "constexpr",
      "TOP": "constexpr", "IDW": "constexpr", "BLOCK": "constexpr", "ROWPOS": "constexpr"},
     {"RATIO": 4, "TOP": 512, "IDW": 2052, "BLOCK": 4096, "ROWPOS": True}),
    ("attention._select_tiles(rowpos)", attention._select_tiles,
     {"SC": "*fp32", "POS0": "*i32", "IDS": "*i32", "NKR": "*i32", "SPR": "*i32", "NB": "i32", "RATIO": "constexpr",
      "TOP": "constexpr", "IDW": "constexpr", "TB": "constexpr", "ROWPOS": "constexpr"},
     {"RATIO": 4, "TOP": 512, "IDW": 2052, "TB": 8192, "ROWPOS": True}),
]
bad = 0
for name, fn, sig, consts in CASES:
    try:
        triton.compile(ASTSource(fn, sig, constexprs=consts), target=GPUTarget("cuda", 86, 32))
        print("ok  ", name, flush=True)
    except Exception as exc:  # noqa: BLE001
        bad += 1
        print("FAIL", name, type(exc).__name__, str(exc).splitlines()[0][:200], flush=True)
sys.exit(1 if bad else 0)
