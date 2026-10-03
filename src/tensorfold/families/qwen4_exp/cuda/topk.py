"""A rank's sampling candidates in two Triton passes: each 1024-logit block's top K, then their merge and the row's
log-sum-exp. Equal logits go lowest id first (deterministic, unlike an unsorted torch.topk); the candidates feed
``forward.candidates``' gather, which only needs each rank's top K values with their ids.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 1024
BIG = tl.constexpr(1 << 30)          # a constexpr global: compiled Triton refuses plain module globals


@triton.jit
def _block_top(L, ld, n, VALS, IDS, MAXS, SUMS, NB, K: tl.constexpr, BLOCK: tl.constexpr):
    """Program (row, block): the block's top K logits and ids (largest first, lowest id among equals), its max and
    its sum of exp(logit - max)."""

    r = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    ok = offs < n
    v = tl.load(L + r * ld + offs, mask=ok, other=float("-inf")).to(tl.float32)
    m = tl.max(v, axis=0)
    tl.store(MAXS + r * NB + blk, m)
    tl.store(SUMS + r * NB + blk, tl.sum(tl.where(ok, tl.exp(v - m), 0.0), axis=0))
    kk = tl.arange(0, K)
    vals = tl.full((K,), float("-inf"), tl.float32)
    ids = tl.zeros((K,), tl.int32)
    for j in tl.static_range(K):
        mj = tl.max(v, axis=0)
        ij = tl.min(tl.where(v == mj, offs, BIG), axis=0)
        vals = tl.where(kk == j, mj, vals)
        ids = tl.where(kk == j, ij, ids)
        v = tl.where(offs == ij, float("-inf"), v)
    tl.store(VALS + (r * NB + blk) * K + kk, vals)
    tl.store(IDS + (r * NB + blk) * K + kk, ids)


@triton.jit
def _merge_top(VALS, IDS, MAXS, SUMS, NB, OUT, MAP, offset, K: tl.constexpr, NBP: tl.constexpr,
               HAS_MAP: tl.constexpr):
    """Program r: the row's top K over its blocks' (lowest id among equals), the log-sum-exp from the blocks' maxima
    and sums, written as ``forward.candidates`` lays them out: K values, K global ids as fp32 bits, the lse."""

    r = tl.program_id(0)
    c = tl.arange(0, NBP * K)
    ok = c < NB * K
    v = tl.load(VALS + r * NB * K + c, mask=ok, other=float("-inf"))
    i = tl.load(IDS + r * NB * K + c, mask=ok, other=BIG)
    kk = tl.arange(0, K)
    ov = tl.full((K,), float("-inf"), tl.float32)
    oi = tl.zeros((K,), tl.int32)
    for j in tl.static_range(K):
        mj = tl.max(v, axis=0)
        ij = tl.min(tl.where(v == mj, i, BIG), axis=0)
        ov = tl.where(kk == j, mj, ov)
        oi = tl.where(kk == j, ij, oi)
        v = tl.where((v == mj) & (i == ij), float("-inf"), v)
    b = tl.arange(0, NBP)
    bok = b < NB
    ms = tl.load(MAXS + r * NB + b, mask=bok, other=float("-inf"))
    ss = tl.load(SUMS + r * NB + b, mask=bok, other=0.0)
    top = tl.max(ms, axis=0)
    lse = top + tl.log(tl.sum(tl.where(bok, ss * tl.exp(ms - top), 0.0), axis=0))
    if HAS_MAP:
        gid = tl.load(MAP + oi).to(tl.int32)
    else:
        gid = oi + offset
    width = 2 * K + 1
    tl.store(OUT + r * width + kk, ov)
    tl.store(OUT + r * width + K + kk, gid.to(tl.float32, bitcast=True))
    tl.store(OUT + r * width + 2 * K, lse)


def candidates(logits: torch.Tensor, out: torch.Tensor, k: int, *, offset: int = 0,
               id_map: torch.Tensor | None = None, scratch: dict | None = None) -> torch.Tensor:
    """logits [R, n] (bf16 or fp32) -> out [R, 2k + 1] fp32: each row's top k logits, their global ids (fp32 bits),
    the row's log-sum-exp. ``scratch``: the caller's own dict for the blocks' partials (one a rank: ranks that are
    threads of one process must not share it)."""

    rows, n = logits.shape
    nb = triton.cdiv(n, BLOCK)
    key = (logits.device, rows, nb, k)
    scratch = {} if scratch is None else scratch
    got = scratch.get(key)
    if got is None:
        dev = logits.device
        got = scratch[key] = (torch.empty((rows * nb * k,), dtype=torch.float32, device=dev),
                                   torch.empty((rows * nb * k,), dtype=torch.int32, device=dev),
                                   torch.empty((rows * nb,), dtype=torch.float32, device=dev),
                                   torch.empty((rows * nb,), dtype=torch.float32, device=dev))
    vals, ids, maxs, sums = got
    _block_top[(rows, nb)](logits, logits.stride(0), n, vals, ids, maxs, sums, nb, K=k, BLOCK=BLOCK, num_warps=4)
    _merge_top[(rows,)](vals, ids, maxs, sums, nb, out, id_map if id_map is not None else ids, int(offset), K=k,
                        NBP=triton.next_power_of_2(nb), HAS_MAP=id_map is not None, num_warps=4)
    return out
