"""Rank-ordered all-reduce and all-gather for ranks on one host, through pinned shared host memory (no P2P needed).

Every call is one kernel launch, capturable in a CUDA graph. An all-reduce sums each element over the ranks in rank
order in fp32 (p_0 + p_1 + ... + p_{W-1}), so a row's bits never depend on the call's size and equal the NCCL
all-gather followed by ``glue.hc_writeback``'s ordered sum. ``TF_COMM=nccl`` keeps NCCL for everything.
"""

from __future__ import annotations

import os
import secrets
import time
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_fastcomm_v1", sources=[str(here / "fastcomm.cpp"), str(here / "fastcomm.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def enabled() -> bool:
    """Whether same-host ranks use the shared-memory collectives (``TF_COMM``: shm, the default, or nccl)."""

    value = os.environ.get("TF_COMM", "shm").strip().lower()
    if value not in ("shm", "nccl"):
        raise ValueError(f"TF_COMM={value!r}: shm or nccl")
    return value == "shm"


LL_MAX = 512 << 10     # bytes of input up to which ``auto`` takes the LL forms (decode windows: 10-160 KB)


def _round16(n: int) -> int:
    return -(-int(n) // 16) * 16


class FastComm:
    """Collectives over one shared region of ``cap`` bytes a rank (the largest input a call takes)."""

    def __init__(self, store, rank: int, world: int, cap: int, *, tag: str = "main",
                 timeout: float | None = None) -> None:
        if not 1 <= world <= 8:
            raise ValueError("fastcomm runs 1 to 8 ranks")
        self.rank, self.world = int(rank), int(world)
        self.cap = _round16(cap)
        # a reduced slice: at most ceil(cap / 2 bytes / world) elements (bf16 in), held as fp32
        self.rcap = _round16((self.cap // 2 // self.world + 16) * 4)
        self.timeout = float(timeout if timeout is not None else os.environ.get("TF_FASTCOMM_TIMEOUT", "600"))
        self.blocks = int(os.environ.get("TF_FASTCOMM_BLOCKS", "0"))      # 0: by size; else exactly this many
        self.shot = os.environ.get("TF_FASTCOMM_SHOT", "auto")
        self.gather_ll = os.environ.get("TF_FASTCOMM_GATHER_LL", "1") != "0"
        ext = _ext()
        self.bytes = int(ext.region_bytes(self.cap, self.rcap))
        key = f"tf_fastcomm/{tag}"
        if self.rank == 0:
            name = f"/tf_fc_{os.getpid()}_{secrets.token_hex(4)}"
            self.host, self.region = ext.open_region(name, self.bytes, True)
            store.set(f"{key}/name", name)
        else:
            name = store.get(f"{key}/name").decode()
            self.host, self.region = ext.open_region(name, self.bytes, False)
        store.set(f"{key}/opened/{self.rank}", "1")
        if self.rank == 0:                          # every rank mapped it: the name can go (the mappings stay)
            from datetime import timedelta

            store.wait([f"{key}/opened/{r}" for r in range(self.world)], timedelta(seconds=600))
            ext.unlink_region(name)
        self.name = name
        self.epoch = torch.zeros((2,), dtype=torch.int32, device="cuda")

    def _mode(self, nbytes: int, fp32: bool) -> int:
        """1/2: one/two-shot with flags; 3/4: one/two-shot LL (data and call number in each 8 bytes: no fence)."""

        if self.shot in ("1", "2", "3", "4"):
            return int(self.shot)
        ll = 2 * nbytes <= min(self.cap, LL_MAX)          # LL moves twice the bytes (a call number in each 8)
        if ll:
            return 3 if self.world <= 2 else 4
        return 1 if self.world <= 2 else 2

    def reduce(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """out (fp32 or bf16) = the rank-ordered fp32 sum of every rank's x (fp32 or bf16), the same on every rank."""

        if self.world == 1:
            out.copy_(x)
            return out
        _ext().reduce(x, out, self.region, self.epoch, self.cap, self.rcap, self.rank, self.world,
                      self._mode(x.numel() * x.element_size(), x.dtype == torch.float32), self.blocks, self.timeout)
        return out

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order (the NCCL wrapper's contract)."""

        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        if self.world == 1:
            recv.copy_(send.reshape(-1))
            return
        _ext().gather(send, recv, self.region, self.epoch, self.cap, self.rcap, self.rank, self.world,
                      self.blocks, self.timeout, self.gather_ll)

    def close(self) -> None:
        if getattr(self, "host", None):
            torch.cuda.synchronize()
            _ext().close_region(self.host, self.bytes)
            self.host = None


def bench(fn, iters: int = 200) -> float:
    """Microseconds a call, after a warm-up (the caller's ranks run it in step)."""

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


__all__ = ["FastComm", "enabled", "bench"]
