"""NCCL all-gather on the current stream so CUDA graphs capture it; a rank-order sum after it keeps ranks bit-equal."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os

import torch

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4}


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
    if os.name == "nt":
        raise RuntimeError("TensorFold does not run tensor-parallel (NCCL) on Windows: CUDA on Windows has no "
                           "libnccl to wrap; use one GPU per process there")
    candidates = [os.environ.get("TF_NCCL_LIB", "")]
    found = ctypes.util.find_library("nccl")
    if found:
        candidates.append(found)
    candidates += glob.glob("/usr/lib/*/libnccl.so.2") + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/nccl/lib/libnccl.so.2")
    candidates += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl*.so*"))
    # pip wheels (a venv, no Docker): nvidia/nccl/lib, or nvidia/cu13/lib in CUDA 13's layout, beside torch
    site = os.path.dirname(os.path.dirname(torch.__file__))
    candidates += sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib", "libnccl.so*")))
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError("libnccl not found (set TF_NCCL_LIB)")


BIG = 1 << 20          # bytes from which a reduce takes NCCL's all-reduce (prompt chunks), unless TF_PREFILL_COMM=shm


def _big_comm() -> str:
    value = os.environ.get("TF_PREFILL_COMM", "nccl").strip().lower()
    if value not in ("nccl", "shm"):
        raise ValueError(f"TF_PREFILL_COMM={value!r}: nccl or shm")
    return value


class NCCL:
    def __init__(self, rank: int, world: int, master: str, port: int, *, local: bool = False) -> None:
        from datetime import timedelta

        from torch.distributed import TCPStore

        self.rank, self.world = rank, world
        self.local = bool(local)          # every rank on this host (one process a GPU): shared memory reaches them
        self.lib = _library()
        lib = self.lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
        lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p]
        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=600))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set("tf_nccl_uid", bytes(uid.internal))
        else:
            raw = self.store.get("tf_nccl_uid")
            ctypes.memmove(ctypes.addressof(uid), raw, 128)
        self.comm = ctypes.c_void_p()
        torch.cuda.current_device()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank))

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NCCL error {code}: {self.lib.ncclGetErrorString(code).decode()}")

    fast = None          # a ``fastcomm.FastComm`` once ``use_fast`` set it up (ranks on one host)

    def use_fast(self, cap: int) -> bool:
        """Route collectives through shared host memory (ranks on this host, TF_COMM=shm); False keeps NCCL."""

        from tensorfold.cuda import fastcomm

        if self.world == 1 or not fastcomm.enabled() or not getattr(self, "local", False):
            return False
        self.fast = fastcomm.FastComm(self.store, self.rank, self.world, cap)
        return True

    def reduce(self, part: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """out = every rank's ``part`` summed in rank order in fp32 (shared memory only; NCCL callers gather)."""

        if self.fast is None:
            raise RuntimeError("reduce runs on the shared-memory collectives (TF_COMM=shm, ranks on one host)")
        nbytes = part.numel() * part.element_size()
        if nbytes > BIG and part.dtype == out.dtype and _big_comm() == "nccl":
            # a prompt chunk's partials (MBs): NCCL's rings move them faster over PCIe (measured 4.7 ms against
            # 7.0 for 10.5 MB on 8 RTX 3090s); its sum order depends on the size, so a resumed prompt can round
            # differently from a fresh one (TF_PREFILL_COMM=shm keeps the rank-ordered sum)
            stream = torch.cuda.current_stream().cuda_stream
            self._check(self.lib.ncclAllReduce(part.data_ptr(), out.data_ptr(), part.numel(), _DTYPES[part.dtype],
                                               0, self.comm, stream))
            return out
        return self.fast.reduce(part, out)

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order (contiguous tensors, same dtype)."""

        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        if self.fast is not None and send.is_contiguous() and (send.numel() * send.element_size()) % 4 == 0 \
                and send.numel() * send.element_size() <= self.fast.cap:
            self.fast.all_gather(send, recv)
            return
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclAllGather(send.data_ptr(), recv.data_ptr(), send.numel(), _DTYPES[send.dtype],
                                           self.comm, stream))

    def ready(self, label: str, *, every: float = 60.0, timeout: float = 3600.0) -> None:
        """Every rank finishes ``label`` before any goes on; a rank missing after ``timeout`` s is named."""

        import time
        from datetime import timedelta

        self.store.set(f"tf_ready/{label}/{self.rank}", "1")
        others = [r for r in range(self.world) if r != self.rank]
        started = time.monotonic()
        while True:
            try:
                self.store.wait([f"tf_ready/{label}/{r}" for r in others], timedelta(seconds=every))
                return
            except Exception as exc:                  # noqa: BLE001  (the store's timeout; anything else goes up)
                if "timeout" not in str(exc).lower():
                    raise
            waited = time.monotonic() - started
            missing = ", ".join(str(r) for r in others)
            if waited >= timeout:
                raise RuntimeError(f"rank {self.rank} finished {label} but rank {missing} has not after "
                                   f"{waited / 60:.0f} min: check that rank's log (a CUDA extension build waiting on "
                                   "a lock names the lock there)")
            print(f"[tensorfold] rank {self.rank} finished {label}; waiting for rank {missing} ({waited:.0f} s)",
                  flush=True)

    def barrier(self) -> None:
        x = torch.zeros((1,), dtype=torch.float32, device="cuda")
        y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
        self.all_gather(x, y)
        torch.cuda.synchronize()
