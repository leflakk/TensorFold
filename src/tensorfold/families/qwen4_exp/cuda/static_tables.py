"""Device tables a captured concurrent step reads: allocated once a shape, refilled before each replay."""

from __future__ import annotations

import numpy as np
import torch

_DTYPES = {np.dtype(np.int32): torch.int32, np.dtype(np.int64): torch.int64}


class StaticTables:
    """Named device tables of fixed sizes (a CUDA graph holds their addresses), each filled from a pinned host twin
    by one non-blocking copy. A table is refilled once a step, after the previous step's read-back (its copy has
    landed by then), so a twin is never overwritten while its copy is in flight."""

    def __init__(self, device) -> None:
        self.device = device
        self.dev: dict[str, torch.Tensor] = {}
        self.host: dict[str, torch.Tensor] = {}

    def put(self, name: str, values) -> torch.Tensor:
        values = np.ascontiguousarray(values).reshape(-1)
        dtype = _DTYPES[values.dtype]
        d = self.dev.get(name)
        if d is None:
            d = self.dev[name] = torch.empty((values.size,), dtype=dtype, device=self.device)
            self.host[name] = torch.empty((values.size,), dtype=dtype, pin_memory=torch.cuda.is_available())
        elif d.numel() != values.size or d.dtype != dtype:
            raise ValueError(f"static table {name!r}: {values.size} {dtype} values for {d.numel()} {d.dtype}")
        h = self.host[name]
        h.numpy()[:] = values
        d.copy_(h, non_blocking=True)
        return d
