"""How Flash Next splits over 1, 2, 4 or 8 tensor-parallel ranks (one process a GPU).

Query heads, DeltaNet heads and the head's vocabulary split evenly. The 2 KV heads are held whole: with more ranks
than KV heads, rank r holds KV head r * kv // world (its query heads' group). Expert widths (640) split in whole
32-input groups, earlier ranks taking one more when they do not divide: 8 ranks hold 96, 96, 96, 96, 64, 64, 64, 64.
"""

from __future__ import annotations

WORLDS = (1, 2, 4, 8)


def span(total: int, world: int, rank: int, unit: int = 1) -> tuple[int, int]:
    """Rank ``rank``'s [lo, hi) of ``total`` in whole ``unit``s, the remainder going to the first ranks."""

    if total % unit:
        raise ValueError(f"{total} is not a whole number of {unit}-wide units")
    units = total // unit
    if units < world:
        raise ValueError(f"{total} in units of {unit} cannot give each of {world} ranks a share")
    base, extra = divmod(units, world)
    lo = rank * base + min(rank, extra)
    return lo * unit, (lo + base + (rank < extra)) * unit


def kv_span(kv_heads: int, world: int, rank: int) -> tuple[int, int]:
    """(first KV head, count) a rank holds: its share when the ranks divide the KV heads, else its group's head."""

    if world <= kv_heads:
        if kv_heads % world:
            raise ValueError(f"{kv_heads} KV heads do not split over {world} ranks")
        return rank * (kv_heads // world), kv_heads // world
    if world % kv_heads:
        raise ValueError(f"{world} ranks do not share {kv_heads} KV heads evenly")
    return rank * kv_heads // world, 1


def check(cfg, world: int) -> None:
    """Refuse a rank count the checkpoint's shapes do not split into (before any weight loads)."""

    if world not in WORLDS:
        raise ValueError(f"--tp {world}: Flash Next runs on {', '.join(map(str, WORLDS))} GPUs")
    for name, value in (("attention heads", cfg.heads), ("DeltaNet key heads", cfg.nk),
                        ("DeltaNet value heads", cfg.nv), ("vocabulary rows", cfg.vocab)):
        if value % world:
            raise ValueError(f"--tp {world}: the {value} {name} do not split over {world} GPUs")
    kv_span(cfg.kv_heads, world, 0)
    if cfg.heads // world > 16:
        raise ValueError("the attention kernels hold at most 16 query heads a KV head on a rank")
    for name, width in (("expert", cfg.moe_width), ("shared expert", cfg.shared_width)):
        span(width, world, 0, 32)
        if span(width, world, world - 1, 32)[1] - span(width, world, world - 1, 32)[0] < 32:
            raise ValueError(f"--tp {world}: the {name} width {width} leaves a rank no 32-wide group")


def widths(width: int, world: int) -> list[int]:
    """Every rank's expert width (the last rank's is the narrowest)."""

    return [hi - lo for lo, hi in (span(width, world, r, 32) for r in range(world))]
