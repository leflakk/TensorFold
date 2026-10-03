"""CUDA Flash Next verify rows match this backend's serial bits, which can differ from the Mac backend's."""

DEPTH = 6            # most MTP drafts a round
CONFIDENCE = 0.7     # a chain ends before a draft the MTP head gives less than this (one stream or many)
# the same cut on tensor-parallel ranks (8 RTX 3090s, TP 8: 0.3 >= 0.7 in every bench cell; then 0.5 > 0.3 in three
# of four, 3.7% faster on average, as for one stream under --parallel)
TP_CONFIDENCE = 0.5
# --parallel on tensor-parallel ranks: the cut for drafts a round of 1, 2, 3 or more streams verifies. A round's rows
# all wait for each other, so the more streams share it, the less a doubtful draft is worth (8 RTX 3090s, TP 8,
# --parallel 4: one stream at a time, 0.5 matched or beat 0.3 and 0.7 in all four bench cells; four at once, 0.7 beat
# 0.3 by 6-19% and 0.5 in three of four)
TP_STREAM_CONFIDENCE = (TP_CONFIDENCE, 0.6, 0.7)
CONTEXT = 8192       # prompt plus reply tokens the caches hold


def stream_confidence(default: tuple[float, ...]) -> float | tuple[float, ...]:
    """--parallel's cut by the streams of the round that verifies the drafts: ``TF_MULTI_CONFIDENCE`` (one
    probability, or one per stream count from 1, comma-separated, the last for more streams), else ``default``.
    One value comes back as a float."""

    import os

    text = os.environ.get("TF_MULTI_CONFIDENCE", "").strip()
    try:
        cuts = tuple(float(v) for v in text.split(",")) if text else tuple(float(c) for c in default)
    except ValueError:
        cuts = ()
    if not cuts or not all(0.0 <= c <= 1.0 for c in cuts):
        raise ValueError(f"TF_MULTI_CONFIDENCE={text!r}: probabilities from 0 to 1, one per stream count from 1 "
                         "(for example 0.5,0.6,0.7)")
    return cuts[0] if len(cuts) == 1 else cuts
