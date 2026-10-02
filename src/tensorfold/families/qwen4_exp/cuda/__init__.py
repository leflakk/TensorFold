"""CUDA Flash Next verify rows match this backend's serial bits, which can differ from the Mac backend's."""

DEPTH = 6            # most MTP drafts a round
CONFIDENCE = 0.7     # a chain ends before a draft the MTP head gives less than this (one stream or many)
TP_CONFIDENCE = 0.3  # the same cut on tensor-parallel ranks (measured on 8 RTX 3090s: 0.3 >= 0.7 in every cell)
CONTEXT = 8192       # prompt plus reply tokens the caches hold
