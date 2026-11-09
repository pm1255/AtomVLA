"""Deterministic branch dropout; unaffected by DataLoader prefetch or restart."""
import hashlib
import math


def gates(seed, stage, step, micro, query_dropout=0.2, pixel_dropout=0.5,
          pose_dropout=0.5, evaluate=False):
    for p in [query_dropout, pixel_dropout, pose_dropout]:
        if not 0 <= p <= 1:
            raise ValueError("Dropout probability outside [0,1]")
    if evaluate:
        return dict(query=True, pixel=True, pose=True)

    def keep(name, probability):
        value = int.from_bytes(
            hashlib.sha256(f"{seed}:{stage}:{step}:{micro}:{name}".encode()).digest()[:8],
            "big",
        )
        return value / (2**64) >= probability

    if stage == 1:
        query = keep("query", query_dropout)
        return dict(query=query, pixel=query, pose=query)
    if stage != 2:
        raise ValueError("Only two stages")
    return dict(query=True, pixel=keep("pixel", pixel_dropout), pose=keep("pose", pose_dropout))


def lr_multiplier(step, total, warmup, floor=0.1):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1, max(0, (step - warmup) / max(1, total - warmup)))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
