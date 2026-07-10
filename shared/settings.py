from __future__ import annotations

TARGET_LOSS = 1e-5


def training_loss_accepted(before: float, after: float) -> bool:
    if after < before:
        return True
    return before <= TARGET_LOSS and after <= TARGET_LOSS
