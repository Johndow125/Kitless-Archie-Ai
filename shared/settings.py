from __future__ import annotations

TARGET_LOSS = 1e-4
MASTERED_LOSS = 0.01

CHAT_MODE_LABELS = {
    "trained": "training",
    "trained+brain": "brain+training",
    "brain": "brain",
    "linear": "network",
    "swarm": "swarm",
    "direct": "quick",
    "none": "offline",
}


def chat_mode_label(mode: str) -> str:
    return CHAT_MODE_LABELS.get(str(mode or "").strip(), str(mode or "server"))


def training_loss_accepted(before: float, after: float) -> bool:
    if after < before - 1e-12:
        return True
    if before <= MASTERED_LOSS and after <= MASTERED_LOSS:
        return True
    return False