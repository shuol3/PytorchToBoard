"""Identify PyTorch checkpoint kinds and extract static hints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..common.checkpoint_loader import (
    extract_pickle_strings,
    extract_serialized_global_refs,
    is_torchscript_archive,
    serialized_user_global_refs,
)

from ..types import CheckpointKind, ModelIdentity


def inspect_checkpoint(path: Path) -> ModelIdentity:
    checkpoint_kind, reasons = detect_checkpoint_kind(path)
    return ModelIdentity(
        framework="torch",
        checkpoint_kind=checkpoint_kind,
        reasons=reasons,
    )


def detect_checkpoint_kind(path: Path) -> tuple[CheckpointKind, list[str]]:
    if is_torchscript_archive(path):
        return "torchscript", ["Archive contains TorchScript code/constants entries."]

    strings = extract_pickle_strings(path)
    user_globals = serialized_user_global_refs(path)
    if user_globals:
        preview = ", ".join(f"{module}.{name}" for module, name in user_globals[:3])
        return "full_module", [f"Serialized user module globals detected: {preview}"]

    lower_strings = [text.lower() for text in strings]
    state_dict_hits = sum(
        1
        for text in lower_strings
        if any(keyword in text for keyword in ("weight", "bias", "running_mean", "running_var"))
    )
    checkpoint_hits = sum(
        1
        for text in lower_strings
        if text in {"model_state_dict", "optimizer_state_dict", "epoch", "scheduler", "state_dict"}
    )

    if checkpoint_hits >= 2:
        return (
            "training_checkpoint",
            ["Detected training-checkpoint keys such as epoch/optimizer/model_state_dict."],
        )
    if state_dict_hits >= 4:
        return (
            "state_dict",
            ["Detected parameter-like tensor keys, suggesting a state_dict payload."],
        )

    hints = detect_architecture_hints(path)
    reasons: list[str] = []
    if hints:
        reasons.append("Architecture hints found: " + ", ".join(hints))
    reasons.append("Checkpoint type could not be proven statically; restore stage will resolve it.")
    return "unknown", reasons


def extract_checkpoint_hints(path: Path) -> dict[str, object]:
    serialized_globals = extract_serialized_global_refs(path)
    return {
        "is_torchscript_archive": is_torchscript_archive(path),
        "serialized_globals": [f"{module}.{name}" for module, name in serialized_globals[:20]],
        "architecture_hints": detect_architecture_hints(path),
    }


def detect_architecture_hints(path: Path) -> list[str]:
    strings = extract_pickle_strings(path)
    return _detect_architecture_hints_from_strings(strings)


def _detect_architecture_hints_from_strings(strings: list[str]) -> list[str]:
    hints = Counter()
    for text in strings:
        lower = text.lower()
        if "conv" in lower:
            hints["conv"] += 1
        if "bn" in lower or "batchnorm" in lower:
            hints["batchnorm"] += 1
        if "lstm" in lower:
            hints["lstm"] += 1
        if "gru" in lower:
            hints["gru"] += 1
        if "linear" in lower or "fc" in lower:
            hints["linear"] += 1
        if "embedding" in lower:
            hints["embedding"] += 1
        if "mel" in lower:
            hints["mel"] += 1
        if "audio" in lower:
            hints["audio"] += 1
    return [name for name, _count in hints.most_common(8)]
