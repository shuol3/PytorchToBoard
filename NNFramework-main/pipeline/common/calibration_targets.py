from __future__ import annotations

from collections.abc import Mapping, Sequence


DEFAULT_TARGET_PER_CLASS = 30
DEFAULT_OTHER_TARGET = 50


def default_target_count_for_class(class_name: str) -> int:
    return DEFAULT_OTHER_TARGET if class_name.strip().lower() == "other" else DEFAULT_TARGET_PER_CLASS


def build_default_target_counts(class_names: Sequence[str]) -> dict[str, int]:
    return {
        str(class_name): default_target_count_for_class(str(class_name))
        for class_name in class_names
    }


def apply_target_count_overrides(
    class_names: Sequence[str],
    overrides: Mapping[str, int] | None = None,
) -> dict[str, int]:
    targets = build_default_target_counts(class_names)
    if not overrides:
        return targets

    for class_name, value in overrides.items():
        if class_name not in targets:
            raise ValueError(f"Unknown class override: {class_name}")
        normalized_value = int(value)
        if normalized_value < 0:
            raise ValueError(f"Target count must not be negative: {class_name}={normalized_value}")
        targets[class_name] = normalized_value
    return targets


def parse_target_count_text(
    class_names: Sequence[str],
    per_class_text: str | None,
) -> dict[str, int]:
    if not per_class_text:
        return build_default_target_counts(class_names)

    overrides: dict[str, int] = {}
    for chunk in per_class_text.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid per-class entry: {entry}")
        class_name, value_text = entry.split("=", 1)
        normalized_name = class_name.strip()
        normalized_value = int(value_text.strip())
        if normalized_value < 0:
            raise ValueError(f"Target count must not be negative: {normalized_name}={normalized_value}")
        overrides[normalized_name] = normalized_value

    return apply_target_count_overrides(class_names, overrides)


def total_target_count(target_counts: Mapping[str, int]) -> int:
    return sum(max(int(value), 0) for value in target_counts.values())


def default_total_calibration_samples(class_names: Sequence[str]) -> int:
    return total_target_count(build_default_target_counts(class_names))


def resolve_calibration_sample_count(
    class_names: Sequence[str],
    calibration_samples: int | None,
) -> int:
    if calibration_samples is not None:
        return max(int(calibration_samples), 1)
    return max(default_total_calibration_samples(class_names), 1)
