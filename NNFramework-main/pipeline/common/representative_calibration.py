from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration_targets import (
    build_default_target_counts,
    resolve_calibration_sample_count,
    total_target_count,
)
from .input_data import AudioWindowDataset, LogMelFeatureExtractor, SegmentRecord, build_split_bundle, load_input_config


def load_training_config(config_path: Path) -> dict[str, Any]:
    return load_input_config(config_path)


def build_identity_quant_mean_std(rank: int) -> tuple[str, str]:
    if rank == 4:
        mean = [[[[0.0]]]]
        std = [[[[1.0]]]]
    elif rank == 3:
        mean = [[[0.0]]]
        std = [[[1.0]]]
    elif rank == 2:
        mean = [[0.0]]
        std = [[1.0]]
    else:
        raise ValueError(f"Unsupported rank for representative calibration mean/std: {rank}")
    return json.dumps(mean), json.dumps(std)


def _records_for_split(split_name: str, split_bundle) -> list[SegmentRecord]:
    split_map = {
        "train": split_bundle.train,
        "val": split_bundle.val,
        "test": split_bundle.test,
    }
    if split_name not in split_map:
        raise ValueError(f"Unsupported calibration split: {split_name}")
    records = list(split_map[split_name])
    if not records:
        raise RuntimeError(f"Calibration split is empty: {split_name}")
    return records


def _select_records_balanced(
    records: list[SegmentRecord],
    class_names: list[str],
    max_samples: int,
    seed: int,
) -> list[SegmentRecord]:
    grouped: list[list[SegmentRecord]] = [[] for _ in class_names]
    for record in records:
        grouped[record.label_index].append(record)

    for class_index, class_records in enumerate(grouped):
        rng = random.Random(seed + class_index)
        rng.shuffle(class_records)

    selected: list[SegmentRecord] = []
    made_progress = True
    while len(selected) < max_samples and made_progress:
        made_progress = False
        for class_records in grouped:
            if not class_records or len(selected) >= max_samples:
                continue
            selected.append(class_records.pop())
            made_progress = True
    if not selected:
        raise RuntimeError("No representative records were selected for calibration")
    return selected


def _count_records_by_class(
    records: list[SegmentRecord],
    class_names: list[str],
) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for record in records:
        counts[class_names[record.label_index]] += 1
    return counts


def _select_records_by_class_targets(
    records: list[SegmentRecord],
    class_names: list[str],
    target_counts: dict[str, int],
    seed: int,
) -> list[SegmentRecord]:
    grouped: list[list[SegmentRecord]] = [[] for _ in class_names]
    for record in records:
        grouped[record.label_index].append(record)

    selected_groups: list[list[SegmentRecord]] = []
    for class_index, class_name in enumerate(class_names):
        class_records = grouped[class_index]
        rng = random.Random(seed + class_index)
        rng.shuffle(class_records)
        selected_groups.append(class_records[: max(int(target_counts.get(class_name, 0)), 0)])

    selected: list[SegmentRecord] = []
    max_group_length = max((len(group) for group in selected_groups), default=0)
    for offset in range(max_group_length):
        for group in selected_groups:
            if offset < len(group):
                selected.append(group[offset])

    if not selected:
        raise RuntimeError("No representative records were selected for calibration")
    return selected


def _fit_array_shape(sample: np.ndarray, target_shape: tuple[int, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    if sample.ndim != len(target_shape):
        raise ValueError(
            "Representative calibration sample rank does not match the recovered model input contract. "
            f"sample_shape={list(sample.shape)} target_shape={list(target_shape)}"
        )

    slices = []
    pad_width = []
    adjustments = []
    for axis, (current_size, target_size) in enumerate(zip(sample.shape, target_shape, strict=True)):
        slice_stop = min(current_size, target_size)
        slices.append(slice(0, slice_stop))
        pad_after = max(target_size - current_size, 0)
        pad_width.append((0, pad_after))
        if current_size != target_size:
            adjustments.append(
                {
                    "axis": axis,
                    "from": int(current_size),
                    "to": int(target_size),
                    "mode": "crop" if current_size > target_size else "pad_zero",
                }
            )

    fitted = sample[tuple(slices)]
    if any(after > 0 for _before, after in pad_width):
        fitted = np.pad(fitted, pad_width, mode="constant", constant_values=0.0)

    return fitted.astype(np.float32, copy=False), {
        "original_shape": list(sample.shape),
        "target_shape": list(target_shape),
        "adjustments": adjustments,
    }


def _feature_to_tf_sample(feature: torch.Tensor, input_shape_nchw: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    sample = feature.detach().cpu().numpy().astype(np.float32, copy=False)

    if len(input_shape_nchw) == 4:
        expected_channels = int(input_shape_nchw[1])
        target_shape = (int(input_shape_nchw[2]), int(input_shape_nchw[3]), expected_channels)
        if sample.ndim == 3 and sample.shape[0] == expected_channels:
            return _fit_array_shape(np.transpose(sample, (1, 2, 0)), target_shape)
        if sample.ndim == 2 and expected_channels == 1:
            return _fit_array_shape(sample[:, :, np.newaxis], target_shape)
    elif len(input_shape_nchw) == 3:
        expected_channels = int(input_shape_nchw[1])
        target_shape = (int(input_shape_nchw[2]), expected_channels)
        if sample.ndim == 2 and sample.shape[0] == expected_channels:
            return _fit_array_shape(np.transpose(sample, (1, 0)), target_shape)
        if sample.ndim == 1 and expected_channels == 1:
            return _fit_array_shape(sample[:, np.newaxis], target_shape)
    elif len(input_shape_nchw) == 2:
        target_shape = (int(input_shape_nchw[1]),)
        return _fit_array_shape(sample.reshape(-1), target_shape)

    raise ValueError(
        "Representative calibration sample shape cannot be aligned to the recovered model input contract. "
        f"feature_shape={list(sample.shape)} input_shape_nchw={input_shape_nchw}"
    )


def _write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def generate_representative_calibration_npy(
    *,
    config_path: Path,
    input_shape_nchw: list[int],
    output_path: Path,
    calibration_samples: int | None,
    split_name: str = "train",
    seed: int = 42,
) -> tuple[Path, str, str, dict[str, Any]]:
    config = load_training_config(config_path)
    split_bundle = build_split_bundle(config)
    class_names = list(config["data"]["classes"])
    split_records = _records_for_split(split_name, split_bundle)
    available_class_counts = _count_records_by_class(split_records, class_names)
    default_target_counts = build_default_target_counts(class_names)

    if calibration_samples is None:
        selection_mode = "default_per_class"
        requested_samples = total_target_count(default_target_counts)
        target_counts: dict[str, int] | None = default_target_counts
        selected_records = _select_records_by_class_targets(
            records=split_records,
            class_names=class_names,
            target_counts=default_target_counts,
            seed=seed,
        )
    else:
        selection_mode = "explicit_total"
        requested_samples = resolve_calibration_sample_count(class_names, calibration_samples)
        target_counts = None
        selected_records = _select_records_balanced(
            records=split_records,
            class_names=class_names,
            max_samples=requested_samples,
            seed=seed,
        )

    feature_extractor = LogMelFeatureExtractor(config)
    dataset = AudioWindowDataset(
        selected_records,
        feature_extractor,
        augment_cfg={},
        training=False,
    )

    tf_samples: list[np.ndarray] = []
    label_counts = {class_name: 0 for class_name in class_names}
    selected_record_payload = []
    shape_adaptation_count = 0
    observed_feature_shapes: dict[str, int] = {}
    for record, (feature, _label) in zip(selected_records, dataset, strict=True):
        tf_sample, shape_metadata = _feature_to_tf_sample(feature, input_shape_nchw)
        tf_samples.append(tf_sample)
        label_name = class_names[record.label_index]
        label_counts[label_name] += 1
        shape_key = "x".join(str(value) for value in shape_metadata["original_shape"])
        observed_feature_shapes[shape_key] = observed_feature_shapes.get(shape_key, 0) + 1
        if shape_metadata["adjustments"]:
            shape_adaptation_count += 1
        selected_record_payload.append(
            {
                "path": record.path,
                "label_index": record.label_index,
                "label_name": label_name,
                "split": record.split,
                "start_sample": record.start_sample,
                "end_sample": record.end_sample,
                "shape_alignment": shape_metadata,
            }
        )

    calibration_array = np.stack(tf_samples, axis=0).astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, calibration_array)

    mean_text, std_text = build_identity_quant_mean_std(calibration_array.ndim)
    metadata = {
        "calibration_source": "representative",
        "config_path": str(config_path.resolve()),
        "data_root": str(Path(config["paths"]["data_root"]).resolve()),
        "split": split_name,
        "selection_mode": selection_mode,
        "requested_samples": int(requested_samples),
        "actual_samples": int(calibration_array.shape[0]),
        "input_shape_nchw": list(input_shape_nchw),
        "tf_calibration_shape": list(calibration_array.shape),
        "class_names": class_names,
        "default_target_counts": default_target_counts,
        "target_counts": target_counts,
        "available_class_counts": available_class_counts,
        "class_counts": label_counts,
        "observed_feature_shapes": observed_feature_shapes,
        "shape_adaptation_count": shape_adaptation_count,
        "selected_records": selected_record_payload,
    }
    metadata_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    metadata["metadata_path"] = str(_write_metadata(metadata_path, metadata))
    return output_path, mean_text, std_text, metadata
