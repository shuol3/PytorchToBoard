from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import wave

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.runtime_env import ensure_default_python_for_script

ensure_default_python_for_script(__file__)

import numpy as np
import torch

from pipeline.common.calibration_targets import parse_target_count_text
from pipeline.common.input_data import (
    LogMelFeatureExtractor,
    collect_class_files,
    enumerate_segment_starts,
    load_wav_mono,
    read_yaml,
    resample_waveform,
    set_random_seed,
)


ROOT = SCRIPT_DIR
DEFAULT_CONFIG = ROOT / "train_1s.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "calibration_data"
DEFAULT_MAX_WINDOWS_PER_FILE = 3
DEFAULT_MIN_RMS = 0.003
DEFAULT_MAX_CLIP_RATIO = 0.01
DEFAULT_MAX_ZERO_RATIO = 0.98
DEFAULT_SEED = 42
ENERGY_BUCKETS = ("low", "mid", "high")


@dataclass(frozen=True)
class WindowCandidate:
    class_name: str
    source_file: str
    source_sample_rate_hz: int
    start_sample: int
    end_sample: int
    window_samples: int
    rms: float
    peak: float
    zero_ratio: float
    clip_ratio: float
    fingerprint: str

    @property
    def start_sec(self) -> float:
        return self.start_sample / float(self.source_sample_rate_hz)

    @property
    def end_sec(self) -> float:
        return self.end_sample / float(self.source_sample_rate_hz)


@dataclass(frozen=True)
class SelectedWindow:
    candidate: WindowCandidate
    energy_bucket: str
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a representative calibration dataset from in/data using the same "
            "audio frontend contract as the deployment pipeline."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training YAML config path")
    parser.add_argument("--data-root", help="Override config.paths.data_root")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Calibration output directory")
    parser.add_argument(
        "--per-class",
        help=(
            "Optional per-class target counts, for example "
            "'eat=40,drink=40,other=60'. Unspecified classes fall back to defaults."
        ),
    )
    parser.add_argument(
        "--hop-sec",
        type=float,
        help="Optional calibration window hop in seconds. Defaults to config.audio.hop_sec.",
    )
    parser.add_argument(
        "--min-gap-sec-per-file",
        type=float,
        help="Minimum spacing between selected windows from the same source file. Defaults to config.audio.window_sec.",
    )
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_FILE,
        help="Maximum number of selected windows taken from the same source file.",
    )
    parser.add_argument("--min-rms", type=float, default=DEFAULT_MIN_RMS, help="Reject windows below this RMS")
    parser.add_argument(
        "--max-clip-ratio",
        type=float,
        default=DEFAULT_MAX_CLIP_RATIO,
        help="Reject windows whose clipped-sample ratio exceeds this threshold",
    )
    parser.add_argument(
        "--max-zero-ratio",
        type=float,
        default=DEFAULT_MAX_ZERO_RATIO,
        help="Reject windows whose exact-zero ratio exceeds this threshold",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for stable selection ordering")
    return parser.parse_args()


def resolve_project_path(path_text: str, *, base_dir: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate.resolve()
    primary = (base_dir / candidate).resolve()
    if primary.exists():
        return primary

    fallback = (PROJECT_ROOT / candidate).resolve()
    if fallback.exists():
        return fallback

    return primary


def waveform_fingerprint(waveform: torch.Tensor) -> str:
    pcm = np.clip(waveform.detach().cpu().numpy(), -1.0, 1.0)
    pcm16 = np.round(pcm * 32767.0).astype("<i2")
    return hashlib.sha1(pcm16.tobytes()).hexdigest()


def build_fixed_window(
    waveform: torch.Tensor,
    *,
    start_sample: int,
    end_sample: int,
    window_samples: int,
) -> torch.Tensor:
    window = waveform[start_sample:end_sample]
    if window.numel() < window_samples:
        pad_length = window_samples - window.numel()
        window = torch.nn.functional.pad(window, (0, pad_length))
    elif window.numel() > window_samples:
        window = window[:window_samples]
    return window.to(dtype=torch.float32)


def compute_waveform_stats(waveform: torch.Tensor) -> tuple[float, float, float, float]:
    array = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(array), dtype=np.float64)))
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    zero_ratio = float(np.mean(array == 0.0)) if array.size else 1.0
    clip_ratio = float(np.mean(np.abs(array) >= 0.999)) if array.size else 0.0
    return rms, peak, zero_ratio, clip_ratio


def assign_energy_buckets(candidates: list[WindowCandidate]) -> dict[str, list[WindowCandidate]]:
    if not candidates:
        return {bucket: [] for bucket in ENERGY_BUCKETS}

    rms_values = np.array([candidate.rms for candidate in candidates], dtype=np.float32)
    low_cut = float(np.quantile(rms_values, 1.0 / 3.0))
    high_cut = float(np.quantile(rms_values, 2.0 / 3.0))
    buckets = {bucket: [] for bucket in ENERGY_BUCKETS}
    for candidate in sorted(candidates, key=lambda item: (item.source_file, item.start_sample)):
        if candidate.rms <= low_cut:
            buckets["low"].append(candidate)
        elif candidate.rms >= high_cut:
            buckets["high"].append(candidate)
        else:
            buckets["mid"].append(candidate)
    return buckets


def distribute_targets(total_target: int, available_counts: dict[str, int]) -> dict[str, int]:
    active_buckets = [bucket for bucket in ENERGY_BUCKETS if available_counts.get(bucket, 0) > 0]
    targets = {bucket: 0 for bucket in ENERGY_BUCKETS}
    if total_target <= 0 or not active_buckets:
        return targets

    base = total_target // len(active_buckets)
    remainder = total_target % len(active_buckets)
    for bucket in active_buckets:
        targets[bucket] = min(base, available_counts[bucket])

    for bucket in active_buckets[:remainder]:
        if targets[bucket] < available_counts[bucket]:
            targets[bucket] += 1

    assigned = sum(targets.values())
    if assigned >= total_target:
        return targets

    for bucket in active_buckets:
        while assigned < total_target and targets[bucket] < available_counts[bucket]:
            targets[bucket] += 1
            assigned += 1
    return targets


def select_round_robin(
    pool: list[WindowCandidate],
    *,
    target_count: int,
    selected_by_file: Counter[str],
    selected_starts_by_file: dict[str, list[int]],
    selected_fingerprints: set[str],
    max_windows_per_file: int,
    min_gap_samples: int,
    seed: int,
) -> list[WindowCandidate]:
    grouped: dict[str, list[WindowCandidate]] = defaultdict(list)
    for candidate in pool:
        grouped[candidate.source_file].append(candidate)
    file_order = sorted(grouped.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(file_order)
    for source_file in file_order:
        grouped[source_file].sort(key=lambda item: item.start_sample)

    selected: list[WindowCandidate] = []
    made_progress = True
    while len(selected) < target_count and made_progress:
        made_progress = False
        for source_file in file_order:
            if len(selected) >= target_count:
                break
            if selected_by_file[source_file] >= max_windows_per_file:
                continue

            candidates = grouped[source_file]
            while candidates:
                candidate = candidates.pop(0)
                if candidate.fingerprint in selected_fingerprints:
                    continue
                prior_starts = selected_starts_by_file[source_file]
                if any(abs(candidate.start_sample - start) < min_gap_samples for start in prior_starts):
                    continue

                selected.append(candidate)
                selected_by_file[source_file] += 1
                selected_starts_by_file[source_file].append(candidate.start_sample)
                selected_fingerprints.add(candidate.fingerprint)
                made_progress = True
                break
    return selected


def select_class_windows(
    class_name: str,
    candidates: list[WindowCandidate],
    *,
    target_count: int,
    max_windows_per_file: int,
    min_gap_samples: int,
    seed: int,
) -> tuple[list[tuple[WindowCandidate, str]], dict[str, object]]:
    buckets = assign_energy_buckets(candidates)
    available_counts = {bucket: len(bucket_candidates) for bucket, bucket_candidates in buckets.items()}
    bucket_targets = distribute_targets(target_count, available_counts)

    selected_by_file: Counter[str] = Counter()
    selected_starts_by_file: dict[str, list[int]] = defaultdict(list)
    selected_fingerprints: set[str] = set()
    selections: list[tuple[WindowCandidate, str]] = []

    for bucket_index, bucket_name in enumerate(ENERGY_BUCKETS):
        chosen = select_round_robin(
            buckets[bucket_name],
            target_count=bucket_targets[bucket_name],
            selected_by_file=selected_by_file,
            selected_starts_by_file=selected_starts_by_file,
            selected_fingerprints=selected_fingerprints,
            max_windows_per_file=max_windows_per_file,
            min_gap_samples=min_gap_samples,
            seed=seed + bucket_index,
        )
        selections.extend((candidate, bucket_name) for candidate in chosen)

    if len(selections) < target_count:
        leftovers = []
        for bucket_name in ENERGY_BUCKETS:
            for candidate in buckets[bucket_name]:
                leftovers.append((candidate, bucket_name))
        leftovers.sort(key=lambda item: (item[0].source_file, item[0].start_sample))

        for candidate, bucket_name in leftovers:
            if len(selections) >= target_count:
                break
            if candidate.fingerprint in selected_fingerprints:
                continue
            source_file = candidate.source_file
            if selected_by_file[source_file] >= max_windows_per_file:
                continue
            prior_starts = selected_starts_by_file[source_file]
            if any(abs(candidate.start_sample - start) < min_gap_samples for start in prior_starts):
                continue
            selections.append((candidate, bucket_name))
            selected_by_file[source_file] += 1
            selected_starts_by_file[source_file].append(candidate.start_sample)
            selected_fingerprints.add(candidate.fingerprint)

    class_summary = {
        "class_name": class_name,
        "requested_count": target_count,
        "available_after_filtering": len(candidates),
        "selected_count": len(selections),
        "selected_by_bucket": dict(Counter(bucket_name for _candidate, bucket_name in selections)),
        "selected_by_file": dict(selected_by_file),
        "bucket_targets": bucket_targets,
        "available_by_bucket": available_counts,
    }
    return selections, class_summary


def write_pcm16_wave(path: Path, waveform: torch.Tensor, sample_rate_hz: int) -> None:
    array = np.clip(waveform.detach().cpu().numpy(), -1.0, 1.0)
    pcm16 = np.round(array * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm16.tobytes())


def reset_output_dir(output_dir: Path) -> None:
    windows_dir = output_dir / "windows"
    if windows_dir.exists():
        shutil.rmtree(windows_dir)
    for filename in (
        "manifest.json",
        "summary.json",
        "calibration_input_nchw.npy",
        "calibration_input_nhwc.npy",
    ):
        candidate = output_dir / filename
        if candidate.exists():
            candidate.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def build_candidates(
    *,
    config: dict[str, object],
    data_root: Path,
    hop_sec: float,
    min_rms: float,
    max_clip_ratio: float,
    max_zero_ratio: float,
) -> tuple[dict[str, list[WindowCandidate]], dict[str, dict[str, int]]]:
    audio_cfg = config["audio"]
    classes = list(config["data"]["classes"])
    allowed_extensions = list(config["data"].get("audio_extensions", [".wav"]))
    sample_rate_hz = int(audio_cfg["sample_rate"])
    window_samples = int(round(sample_rate_hz * float(audio_cfg["window_sec"])))
    hop_samples = int(round(sample_rate_hz * hop_sec))
    pad_short = bool(audio_cfg.get("pad_short", True))
    class_files = collect_class_files(data_root, classes, allowed_extensions)

    candidates_by_class = {class_name: [] for class_name in classes}
    rejection_summary = {
        class_name: {
            "accepted": 0,
            "rejected_low_rms": 0,
            "rejected_high_clip_ratio": 0,
            "rejected_high_zero_ratio": 0,
        }
        for class_name in classes
    }

    for class_name in classes:
        for source_path in class_files[class_name]:
            waveform, source_rate_hz = load_wav_mono(source_path)
            waveform = resample_waveform(
                waveform=waveform,
                source_rate_hz=source_rate_hz,
                target_rate_hz=sample_rate_hz,
            )
            starts = enumerate_segment_starts(
                total_samples=int(waveform.numel()),
                window_samples=window_samples,
                hop_samples=hop_samples,
                pad_short=pad_short,
            )
            for start_sample in starts:
                end_sample = start_sample + window_samples
                window = build_fixed_window(
                    waveform,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    window_samples=window_samples,
                )
                rms, peak, zero_ratio, clip_ratio = compute_waveform_stats(window)
                if rms < min_rms:
                    rejection_summary[class_name]["rejected_low_rms"] += 1
                    continue
                if clip_ratio > max_clip_ratio:
                    rejection_summary[class_name]["rejected_high_clip_ratio"] += 1
                    continue
                if zero_ratio > max_zero_ratio:
                    rejection_summary[class_name]["rejected_high_zero_ratio"] += 1
                    continue

                candidates_by_class[class_name].append(
                    WindowCandidate(
                        class_name=class_name,
                        source_file=str(source_path),
                        source_sample_rate_hz=sample_rate_hz,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        window_samples=window_samples,
                        rms=rms,
                        peak=peak,
                        zero_ratio=zero_ratio,
                        clip_ratio=clip_ratio,
                        fingerprint=waveform_fingerprint(window),
                    )
                )
                rejection_summary[class_name]["accepted"] += 1

    return candidates_by_class, rejection_summary


def materialize_selected_windows(
    selections_by_class: dict[str, list[tuple[WindowCandidate, str]]],
    *,
    output_dir: Path,
    config: dict[str, object],
) -> tuple[list[SelectedWindow], np.ndarray, np.ndarray]:
    sample_rate_hz = int(config["audio"]["sample_rate"])
    feature_extractor = LogMelFeatureExtractor(config)
    waveform_cache: dict[str, torch.Tensor] = {}
    selected_windows: list[SelectedWindow] = []
    features_nchw: list[np.ndarray] = []

    for class_name, selections in selections_by_class.items():
        class_dir = output_dir / "windows" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index, (candidate, bucket_name) in enumerate(selections, start=1):
            cache_key = candidate.source_file
            if cache_key not in waveform_cache:
                waveform, source_rate_hz = load_wav_mono(Path(candidate.source_file))
                waveform = resample_waveform(
                    waveform=waveform,
                    source_rate_hz=source_rate_hz,
                    target_rate_hz=sample_rate_hz,
                )
                waveform_cache[cache_key] = waveform

            waveform = waveform_cache[cache_key]
            window = build_fixed_window(
                waveform,
                start_sample=candidate.start_sample,
                end_sample=candidate.end_sample,
                window_samples=candidate.window_samples,
            )
            feature = feature_extractor(window).detach().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(feature).all():
                raise RuntimeError(
                    f"Feature extractor produced non-finite values for {candidate.source_file} "
                    f"window starting at {candidate.start_sec:.3f} seconds"
                )

            filename = (
                f"{index:03d}_"
                f"{bucket_name}_"
                f"{Path(candidate.source_file).stem}_"
                f"{candidate.start_sample:08d}.wav"
            )
            output_path = class_dir / filename
            write_pcm16_wave(output_path, window, sample_rate_hz)

            selected_windows.append(
                SelectedWindow(
                    candidate=candidate,
                    energy_bucket=bucket_name,
                    output_path=output_path,
                )
            )
            features_nchw.append(feature)

    if not features_nchw:
        raise RuntimeError("No calibration windows were selected; nothing to export")

    stacked_nchw = np.stack(features_nchw, axis=0).astype(np.float32, copy=False)
    stacked_nhwc = np.transpose(stacked_nchw, (0, 2, 3, 1)).astype(np.float32, copy=False)
    return selected_windows, stacked_nchw, stacked_nhwc


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config, base_dir=ROOT)
    config = read_yaml(config_path)
    classes = list(config.get("data", {}).get("classes", []))
    if not classes:
        raise RuntimeError(f"Config does not define data.classes: {config_path}")

    data_root_text = args.data_root or str(config.get("paths", {}).get("data_root", ""))
    if not data_root_text:
        raise RuntimeError("No data root configured. Pass --data-root or set paths.data_root in the YAML.")
    data_root = resolve_project_path(data_root_text, base_dir=ROOT)
    output_dir = resolve_project_path(args.output_dir, base_dir=ROOT)
    hop_sec = float(args.hop_sec) if args.hop_sec is not None else float(config["audio"]["hop_sec"])
    if hop_sec <= 0:
        raise RuntimeError("--hop-sec must be greater than 0")

    window_sec = float(config["audio"]["window_sec"])
    min_gap_sec_per_file = (
        float(args.min_gap_sec_per_file)
        if args.min_gap_sec_per_file is not None
        else window_sec
    )
    if min_gap_sec_per_file < 0:
        raise RuntimeError("--min-gap-sec-per-file must not be negative")

    sample_rate_hz = int(config["audio"]["sample_rate"])
    min_gap_samples = int(round(min_gap_sec_per_file * sample_rate_hz))
    target_counts = parse_target_count_text(classes, args.per_class)

    set_random_seed(int(args.seed))
    candidates_by_class, rejection_summary = build_candidates(
        config=config,
        data_root=data_root,
        hop_sec=hop_sec,
        min_rms=float(args.min_rms),
        max_clip_ratio=float(args.max_clip_ratio),
        max_zero_ratio=float(args.max_zero_ratio),
    )

    selections_by_class: dict[str, list[tuple[WindowCandidate, str]]] = {}
    selection_summaries: dict[str, dict[str, object]] = {}
    for class_index, class_name in enumerate(classes):
        selections, summary = select_class_windows(
            class_name,
            candidates_by_class[class_name],
            target_count=target_counts[class_name],
            max_windows_per_file=int(args.max_windows_per_file),
            min_gap_samples=min_gap_samples,
            seed=int(args.seed) + class_index,
        )
        selections_by_class[class_name] = selections
        selection_summaries[class_name] = summary

    reset_output_dir(output_dir)
    selected_windows, stacked_nchw, stacked_nhwc = materialize_selected_windows(
        selections_by_class,
        output_dir=output_dir,
        config=config,
    )

    np.save(output_dir / "calibration_input_nchw.npy", stacked_nchw)
    np.save(output_dir / "calibration_input_nhwc.npy", stacked_nhwc)

    manifest = {
        "config_path": str(config_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "classes": classes,
        "target_counts": target_counts,
        "selected_windows": [
            {
                "class_name": item.candidate.class_name,
                "source_file": item.candidate.source_file,
                "start_sample": item.candidate.start_sample,
                "end_sample": item.candidate.end_sample,
                "start_sec": item.candidate.start_sec,
                "end_sec": item.candidate.end_sec,
                "rms": item.candidate.rms,
                "peak": item.candidate.peak,
                "zero_ratio": item.candidate.zero_ratio,
                "clip_ratio": item.candidate.clip_ratio,
                "energy_bucket": item.energy_bucket,
                "output_path": str(item.output_path),
            }
            for item in selected_windows
        ],
    }
    summary = {
        "config_path": str(config_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "selection_parameters": {
            "hop_sec": hop_sec,
            "window_sec": window_sec,
            "min_gap_sec_per_file": min_gap_sec_per_file,
            "max_windows_per_file": int(args.max_windows_per_file),
            "min_rms": float(args.min_rms),
            "max_clip_ratio": float(args.max_clip_ratio),
            "max_zero_ratio": float(args.max_zero_ratio),
            "seed": int(args.seed),
        },
        "target_counts": target_counts,
        "candidate_counts": {
            class_name: len(candidates_by_class[class_name])
            for class_name in classes
        },
        "rejection_summary": rejection_summary,
        "selection_summary": selection_summaries,
        "exported_arrays": {
            "nchw_path": str(output_dir / "calibration_input_nchw.npy"),
            "nhwc_path": str(output_dir / "calibration_input_nhwc.npy"),
            "nchw_shape": list(stacked_nchw.shape),
            "nhwc_shape": list(stacked_nhwc.shape),
        },
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Calibration data prepared.")
    print(f"Config: {config_path}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"NCHW shape: {tuple(stacked_nchw.shape)}")
    print(f"NHWC shape: {tuple(stacked_nhwc.shape)}")
    for class_name in classes:
        selected_count = selection_summaries[class_name]["selected_count"]
        requested_count = selection_summaries[class_name]["requested_count"]
        print(f"{class_name}: selected {selected_count}/{requested_count}")


if __name__ == "__main__":
    main()
