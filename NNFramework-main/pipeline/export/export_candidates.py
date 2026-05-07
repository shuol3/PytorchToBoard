"""Export float and int8 TFLite candidate models."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from ..common.calibration_targets import build_default_target_counts, resolve_calibration_sample_count
from ..common.onnx_export_config import DEFAULT_ONNX_EXPORT_DYNAMO
from ..common.representative_calibration import generate_representative_calibration_npy, load_training_config

from ..exceptions import PipelineStageError
from ..graph.inspect_tflite import inspect_tflite_model
from ..types import CandidateArtifact, ModelRestoreResult, Precision, ProcessContract, SelectionResult
from ..utils.logger import log_stage_ok, log_stage_start
from ..utils.serde import write_json

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import guard
    torch = None
    nn = None


PRECISION_ORDER: tuple[Precision, ...] = ("int8", "float16", "float32")
DEFAULT_INPUT_NAME = "input"
DEFAULT_OUTPUT_NAME = "output"
DEFAULT_INT8_CALIBRATION_SAMPLES: int | None = None
DEFAULT_INT8_CALIBRATION_SEED = 42


def export_all_candidates(
    restore_result: ModelRestoreResult,
    process_contract: ProcessContract,
    output_dir: Path,
    config_path: Path | None = None,
) -> list[CandidateArtifact]:
    model = _require_torch_model(restore_result)
    input_shape = _extract_input_shape(process_contract)
    candidates_root = output_dir / "candidates"

    float_candidates = _export_float_candidates(
        model=model,
        input_shape=input_shape,
        candidates_root=candidates_root,
    )
    int8_candidate = _export_int8_candidate(
        model=model,
        input_shape=input_shape,
        candidate_dir=candidates_root / "int8",
        config_path=config_path,
    )

    candidate_map = {
        "float32": float_candidates["float32"],
        "float16": float_candidates["float16"],
        "int8": int8_candidate,
    }
    return [candidate_map[precision] for precision in PRECISION_ORDER]


def select_best_export_candidate(candidates: list[CandidateArtifact]) -> SelectionResult:
    candidates_by_precision = {candidate.precision: candidate for candidate in candidates}
    rejected: dict[str, str] = {}

    for precision in PRECISION_ORDER:
        candidate = candidates_by_precision.get(precision)
        if candidate is None:
            rejected[precision] = "candidate was not produced"
            continue
        if candidate.export_ok:
            return SelectionResult(
                selected_precision=precision,
                selected_candidate=candidate,
                rejected=rejected,
                selection_reason="Highest-priority candidate that exported successfully.",
            )
        rejected[precision] = candidate.export_error or "export failed"

    return SelectionResult(
        selected_precision=None,
        selected_candidate=None,
        rejected=rejected,
        selection_reason="No precision candidate exported successfully.",
    )


def _export_float_candidates(
    model: nn.Module,
    input_shape: list[int],
    candidates_root: Path,
) -> dict[str, CandidateArtifact]:
    stage_started = log_stage_start("EXPORT/FLOAT", "float32 + float16")
    shared_dir = candidates_root / "_shared_float_export"
    shared_onnx_path = shared_dir / "torch_model.onnx"
    shared_output_dir = shared_dir / "onnx2tf_output"
    candidate_dirs = {
        "float32": candidates_root / "float32",
        "float16": candidates_root / "float16",
    }

    try:
        export_onnx_model(
            model=model,
            input_shape=input_shape,
            onnx_path=shared_onnx_path,
            input_name=DEFAULT_INPUT_NAME,
            output_name=DEFAULT_OUTPUT_NAME,
        )
        run_onnx2tf_float_exports(shared_onnx_path, shared_output_dir)

        source_paths = {
            "float32": find_single_file(shared_output_dir, "*_float32.tflite"),
            "float16": find_single_file(shared_output_dir, "*_float16.tflite"),
        }
        results: dict[str, CandidateArtifact] = {}
        for precision, source_path in source_paths.items():
            candidate_dir = candidate_dirs[precision]
            candidate_dir.mkdir(parents=True, exist_ok=True)
            exported_model_path = candidate_dir / source_path.name
            shutil.copy2(source_path, exported_model_path)

            graph_info = inspect_tflite_model(exported_model_path)
            warnings: list[str] = []
            if precision == "float16":
                warnings.append("Float16 candidate exported for the TFLite builtin runtime path.")
            export_metadata = {
                "step": "export_float_candidate",
                "input_shape_nchw": input_shape,
                "onnx_path": shared_onnx_path,
                "source_tflite_path": source_path,
            }
            candidate = CandidateArtifact(
                precision=precision,
                export_ok=True,
                model_path=exported_model_path,
                manifest_path=candidate_dir / "manifest.json",
                export_metadata=export_metadata,
                graph_info=graph_info,
                warnings=warnings,
            )
            write_json(
                candidate.manifest_path,
                build_candidate_manifest_payload(candidate, export_metadata),
            )
            results[precision] = candidate

        log_stage_ok("EXPORT/FLOAT", stage_started, "float32 + float16 exported")
        return results
    except Exception as exc:
        error_text = format_exception_message(exc)
        results = {}
        for precision, candidate_dir in candidate_dirs.items():
            candidate_dir.mkdir(parents=True, exist_ok=True)
            export_metadata = {
                "step": "export_float_candidate",
                "input_shape_nchw": input_shape,
                "onnx_path": shared_onnx_path,
            }
            candidate = CandidateArtifact(
                precision=precision,
                export_ok=False,
                export_error=error_text,
                manifest_path=candidate_dir / "manifest.json",
                export_metadata=export_metadata,
                warnings=["Shared float export step failed before this candidate could be materialized."],
            )
            write_json(
                candidate.manifest_path,
                build_candidate_manifest_payload(candidate, export_metadata),
            )
            results[precision] = candidate
        return results


def _export_int8_candidate(
    model: nn.Module,
    input_shape: list[int],
    candidate_dir: Path,
    config_path: Path | None,
) -> CandidateArtifact:
    stage_started = log_stage_start("EXPORT/INT8", "representative_or_pseudo calibration")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = candidate_dir / "torch_model.onnx"
    manifest_path = candidate_dir / "manifest.json"
    output_dir = candidate_dir / "onnx2tf_output"
    export_metadata: dict[str, object] = {
        "step": "export_int8_candidate",
        "input_shape_nchw": input_shape,
        "onnx_path": onnx_path,
    }
    warnings: list[str] = []

    try:
        export_onnx_model(
            model=model,
            input_shape=input_shape,
            onnx_path=onnx_path,
            input_name=DEFAULT_INPUT_NAME,
            output_name=DEFAULT_OUTPUT_NAME,
        )
        calibration_path, calibration_mean, calibration_std, calibration_metadata, calibration_warnings = (
            _prepare_int8_calibration(
                input_shape=input_shape,
                candidate_dir=candidate_dir,
                config_path=config_path,
            )
        )
        export_metadata.update(calibration_metadata)
        warnings.extend(calibration_warnings)

        selected_tflite_path = run_onnx2tf_full_int8(
            onnx_path=onnx_path,
            output_dir=output_dir,
            input_name=DEFAULT_INPUT_NAME,
            calibration_npy=calibration_path,
            calibration_mean=calibration_mean,
            calibration_std=calibration_std,
        )
        exported_model_path = candidate_dir / selected_tflite_path.name
        shutil.copy2(selected_tflite_path, exported_model_path)

        graph_info = inspect_tflite_model(exported_model_path)
        candidate = CandidateArtifact(
            precision="int8",
            export_ok=True,
            model_path=exported_model_path,
            manifest_path=manifest_path,
            export_metadata=export_metadata,
            graph_info=graph_info,
            warnings=warnings,
        )
        write_json(
            manifest_path,
            build_candidate_manifest_payload(candidate, export_metadata),
        )
        log_stage_ok("EXPORT/INT8", stage_started, "int8 exported")
        return candidate
    except Exception as exc:
        candidate = CandidateArtifact(
            precision="int8",
            export_ok=False,
            export_error=format_exception_message(exc),
            manifest_path=manifest_path,
            export_metadata=export_metadata,
            warnings=warnings or ["INT8 export failed."],
        )
        write_json(
            manifest_path,
            build_candidate_manifest_payload(candidate, export_metadata),
        )
        return candidate


def export_onnx_model(
    model: nn.Module,
    input_shape: list[int],
    onnx_path: Path,
    input_name: str,
    output_name: str,
) -> None:
    if torch is None:
        raise PipelineStageError(
            stage="export",
            reason="PyTorch is required to export ONNX candidates",
        )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    model = model.eval().cpu()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=[input_name],
        output_names=[output_name],
        dynamo=DEFAULT_ONNX_EXPORT_DYNAMO,
    )


def run_onnx2tf_float_exports(onnx_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            str(onnx_path),
            "-o",
            str(output_dir),
            "-v",
            "error",
        ]
    )


def run_onnx2tf_full_int8(
    onnx_path: Path,
    output_dir: Path,
    input_name: str,
    calibration_npy: Path,
    calibration_mean: str,
    calibration_std: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            str(onnx_path),
            "-o",
            str(output_dir),
            "-oiqt",
            "-iqd",
            "int8",
            "-oqd",
            "int8",
            "-qt",
            "per-channel",
            "-cind",
            input_name,
            str(calibration_npy),
            calibration_mean,
            calibration_std,
            "-v",
            "error",
        ]
    )

    full_int8_candidates: list[Path] = []
    for candidate_path in sorted(output_dir.glob("*.tflite")):
        try:
            graph_info = inspect_tflite_model(candidate_path)
        except Exception:
            continue
        if is_full_int8_graph(graph_info):
            full_int8_candidates.append(candidate_path)

    if not full_int8_candidates:
        candidate_names = ", ".join(path.name for path in sorted(output_dir.glob("*.tflite")))
        raise RuntimeError(
            "onnx2tf completed but did not produce a full-int8 TFLite artifact. "
            f"Available files: {candidate_names or 'none'}."
        )

    full_int8_candidates.sort(key=lambda path: (-path.stat().st_mtime_ns, path.name))
    return full_int8_candidates[0]


def generate_pseudo_calibration_npy(
    input_shape: list[int],
    output_path: Path,
    calibration_samples: int,
    seed: int,
) -> tuple[Path, str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf_shape = build_tf_calibration_shape(input_shape, calibration_samples)
    rng = np.random.default_rng(seed)
    calibration_data = rng.random(tf_shape, dtype=np.float32)
    np.save(output_path, calibration_data.astype(np.float32, copy=False))
    mean_text, std_text = build_quant_mean_std(len(tf_shape))
    return output_path, mean_text, std_text


def _resolve_default_calibration_request(config_path: Path | None) -> tuple[int, dict[str, int] | None]:
    if config_path is None or not config_path.is_file():
        return 1, None

    try:
        config = load_training_config(config_path)
        class_names = list(config.get("data", {}).get("classes", []))
    except Exception:
        return 1, None

    if not class_names:
        return 1, None

    target_counts = build_default_target_counts(class_names)
    return resolve_calibration_sample_count(class_names, None), target_counts


def _prepare_int8_calibration(
    *,
    input_shape: list[int],
    candidate_dir: Path,
    config_path: Path | None,
) -> tuple[Path, str, str, dict[str, object], list[str]]:
    representative_path = candidate_dir / "representative_calibration_input.npy"
    pseudo_path = candidate_dir / "pseudo_calibration_input.npy"
    default_requested_samples, default_target_counts = _resolve_default_calibration_request(config_path)

    if config_path is not None and config_path.is_file():
        try:
            calibration_path, calibration_mean, calibration_std, representative_metadata = (
                generate_representative_calibration_npy(
                    config_path=config_path,
                    input_shape_nchw=input_shape,
                    output_path=representative_path,
                    calibration_samples=DEFAULT_INT8_CALIBRATION_SAMPLES,
                    seed=DEFAULT_INT8_CALIBRATION_SEED,
                )
            )
            export_metadata = {
                "calibration_path": calibration_path,
                "representative_calibration": True,
                "pseudo_calibration": False,
                "calibration_source": "representative",
                "calibration_samples": representative_metadata.get("actual_samples"),
                "calibration_requested_samples": representative_metadata.get("requested_samples"),
                "calibration_selection_mode": representative_metadata.get("selection_mode"),
                "calibration_manifest_path": representative_metadata.get("metadata_path"),
                "calibration_split": representative_metadata.get("split"),
                "calibration_class_counts": representative_metadata.get("class_counts"),
                "calibration_target_counts": representative_metadata.get("target_counts"),
                "calibration_config_path": representative_metadata.get("config_path"),
                "calibration_data_root": representative_metadata.get("data_root"),
                "int8_confidence": "higher",
            }
            warnings = [
                "Using representative calibration data generated from real audio features.",
            ]
            return calibration_path, calibration_mean, calibration_std, export_metadata, warnings
        except Exception as exc:
            fallback_reason = (
                "Representative calibration generation failed; fell back to pseudo calibration. "
                f"Reason: {format_exception_message(exc)}"
            )
    elif config_path is not None:
        fallback_reason = (
            "Representative calibration was requested but the config file does not exist; "
            "fell back to pseudo calibration."
        )
    else:
        fallback_reason = (
            "No config file was provided for representative calibration; "
            "fell back to pseudo calibration."
        )

    calibration_path, calibration_mean, calibration_std = generate_pseudo_calibration_npy(
        input_shape=input_shape,
        output_path=pseudo_path,
        calibration_samples=default_requested_samples,
        seed=DEFAULT_INT8_CALIBRATION_SEED,
    )
    export_metadata = {
        "calibration_path": calibration_path,
        "representative_calibration": False,
        "pseudo_calibration": True,
        "calibration_source": "pseudo",
        "calibration_samples": default_requested_samples,
        "calibration_requested_samples": default_requested_samples,
        "calibration_selection_mode": "pseudo_fallback",
        "calibration_target_counts": default_target_counts,
        "calibration_fallback_reason": fallback_reason,
        "int8_confidence": "low",
    }
    warnings = [
        fallback_reason,
        "Pseudo calibration remains a provisional compatibility path.",
    ]
    return calibration_path, calibration_mean, calibration_std, export_metadata, warnings


def build_tf_calibration_shape(input_shape: list[int], calibration_samples: int) -> list[int]:
    if len(input_shape) == 4:
        _batch, channels, height, width = input_shape
        return [calibration_samples, height, width, channels]
    if len(input_shape) == 3:
        _batch, channels, width = input_shape
        return [calibration_samples, width, channels]
    if len(input_shape) == 2:
        _batch, width = input_shape
        return [calibration_samples, width]
    raise ValueError(f"Unsupported input rank for pseudo calibration: {len(input_shape)}")


def build_quant_mean_std(rank: int) -> tuple[str, str]:
    if rank == 4:
        mean = [[[[0.5]]]]
        std = [[[[0.25]]]]
    elif rank == 3:
        mean = [[[0.5]]]
        std = [[[0.25]]]
    elif rank == 2:
        mean = [[0.5]]
        std = [[0.25]]
    else:
        raise ValueError(f"Unsupported rank for pseudo calibration mean/std: {rank}")
    return _json_array(mean), _json_array(std)


def build_candidate_manifest_payload(
    candidate: CandidateArtifact,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "precision": candidate.precision,
        "export_ok": candidate.export_ok,
        "export_error": candidate.export_error,
        "model_path": candidate.model_path,
        "export_metadata": candidate.export_metadata,
        "warnings": candidate.warnings,
        "graph_info": candidate.graph_info,
    }
    if extra:
        payload.update(extra)
    return payload


def find_single_file(root: Path, pattern: str) -> Path:
    matches = sorted(
        root.glob(pattern),
        key=lambda path: (-path.stat().st_mtime_ns, path.name),
    )
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} was produced in {root}")
    return matches[0]


def run_command(command: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if completed.returncode == 0:
        return

    command_text = " ".join(command)
    output_text = trim_output("\n".join(part for part in (completed.stdout, completed.stderr) if part))
    raise RuntimeError(
        f"Command failed with exit code {completed.returncode}: {command_text}\n{output_text}"
    )


def trim_output(text: str, max_lines: int = 40) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def is_full_int8_graph(graph_info) -> bool:
    if not graph_info.input_tensors or not graph_info.output_tensors:
        return False
    input_tensor = graph_info.input_tensors[0]
    output_tensor = graph_info.output_tensors[0]
    return (
        input_tensor.get("dtype") == "int8"
        and output_tensor.get("dtype") == "int8"
        and float(input_tensor.get("quantization", {}).get("scale", 0.0)) > 0.0
        and float(output_tensor.get("quantization", {}).get("scale", 0.0)) > 0.0
    )


def format_exception_message(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _extract_input_shape(process_contract: ProcessContract) -> list[int]:
    shape = process_contract.model_input.get("shape_nchw")
    if not isinstance(shape, list) or not shape:
        raise PipelineStageError(
            stage="export",
            reason="Process contract does not provide model_input.shape_nchw",
        )
    try:
        normalized = [int(value) for value in shape]
    except Exception as exc:
        raise PipelineStageError(
            stage="export",
            reason="model_input.shape_nchw is not a valid integer list",
            details={"shape_nchw": shape},
        ) from exc
    if any(value <= 0 for value in normalized):
        raise PipelineStageError(
            stage="export",
            reason="model_input.shape_nchw must contain only positive dimensions",
            details={"shape_nchw": normalized},
        )
    return normalized


def _json_array(value: object) -> str:
    return np.array(value, dtype=np.float32).tolist().__repr__().replace("'", "")


def _require_torch_model(restore_result: ModelRestoreResult) -> nn.Module:
    if torch is None or nn is None:
        raise PipelineStageError(
            stage="export",
            reason="PyTorch is required to export model candidates",
        )
    if not isinstance(restore_result.model, nn.Module):
        raise PipelineStageError(
            stage="export",
            reason="Restored model is not a torch.nn.Module instance",
            details={"model_type": type(restore_result.model).__name__},
        )
    return restore_result.model.eval().cpu()
