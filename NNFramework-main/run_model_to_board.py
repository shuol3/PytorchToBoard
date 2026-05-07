"""零参数入口：固定读取 in 目录中的模型、配置与数据，并生成板端代码。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# 整个部署入口固定读取 NNFramework-main/in，不再要求命令行参数。
CORE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_ROOT.parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pipeline.common.runtime_env import ensure_default_python_for_script

ensure_default_python_for_script(__file__)

from pipeline.common.frontend_shape_utils import infer_feature_frame_count
from pipeline.common.input_data import read_yaml, validate_input_config

INPUT_DIR = CORE_ROOT / "in"
ARTIFACT_ROOT = CORE_ROOT / "artifacts" / "latest_run"
PIPELINE_OUTPUT_DIR = ARTIFACT_ROOT / "pipeline"
GENERATED_DIR = PROJECT_ROOT / "pencilv_stage_tflite_end2end" / "src" / "generated"
BOARD_GENERATOR_SCRIPT = CORE_ROOT / "pipeline" / "board" / "tflite2cppfile.py"
PIPELINE_LOG_PATH = ARTIFACT_ROOT / "pipeline_run.log"
BOARD_LOG_PATH = ARTIFACT_ROOT / "board_export.log"
SUMMARY_PATH = ARTIFACT_ROOT / "entrypoint_summary.json"


def resolve_checkpoint_path() -> Path:
    """解析固定输入目录下唯一有效的 checkpoint 文件。"""

    preferred = [INPUT_DIR / "best_model.pt", INPUT_DIR / "model.pt"]
    named = [path for path in preferred if path.is_file()]
    other = sorted(
        path for path in INPUT_DIR.glob("*.pt")
        if path.name not in {candidate.name for candidate in preferred}
    )

    if len(named) > 1:
        raise RuntimeError(
            f"Found multiple preferred checkpoints in {INPUT_DIR}. Keep only one of best_model.pt / model.pt."
        )
    if named and other:
        all_paths = [*(str(path) for path in named), *(str(path) for path in other)]
        raise RuntimeError(
            "Found multiple checkpoint files in the input directory. Keep only one .pt file:\n"
            + "\n".join(all_paths)
        )
    if named:
        return named[0]
    if len(other) == 1:
        return other[0]
    if len(other) > 1:
        raise RuntimeError(
            "Found multiple checkpoint files in the input directory. Keep only one .pt file:\n"
            + "\n".join(str(path) for path in other)
        )
    raise FileNotFoundError(f"No checkpoint file was found in {INPUT_DIR}")


def ensure_input_layout() -> tuple[Path, Path, Path, Path]:
    """校验固定输入目录中的四项输入是否齐全。"""

    model_py_path = INPUT_DIR / "model.py"
    config_path = INPUT_DIR / "train_1s.yaml"
    checkpoint_path = resolve_checkpoint_path()
    data_dir = INPUT_DIR / "data"

    missing = [str(path) for path in (model_py_path, config_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input files are missing:\n" + "\n".join(missing))
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input data directory not found: {data_dir}")
    return model_py_path, checkpoint_path, config_path, data_dir


def prepare_artifact_root() -> None:
    """重建 latest_run 目录，保证本次输出干净可追踪。"""

    if ARTIFACT_ROOT.exists():
        shutil.rmtree(ARTIFACT_ROOT)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def build_resolved_config(config_path: Path, data_dir: Path) -> tuple[dict[str, Any], Path]:
    """把源 YAML 规范化为本次部署可直接消费的 resolved config。"""

    config = read_yaml(config_path)
    paths_cfg = config.setdefault("paths", {})
    paths_cfg["data_root"] = str(data_dir.resolve())
    paths_cfg["output_dir"] = str(ARTIFACT_ROOT.resolve())
    validate_input_config(config)

    resolved_config_path = ARTIFACT_ROOT / "resolved_config.yaml"
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyYAML is required to write the resolved config") from exc
    resolved_config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config, resolved_config_path


def run_subprocess(command: list[str], cwd: Path, log_path: Path) -> None:
    """运行子进程，并把 stdout/stderr 全量落盘。"""

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed: {command}\n"
            f"See log: {log_path}\n"
            f"stderr:\n{completed.stderr}"
        )


def run_pipeline(model_py_path: Path, checkpoint_path: Path, resolved_config_path: Path) -> dict[str, Any]:
    """调用 pipeline.cli，完成模型恢复、导出、校准与验证。"""

    PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pipeline.cli",
        "run",
        "--model-py",
        str(model_py_path),
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(resolved_config_path),
        "--output-dir",
        str(PIPELINE_OUTPUT_DIR),
    ]
    run_subprocess(command, CORE_ROOT, PIPELINE_LOG_PATH)

    manifest_path = PIPELINE_OUTPUT_DIR / "manifest.json"
    comparison_path = PIPELINE_OUTPUT_DIR / "candidate_comparison.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    return {
        "manifest": manifest,
        "candidate_comparison": comparison,
        "manifest_path": str(manifest_path),
        "candidate_comparison_path": str(comparison_path),
    }


def infer_feature_frame_count_from_config(config: dict[str, Any]) -> int:
    """按当前输入配置推导固定特征帧数。"""

    audio_cfg = config["audio"]
    feature_cfg = config["feature"]
    return infer_feature_frame_count(
        sample_rate_hz=int(audio_cfg["sample_rate"]),
        window_sec=float(audio_cfg["window_sec"]),
        hop_length=int(feature_cfg["hop_length"]),
        center=bool(feature_cfg.get("center", False)),
        n_fft=int(feature_cfg["n_fft"]),
        win_length=int(feature_cfg["win_length"]),
    )


def resolve_selected_model_path(candidate_comparison: dict[str, Any]) -> tuple[str, Path]:
    """从候选对比清单中找出最终选中的 TFLite 文件。"""

    selected_precision = candidate_comparison.get("selected_precision")
    if not selected_precision:
        raise RuntimeError("Pipeline did not select any precision candidate")
    for candidate in candidate_comparison.get("candidates", []):
        if candidate.get("precision") == selected_precision:
            model_path = candidate.get("model_path")
            if not model_path:
                raise RuntimeError("Selected candidate does not include model_path")
            return selected_precision, Path(str(model_path)).resolve()
    raise RuntimeError(f"Selected precision not found in candidate list: {selected_precision}")


def generate_board_artifacts(config: dict[str, Any], candidate_comparison: dict[str, Any]) -> dict[str, Any]:
    """根据 pipeline 选择结果生成板端 C/C++ 文件。"""

    selected_precision, model_path = resolve_selected_model_path(candidate_comparison)
    audio_cfg = config["audio"]
    feature_cfg = config["feature"]
    classes = list(config["data"]["classes"])
    sample_rate_hz = int(audio_cfg["sample_rate"])
    capture_window_ms = int(round(float(audio_cfg["window_sec"]) * 1000.0))
    frame_length_ms = int(round(int(feature_cfg["win_length"]) * 1000.0 / sample_rate_hz))
    frame_stride_ms = int(round(int(feature_cfg["hop_length"]) * 1000.0 / sample_rate_hz))
    feature_frame_count = infer_feature_frame_count_from_config(config)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(BOARD_GENERATOR_SCRIPT),
        str(model_path),
        "--output-dir",
        str(GENERATED_DIR),
        "--process-contract",
        str(PIPELINE_OUTPUT_DIR / "process_contract.json"),
        "--labels",
        ",".join(classes),
        "--sample-rate-hz",
        str(sample_rate_hz),
        "--capture-window-ms",
        str(capture_window_ms),
        "--frame-length-ms",
        str(frame_length_ms),
        "--frame-stride-ms",
        str(frame_stride_ms),
        "--fft-length",
        str(int(feature_cfg["n_fft"])),
        "--mel-bin-count",
        str(int(feature_cfg["n_mels"])),
        "--feature-frame-count",
        str(feature_frame_count),
        "--mel-lower-edge-hz",
        str(float(feature_cfg["fmin"])),
        "--mel-upper-edge-hz",
        str(float(feature_cfg["fmax"])),
        "--top-db",
        str(float(feature_cfg.get("top_db", 80.0))),
    ]
    run_subprocess(command, CORE_ROOT, BOARD_LOG_PATH)
    return {
        "selected_precision": selected_precision,
        "selected_model_path": str(model_path),
        "generated_dir": str(GENERATED_DIR),
    }


def write_summary(
    *,
    model_py_path: Path,
    checkpoint_path: Path,
    source_config_path: Path,
    resolved_config_path: Path,
    pipeline_summary: dict[str, Any],
    board_summary: dict[str, Any],
) -> None:
    """写出零参数入口级别的统一摘要。"""

    payload = {
        "input_dir": str(INPUT_DIR),
        "inputs": {
            "model_py": str(model_py_path),
            "checkpoint": str(checkpoint_path),
            "source_config": str(source_config_path),
            "resolved_config": str(resolved_config_path),
            "data_root": str((INPUT_DIR / "data").resolve()),
        },
        "pipeline": pipeline_summary,
        "board_export": board_summary,
        "logs": {
            "pipeline": str(PIPELINE_LOG_PATH),
            "board_export": str(BOARD_LOG_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    """固定读取 in 目录并完成整条模型到板端的转换链。"""

    prepare_artifact_root()
    model_py_path, checkpoint_path, config_path, data_dir = ensure_input_layout()
    config, resolved_config_path = build_resolved_config(config_path, data_dir)
    pipeline_summary = run_pipeline(model_py_path, checkpoint_path, resolved_config_path)
    board_summary = generate_board_artifacts(config, pipeline_summary["candidate_comparison"])
    write_summary(
        model_py_path=model_py_path,
        checkpoint_path=checkpoint_path,
        source_config_path=config_path,
        resolved_config_path=resolved_config_path,
        pipeline_summary=pipeline_summary,
        board_summary=board_summary,
    )

    print(f"Pipeline artifacts: {PIPELINE_OUTPUT_DIR}")
    print(f"Generated board files: {GENERATED_DIR}")
    print(f"Summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
