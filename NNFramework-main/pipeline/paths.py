"""集中定义产物输出路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 统一管理顶层产物文件路径，避免各阶段各自拼接。

@dataclass
class ArtifactPaths:
    root: Path
    input_bundle_json: Path
    model_identity_json: Path
    process_contract_json: Path
    manifest_json: Path
    candidate_comparison_json: Path
    acceleration_report_json: Path
    unsupported_report_json: Path
    validation_report_json: Path
    candidates_root: Path


# 基于输出目录派生整套标准产物路径。
def build_artifact_paths(output_dir: Path) -> ArtifactPaths:
    # 所有路径都基于输出目录绝对化后再派生。
    root = output_dir.resolve()
    return ArtifactPaths(
        root=root,
        input_bundle_json=root / "input_bundle.json",
        model_identity_json=root / "model_identity.json",
        process_contract_json=root / "process_contract.json",
        manifest_json=root / "manifest.json",
        candidate_comparison_json=root / "candidate_comparison.json",
        acceleration_report_json=root / "acceleration_report.json",
        unsupported_report_json=root / "unsupported_report.json",
        validation_report_json=root / "validation_report.json",
        candidates_root=root / "candidates",
    )
