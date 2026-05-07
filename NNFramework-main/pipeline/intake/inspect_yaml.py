"""读取并解析训练配置 YAML。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import PipelineStageError

try:
    import yaml
except ImportError:  # pragma: no cover - import guard
    yaml = None


# 读取训练侧 YAML 配置，供恢复阶段统一消费。
def load_yaml_config(path: Path) -> dict[str, Any]:
    # YAML 是后续恢复流程的主来源，缺依赖时直接中断。
    if yaml is None:
        raise PipelineStageError(
            stage="identify",
            reason=(
                "PyYAML is required to load the training config. "
                "Install it into D:\\work\\AnacondaEnvironment."
            ),
        )
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise PipelineStageError(
            stage="identify",
            reason="YAML config must deserialize to a mapping",
            details={"config_path": str(path)},
        )
    return config


# 抽取恢复流程关心的主配置段。
def extract_yaml_process_hints(config: dict[str, Any]) -> dict[str, Any]:
    # 只抽取当前流程真正关心的几个主配置段。
    return {
        "audio": config.get("audio", {}),
        "feature": config.get("feature", {}),
        "model": config.get("model", {}),
        "data": config.get("data", {}),
    }


# 从 YAML 中恢复分类标签列表。
def extract_yaml_labels(config: dict[str, Any]) -> list[str]:
    # 标签目前约定从 data.classes 读取。
    classes = config.get("data", {}).get("classes")
    if isinstance(classes, list) and all(isinstance(item, str) for item in classes):
        return classes
    return []
