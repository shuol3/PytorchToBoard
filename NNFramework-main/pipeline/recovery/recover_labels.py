"""从 YAML 和检查点恢复标签信息。"""

from __future__ import annotations

from typing import Any

from ..exceptions import PipelineStageError
from ..intake.inspect_yaml import extract_yaml_labels


# 从 YAML 与检查点元数据中恢复标签列表。
def recover_labels(
    yaml_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> list[str]:
    # 标签优先取 YAML，缺失时再回退到检查点元数据。
    labels = extract_yaml_labels(yaml_config)
    if labels:
        return labels

    metadata_labels = checkpoint_metadata.get("classes")
    if isinstance(metadata_labels, list) and all(isinstance(item, str) for item in metadata_labels):
        return metadata_labels

    raise PipelineStageError(
        stage="recovery",
        reason="Unable to recover labels from yaml or checkpoint metadata",
    )
