"""构建并校验输入文件集合。"""

from __future__ import annotations

from pathlib import Path

from ..exceptions import PipelineStageError
from ..types import InputBundle


# 构建最小输入集合，并对三项输入做强校验。
def build_input_bundle(
    model_py: str,
    checkpoint: str,
    config: str,
) -> InputBundle:
    # 先统一转为绝对路径，后续各阶段都只处理规范化后的输入。
    bundle = InputBundle(
        model_py=Path(model_py).resolve(),
        checkpoint=Path(checkpoint).resolve(),
        config=Path(config).resolve(),
    )
    errors = validate_input_bundle(bundle)
    if errors:
        raise PipelineStageError(
            stage="identify",
            reason="Invalid required input bundle",
            details={"errors": errors},
        )
    return bundle


# 校验最小输入文件是否存在且路径类型正确。
def validate_input_bundle(bundle: InputBundle) -> list[str]:
    # 这里仅校验三项最小输入是否存在，不做内容级检查。
    errors: list[str] = []
    if not bundle.model_py.is_file():
        errors.append(f"model.py not found: {bundle.model_py}")
    if not bundle.checkpoint.is_file():
        errors.append(f"model.pt not found: {bundle.checkpoint}")
    if not bundle.config.is_file():
        errors.append(f"yaml config not found: {bundle.config}")
    return errors
