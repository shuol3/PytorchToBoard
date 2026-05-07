"""定义流水线统一异常类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 承载阶段化错误信息，供 CLI 和 manifest 统一输出。
@dataclass
class PipelineStageError(RuntimeError):
    """Structured error raised by pipeline stages."""

    # stage/reason/details 会直接进入 manifest 和 CLI 输出。
    stage: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = False

    # 初始化 RuntimeError 文本，方便直接抛出和打印。
    def __post_init__(self) -> None:
        super().__init__(f"[{self.stage}] {self.reason}")

    # 转成可直接写入 JSON 的结构化字典。
    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "reason": self.reason,
            "details": self.details,
            "recoverable": self.recoverable,
        }
