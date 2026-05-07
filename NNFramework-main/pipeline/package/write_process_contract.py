"""写出恢复后的处理契约。"""

from __future__ import annotations

from pathlib import Path

from ..types import ProcessContract
from ..utils.serde import write_json


# 将恢复得到的处理契约单独落盘。
def write_process_contract(path: Path, contract: ProcessContract) -> None:
    # 处理契约单独落盘，便于板端流程复刻时直接消费。
    write_json(path, contract)
