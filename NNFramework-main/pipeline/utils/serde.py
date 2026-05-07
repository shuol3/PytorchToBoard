"""提供 JSON 序列化与写盘工具。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


# 递归转换对象，保证最终可以直接写成 JSON。
def to_jsonable(obj: Any) -> Any:
    # 递归展开 dataclass 和 Path，保证最终可直接序列化为 JSON。
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    return obj


# 以统一编码和缩进格式写出 JSON 文件。
def write_json(path: Path, obj: Any) -> None:
    # 统一使用 UTF-8 和缩进格式，便于人工检查。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
