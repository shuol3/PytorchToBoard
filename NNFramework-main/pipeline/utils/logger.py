"""提供统一的阶段日志输出。"""

from __future__ import annotations

from time import perf_counter


# 打印阶段开始日志，并返回耗时计时起点。
def log_stage_start(stage: str, detail: str | None = None) -> float:
    # 返回起始时间，供调用方在结束时计算耗时。
    message = f"[{stage}] START"
    if detail:
        message += f": {detail}"
    print(message)
    return perf_counter()


# 打印阶段成功日志，并附带耗时信息。
def log_stage_ok(stage: str, started_at: float, detail: str | None = None) -> None:
    # OK 日志统一带上耗时，便于 CLI 观察长步骤进度。
    elapsed_s = perf_counter() - started_at
    message = f"[{stage}] OK ({elapsed_s:.2f}s)"
    if detail:
        message += f": {detail}"
    print(message)


# 打印阶段失败日志，并尽量带上耗时。
def log_stage_fail(stage: str, reason: str, started_at: float | None = None) -> None:
    # 失败日志允许没有 started_at，用于异常兜底场景。
    message = f"[{stage}] FAILED"
    if started_at is not None:
        elapsed_s = perf_counter() - started_at
        message += f" ({elapsed_s:.2f}s)"
    message += f": {reason}"
    print(message)
