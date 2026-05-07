"""定义流水线命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .common.runtime_env import ensure_default_python_for_module

from .exceptions import PipelineStageError
from .intake.detect_inputs import build_input_bundle
from .intake.inspect_pt import inspect_checkpoint
from .paths import build_artifact_paths
from .utils.serde import write_json


# 构建命令行参数解析器。
# 当前暴露 identify、recover、run 三个子命令，
# 并为它们统一注册模型源码、检查点、配置文件和输出目录参数。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch to nRF5340 pipeline entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 三个子命令共享同一组最小输入参数。
    for command in ("identify", "recover", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--model-py", required=True)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--config", required=True)
        sub.add_argument("--output-dir", required=True)

    return parser


# 执行 identify 子命令。
# 该命令只做输入清单构建和检查点静态识别，
# 不会进入模型恢复、导出或验证流程。
def cmd_identify(args: argparse.Namespace) -> int:
    # identify 只产出输入清单和检查点身份信息。
    bundle = build_input_bundle(args.model_py, args.checkpoint, args.config)
    identity = inspect_checkpoint(bundle.checkpoint)
    paths = build_artifact_paths(Path(args.output_dir))
    write_json(paths.input_bundle_json, bundle)
    write_json(paths.model_identity_json, identity)
    print(f"Wrote: {paths.input_bundle_json}")
    print(f"Wrote: {paths.model_identity_json}")
    return 0


# 执行 recover 子命令。
# 该命令会恢复模型和处理契约，并把 recover 阶段的中间产物落盘，
# 但不会继续生成多精度候选或执行后续分析。
def cmd_recover(args: argparse.Namespace) -> int:
    from .orchestrator import run_recover

    # recover 会额外落盘处理契约，但不进入导出阶段。
    bundle = build_input_bundle(args.model_py, args.checkpoint, args.config)
    identity = inspect_checkpoint(bundle.checkpoint)
    restore_result, process_contract = run_recover(bundle, identity)
    paths = build_artifact_paths(Path(args.output_dir))
    from .package.write_process_contract import write_process_contract

    write_json(paths.input_bundle_json, bundle)
    write_json(paths.model_identity_json, identity)
    write_process_contract(paths.process_contract_json, process_contract)
    write_json(paths.manifest_json, {
        "deployment_status": "recover_only",
        "failure_stage": None,
        "warnings": ["This command writes identify and recover artifacts only."],
        "errors": [],
        "input_bundle": bundle,
        "identity": identity,
        "process_contract": process_contract,
        "restore_load_mode": restore_result.load_mode,
    })
    print(f"Wrote: {paths.input_bundle_json}")
    print(f"Wrote: {paths.model_identity_json}")
    print(f"Wrote: {paths.process_contract_json}")
    print(f"Wrote: {paths.manifest_json}")
    return 0


# 执行 run 子命令。
# 这是完整主流程入口，会调用 orchestrator 跑完整流水线，
# 并把顶层 manifest、候选对比、能力分析、验证报告等产物统一写出。
def cmd_run(args: argparse.Namespace) -> int:
    from .orchestrator import run_pipeline
    from .package.write_analysis_reports import (
        write_acceleration_report,
        write_unsupported_report,
        write_validation_report,
    )
    from .package.write_candidate_comparison import write_candidate_comparison
    from .package.write_candidate_manifests import write_candidate_manifests
    from .package.write_manifest import write_pipeline_manifest
    from .package.write_process_contract import write_process_contract

    # run 是完整主流程，会把所有中间与最终产物统一写出。
    bundle = build_input_bundle(args.model_py, args.checkpoint, args.config)
    paths = build_artifact_paths(Path(args.output_dir))
    result = run_pipeline(bundle, paths.root)
    write_json(paths.input_bundle_json, bundle)
    write_json(paths.model_identity_json, result.identity)
    if result.process_contract is not None:
        write_process_contract(paths.process_contract_json, result.process_contract)
    write_acceleration_report(paths.acceleration_report_json, result.candidates)
    write_unsupported_report(paths.unsupported_report_json, result.candidates)
    write_validation_report(paths.validation_report_json, result.validation_report)
    write_candidate_manifests(result.candidates, result.selection, result.deployment_status)
    write_candidate_comparison(paths.candidate_comparison_json, result.candidates, result.selection)
    write_pipeline_manifest(paths.manifest_json, result)
    print(f"Wrote: {paths.input_bundle_json}")
    print(f"Wrote: {paths.model_identity_json}")
    if result.process_contract is not None:
        print(f"Wrote: {paths.process_contract_json}")
    print(f"Wrote: {paths.acceleration_report_json}")
    print(f"Wrote: {paths.unsupported_report_json}")
    print(f"Wrote: {paths.validation_report_json}")
    print(f"Wrote: {paths.candidate_comparison_json}")
    print(f"Wrote: {paths.manifest_json}")
    return 0 if result.deployment_status != "failed" else 1


# CLI 程序主入口。
# 它负责切换到默认 Python 环境、解析参数、分发子命令，
# 并将 PipelineStageError 转换为终端可读的失败输出和退出码。
def main() -> int:
    # 先切到约定环境，再解析命令行参数。
    ensure_default_python_for_module("pipeline.cli")
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "identify":
            return cmd_identify(args)
        if args.command == "recover":
            return cmd_recover(args)
        if args.command == "run":
            return cmd_run(args)
    except PipelineStageError as exc:
        print(f"[{exc.stage.upper()}] FAILED: {exc.reason}")
        if exc.details:
            print(exc.details)
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
