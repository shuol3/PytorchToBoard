from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
import venv
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - depends on local environment
    torch = None

try:
    import numpy as np
except Exception:  # pragma: no cover - depends on local environment
    np = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - depends on local environment
    tf = None


# 这些模块类型在导出或部署时更容易遇到限制。
HIGH_RISK_MODULE_TYPES = {
    "LSTM",
    "GRU",
    "RNN",
    "MultiheadAttention",
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "EmbeddingBag",
}

# 自举阶段会安装模型检测和转换所需的运行时依赖。
REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "tensorflow",
    "onnx2tf",
]

# 这些环境变量标记用于避免脚本无限次重进程执行。
REEXEC_ENV_FLAG = "PRECHECK_RUNNING_IN_VENV"
REEXEC_COMPAT_FLAG = "PRECHECK_RUNNING_WITH_COMPAT_PYTHON"
# 当启动器找不到兼容 Python 时，回退到这些固定解释器路径。
PREFERRED_PYTHON_PATHS = [
    pathlib.Path(r"D:\work\python311\python.exe"),
]
# 受 TensorFlow 打包支持范围限制，这里约束可用的 Python 版本。
MIN_SUPPORTED_PYTHON = (3, 9)
MAX_SUPPORTED_PYTHON = (3, 12)


@dataclass
class CheckResult:
    step: str
    ok: bool
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)


# 将命令行中的 shape 字符串如 "1,20" 解析成整数列表，用于构造假输入。
def parse_shape(shape_text: str) -> list[int]:
    values = [part.strip() for part in shape_text.split(",") if part.strip()]
    if not values:
        raise ValueError("input shape must not be empty")
    return [int(value) for value in values]


# 当用户未传模型路径时，自动选取 tools/in 中唯一的 .pt 文件。
def find_default_model_path(script_dir: pathlib.Path) -> pathlib.Path:
    input_dir = script_dir / "in"
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    pt_files = sorted(input_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"no .pt file found in {input_dir}")
    if len(pt_files) > 1:
        names = ", ".join(path.name for path in pt_files)
        raise FileExistsError(f"multiple .pt files found in {input_dir}: {names}")
    return pt_files[0]


# 对仍保留图信息的 TorchScript 模型，尽力推断输入 shape。
def infer_input_shape_from_torchscript(model: torch.nn.Module) -> list[int] | None:
    graph = getattr(model, "graph", None)
    if graph is None:
        return None

    try:
        graph_inputs = list(graph.inputs())
    except Exception:
        return None

    for graph_input in graph_inputs[1:]:
        try:
            tensor_type = graph_input.type()
            sizes = tensor_type.sizes()
        except Exception:
            continue

        if not sizes:
            continue
        if any(size is None or size <= 0 for size in sizes):
            continue
        return [int(size) for size in sizes]
    return None


# 这里只加载可直接运行的 .pt：TorchScript 或完整保存的 nn.Module。
def load_model(model_path: pathlib.Path) -> tuple[torch.nn.Module, str]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed, cannot load a .pt model")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"only .pt files are supported, got: {model_path.name}")

    try:
        scripted_model = torch.jit.load(str(model_path), map_location="cpu")
        return scripted_model.eval(), "torchscript"
    except Exception:
        pass

    loaded_obj = torch.load(model_path, map_location="cpu")

    if isinstance(loaded_obj, torch.nn.Module):
        return loaded_obj.eval(), "torch_module"

    if isinstance(loaded_obj, dict):
        keys = list(loaded_obj.keys())
        preview = ", ".join(str(key) for key in keys[:10]) if keys else "no keys"
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            raise TypeError(
                "loaded object is a checkpoint dict containing 'state_dict'. "
                "This script cannot rebuild the model architecture automatically. "
                f"Top-level keys: {preview}"
            )
        if all(isinstance(key, str) for key in keys):
            raise TypeError(
                "loaded object looks like a raw state_dict, not a runnable model. "
                "This script needs a TorchScript model or a full nn.Module saved with torch.save(model, 'model.pt'). "
                f"Top-level keys: {preview}"
            )
        raise TypeError(
            "loaded object is a dict checkpoint, but its structure is not directly runnable. "
            f"Top-level keys: {preview}"
        )

    raise TypeError(
        f"unsupported object loaded from .pt file: {type(loaded_obj).__name__}. "
        "Save the model as TorchScript or as a full nn.Module with torch.save(model, 'model.pt')."
    )


# 优先使用命令行传入的 shape，否则在可能时从 TorchScript 中推断。
def resolve_input_shape(args: argparse.Namespace, model: torch.nn.Module, load_mode: str) -> tuple[list[int], str]:
    if args.input_shape:
        return parse_shape(args.input_shape), "cli"

    if load_mode == "torchscript":
        inferred = infer_input_shape_from_torchscript(model)
        if inferred is not None:
            return inferred, "torchscript_graph"

    raise ValueError(
        "input shape could not be inferred automatically. "
        "Please provide --input-shape, for example --input-shape 1,20"
    )


# 将嵌套输出整理成可读的扁平摘要，便于终端展示。
def flatten_tensor_shapes(output: Any) -> list[str]:
    if torch is not None and isinstance(output, torch.Tensor):
        return [f"Tensor(shape={list(output.shape)}, dtype={output.dtype})"]
    if isinstance(output, (list, tuple)):
        items: list[str] = []
        for value in output:
            items.extend(flatten_tensor_shapes(value))
        return items
    if isinstance(output, dict):
        items = []
        for key, value in output.items():
            for child in flatten_tensor_shapes(value):
                items.append(f"{key}: {child}")
        return items
    return [f"{type(output).__name__}"]


# 从参数、缓冲区和模块类型中提取一个轻量部署摘要。
def parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_param_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    total_buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    module_types = sorted({module.__class__.__name__ for module in model.modules()})
    risky_modules = [name for name in module_types if name in HIGH_RISK_MODULE_TYPES]

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "parameter_bytes": total_param_bytes,
        "buffer_bytes": total_buffer_bytes,
        "parameter_megabytes": total_param_bytes / (1024 * 1024),
        "buffer_megabytes": total_buffer_bytes / (1024 * 1024),
        "module_types": module_types,
        "high_risk_modules": risky_modules,
    }


# 记录当前目录中的 TFLite 文件状态，用于识别本次运行生成了哪些输出。
def snapshot_tflite_files(folder: pathlib.Path) -> dict[pathlib.Path, tuple[int, int]]:
    snapshot = {}
    for path in folder.glob("*.tflite"):
        stat = path.stat()
        snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


# 优先使用本次新增或更新的 TFLite 文件，必要时回退到目录中的全部候选。
def collect_updated_tflite_files(
    folder: pathlib.Path,
    before_snapshot: dict[pathlib.Path, tuple[int, int]],
) -> list[pathlib.Path]:
    current = snapshot_tflite_files(folder)
    updated = [path for path, state in current.items() if before_snapshot.get(path) != state]
    return updated or sorted(current.keys())


# 对生成的 TFLite 文件做类型判断，以便优先选择 int8 而不是浮点模型。
def classify_tflite_variant(model_path: pathlib.Path) -> str:
    lower_name = model_path.name.lower()
    if "int8" in lower_name:
        return "int8"
    if "float16" in lower_name or "fp16" in lower_name:
        return "float16"
    if "float32" in lower_name or "fp32" in lower_name:
        return "float32"

    if tf is None:
        return "unknown"

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    if np is not None:
        has_quantized_tensor = any(
            detail["dtype"] in (np.int8, np.uint8)
            and detail.get("quantization_parameters", {}).get("scales", np.array([])).size > 0
            for detail in tensor_details
        )
    else:
        has_quantized_tensor = False
    if has_quantized_tensor:
        return "int8"

    dtypes = {detail["dtype"] for detail in tensor_details}
    if np is not None and np.float16 in dtypes:
        return "float16"
    if np is not None and np.float32 in dtypes:
        return "float32"
    return "unknown"


# 对候选 TFLite 输出排序，并保留最适合部署的那个。
def select_best_tflite_model(candidates: list[pathlib.Path]) -> pathlib.Path:
    priority = {"int8": 0, "float32": 1, "float16": 2, "unknown": 3}
    scored = []
    for path in candidates:
        variant = classify_tflite_variant(path)
        scored.append((priority.get(variant, 99), -path.stat().st_mtime_ns, path.name, path))
    scored.sort()
    return scored[0][3]


# 用 TensorFlow Lite 加载生成的 TFLite 文件，并提取输入输出元数据。
def validate_tflite_model(model_path: pathlib.Path) -> dict[str, Any]:
    if tf is None:
        raise RuntimeError("TensorFlow is not installed, cannot validate the generated TFLite model")
    if np is None:
        raise RuntimeError("NumPy is not installed, cannot inspect generated TFLite tensor metadata")

    resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=resolver_type,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return {
        "variant": classify_tflite_variant(model_path),
        "size_bytes": model_path.stat().st_size,
        "inputs": [
            {
                "name": detail["name"],
                "shape": detail["shape"].tolist(),
                "dtype": str(detail["dtype"]),
                "quantization": detail["quantization"],
            }
            for detail in input_details
        ],
        "outputs": [
            {
                "name": detail["name"],
                "shape": detail["shape"].tolist(),
                "dtype": str(detail["dtype"]),
                "quantization": detail["quantization"],
            }
            for detail in output_details
        ],
    }


# 将字节数格式化成易读字符串，便于日志和部署提示展示。
def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{value} B"


# 用统一的 PASS/FAIL 格式打印单条检测结果。
def print_result(result: CheckResult):
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] {result.step}: {result.detail}")
    for key, value in result.extra.items():
        print(f"  - {key}: {value}")


# 自举阶段只依赖部分导入项；这里在重进程后检查真正缺失的运行时依赖。
def missing_runtime_dependencies() -> list[str]:
    missing = []
    if torch is None:
        missing.append("torch")
    if np is None:
        missing.append("numpy")
    if tf is None:
        missing.append("tensorflow")
    return missing


# 只有能安装所需依赖的 Python 版本，才允许自动环境准备流程继续执行。
def python_version_supported(version_info: tuple[int, int]) -> bool:
    return MIN_SUPPORTED_PYTHON <= version_info <= MAX_SUPPORTED_PYTHON


# 将支持的 Python 版本范围格式化为面向用户的错误提示文本。
def format_supported_python_range() -> str:
    min_text = ".".join(str(value) for value in MIN_SUPPORTED_PYTHON)
    max_text = ".".join(str(value) for value in MAX_SUPPORTED_PYTHON)
    return f"Python {min_text} to {max_text}"


# 查找兼容解释器，先尝试 Windows 启动器，再回退到固定路径。
def find_supported_python_launcher() -> list[str] | None:
    if os.name != "nt":
        for path in PREFERRED_PYTHON_PATHS:
            if path.exists():
                return [str(path)]
        return None

    candidate_versions = ["3.11", "3.10"]
    for version in candidate_versions:
        try:
            result = subprocess.run(
                ["py", f"-{version}", "-c", "import sys; print(sys.executable)"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue

        executable = result.stdout.strip()
        if executable:
            return ["py", f"-{version}"]

    for path in PREFERRED_PYTHON_PATHS:
        if path.exists():
            return [str(path)]
    return None


# 解析受管虚拟环境中的 Python 可执行文件路径。
def venv_python_path(venv_dir: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


# 读取已有虚拟环境创建时所使用的 Python 版本。
def read_venv_version(venv_dir: pathlib.Path) -> tuple[int, int] | None:
    config_path = venv_dir / "pyvenv.cfg"
    if not config_path.exists():
        return None

    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if not line.lower().startswith("version"):
                continue
            _, raw_value = line.split("=", 1)
            parts = raw_value.strip().split(".")
            if len(parts) < 2:
                return None
            return int(parts[0]), int(parts[1])
    except Exception:
        return None
    return None


# 创建或重建 tools/.venv，确保它与本次选中的解释器版本一致。
def ensure_virtualenv(venv_dir: pathlib.Path):
    python_path = venv_python_path(venv_dir)
    existing_version = read_venv_version(venv_dir)
    current_version = (sys.version_info.major, sys.version_info.minor)
    if python_path.exists() and existing_version == current_version:
        return python_path
    if venv_dir.exists() and existing_version != current_version:
        print(
            "[INFO] bootstrap: recreating virtual environment because its Python version "
            f"{existing_version} does not match the current interpreter {current_version}"
        )
        shutil.rmtree(venv_dir)
    print(f"[INFO] bootstrap: creating virtual environment at {venv_dir}")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(str(venv_dir))
    return python_path


# 在受管虚拟环境中安装运行时依赖。
def install_requirements(python_path: pathlib.Path):
    print("[INFO] bootstrap: upgrading pip")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    print("[INFO] bootstrap: installing required packages")
    subprocess.run([str(python_path), "-m", "pip", "install", *REQUIRED_PACKAGES], check=True)


# 先切换到兼容解释器，再进入受管虚拟环境中重新执行脚本。
def bootstrap_and_reexec(script_dir: pathlib.Path):
    current_version = (sys.version_info.major, sys.version_info.minor)
    if not python_version_supported(current_version):
        if os.environ.get(REEXEC_COMPAT_FLAG) != "1":
            launcher_prefix = find_supported_python_launcher()
            if launcher_prefix is not None:
                command = [*launcher_prefix, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]]
                env = os.environ.copy()
                env[REEXEC_COMPAT_FLAG] = "1"
                print(
                    "[INFO] bootstrap: current Python is unsupported for TensorFlow; "
                    "rerunning with a compatible Python from the py launcher"
                )
                raise SystemExit(subprocess.call(command, env=env))

        supported_range = format_supported_python_range()
        current_version_text = f"Python {current_version[0]}.{current_version[1]}"
        print(
            f"[FAIL] bootstrap: {current_version_text} is not supported for automatic dependency setup. "
            f"Use {supported_range}, then rerun this script."
        )
        print("[INFO] bootstrap: TensorFlow is the limiting dependency on this machine.")
        raise SystemExit(1)

    venv_dir = script_dir / ".venv"
    python_path = ensure_virtualenv(venv_dir)
    install_requirements(python_path)

    current_python = pathlib.Path(sys.executable).resolve()
    if current_python == python_path.resolve() or os.environ.get(REEXEC_ENV_FLAG) == "1":
        return

    forwarded_args = list(sys.argv[1:])
    command = [str(python_path), str(pathlib.Path(__file__).resolve()), *forwarded_args]
    env = os.environ.copy()
    env[REEXEC_ENV_FLAG] = "1"
    print("[INFO] bootstrap: rerunning precheck inside virtual environment")
    raise SystemExit(subprocess.call(command, env=env))


# 驱动完整的预检流程：环境准备、模型加载、导出、转换和结果汇总。
def main():
    script_dir = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pre-check whether a .pt PyTorch model can be converted to TFLite and deployed."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Optional path to a .pt TorchScript file or full nn.Module checkpoint. "
        "If omitted, the script searches tools/in and auto-uses the only .pt file there.",
    )
    parser.add_argument(
        "--input-shape",
        help="Comma-separated input shape used for forward validation and ONNX export, for example 1,20. "
        "If omitted, the script will try to infer it from a TorchScript model.",
    )
    parser.add_argument(
        "--flash-bytes",
        type=int,
        help="Optional flash budget used to judge whether the generated TFLite model file size fits.",
    )
    parser.add_argument(
        "--ram-bytes",
        type=int,
        help="Optional RAM budget used for a coarse parameter+buffer footprint comparison.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(script_dir / "out" / "precheck"),
        help="Directory used for generated ONNX/TFLite artifacts.",
    )
    args = parser.parse_args()

    # 先确保后续逻辑运行在兼容且依赖完整的环境中。
    bootstrap_and_reexec(script_dir)

    missing_deps = missing_runtime_dependencies()
    if missing_deps:
        print("[FAIL] environment: missing dependencies after bootstrap: " + ", ".join(missing_deps))
        sys.exit(1)

    work_dir = pathlib.Path(args.work_dir).resolve()
    onnx_path = work_dir / "model.onnx"
    tflite_dir = work_dir / "tflite_models"
    work_dir.mkdir(parents=True, exist_ok=True)
    tflite_dir.mkdir(parents=True, exist_ok=True)

    results: list[CheckResult] = []
    # 在进入高开销的运行时流程前，先解析目标模型路径。
    try:
        if args.model_path:
            model_path = pathlib.Path(args.model_path).resolve()
        else:
            model_path = find_default_model_path(script_dir)
        results.append(CheckResult("model_path", True, f"using model file {model_path}"))
    except Exception as exc:
        results.append(CheckResult("model_path", False, str(exc)))
        for result in results:
            print_result(result)
        sys.exit(1)

    try:
        model, load_mode = load_model(model_path)
        results.append(CheckResult("load_model", True, f"loaded model via {load_mode}"))
    except Exception as exc:
        results.append(CheckResult("load_model", False, str(exc)))
        for result in results:
            print_result(result)
        sys.exit(1)

    # 加载模型，并统一切到 cpu/eval 模式，便于稳定执行导出检查。
    model = model.cpu().eval()
    try:
        input_shape, input_shape_source = resolve_input_shape(args, model, load_mode)
        results.append(
            CheckResult(
                "input_shape",
                True,
                f"using input shape from {input_shape_source}",
                {"shape": input_shape},
            )
        )
    except Exception as exc:
        results.append(CheckResult("input_shape", False, str(exc)))
        for result in results:
            print_result(result)
        sys.exit(1)

    # 在转换前先汇总模型信息，确保体积和风险数据始终可见。
    summary = parameter_summary(model)
    results.append(
        CheckResult(
            "model_summary",
            True,
            "collected parameter and module statistics",
            {
                "total_params": summary["total_params"],
                "trainable_params": summary["trainable_params"],
                "parameter_size": format_bytes(summary["parameter_bytes"]),
                "buffer_size": format_bytes(summary["buffer_bytes"]),
                "high_risk_modules": ", ".join(summary["high_risk_modules"]) or "none",
            },
        )
    )

    # 用假输入实际跑一次前向，验证模型是否真的可执行。
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        output_shapes = flatten_tensor_shapes(output)
        results.append(
            CheckResult(
                "forward",
                True,
                "forward pass succeeded",
                {"outputs": " | ".join(output_shapes[:10])},
            )
        )
    except Exception as exc:
        results.append(CheckResult("forward", False, str(exc)))
        for result in results:
            print_result(result)
        sys.exit(1)

    # 先导出 ONNX，因为当前的 TFLite 路径依赖 onnx2tf。
    try:
        if onnx_path.exists():
            onnx_path.unlink()
        torch.onnx.export(model, dummy_input, onnx_path, dynamo=True)
        results.append(CheckResult("onnx_export", True, f"exported ONNX to {onnx_path}"))
    except Exception as exc:
        results.append(
            CheckResult(
                "onnx_export",
                False,
                str(exc),
                {"hint": "model may contain unsupported operators or dynamic behavior for ONNX export"},
            )
        )
        for result in results:
            print_result(result)
        sys.exit(1)

    # onnx2tf 必须在当前环境中可作为命令行或模块调用。
    if shutil.which("onnx2tf") is None:
        results.append(
            CheckResult(
                "onnx2tf",
                False,
                "onnx2tf is not available in PATH or as a console script",
                {"hint": f"try: {sys.executable} -m pip install onnx2tf"},
            )
        )
        for result in results:
            print_result(result)
        sys.exit(1)

    # 将 ONNX 转成 TFLite，并保留最合适的生成结果。
    before_snapshot = snapshot_tflite_files(tflite_dir)
    try:
        subprocess.run(
            [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(tflite_dir)],
            check=True,
        )
        candidates = collect_updated_tflite_files(tflite_dir, before_snapshot)
        if not candidates:
            raise FileNotFoundError("onnx2tf finished but did not create any .tflite files")
        selected_tflite = select_best_tflite_model(candidates)
        results.append(
            CheckResult(
                "onnx2tf",
                True,
                "generated TFLite model",
                {
                    "candidate_count": len(candidates),
                    "selected_model": selected_tflite.name,
                },
            )
        )
    except Exception as exc:
        results.append(
            CheckResult(
                "onnx2tf",
                False,
                str(exc),
                {"hint": "conversion may have failed due to unsupported ONNX operators or TensorFlow mapping issues"},
            )
        )
        for result in results:
            print_result(result)
        sys.exit(1)

    # 校验选中的 TFLite 模型，并输出它的输入输出元数据。
    try:
        tflite_info = validate_tflite_model(selected_tflite)
        extra = {
            "variant": tflite_info["variant"],
            "size": format_bytes(tflite_info["size_bytes"]),
            "inputs": tflite_info["inputs"],
            "outputs": tflite_info["outputs"],
        }
        results.append(CheckResult("tflite_validate", True, "TFLite interpreter loaded the model", extra))
    except Exception as exc:
        results.append(
            CheckResult(
                "tflite_validate",
                False,
                str(exc),
                {"traceback": traceback.format_exc(limit=1).strip()},
            )
        )
        for result in results:
            print_result(result)
        sys.exit(1)

    # 根据文件大小、参数占用和高风险层信息生成一个粗略部署摘要。
    deploy_notes = []
    if args.flash_bytes is not None:
        fits_flash = tflite_info["size_bytes"] <= args.flash_bytes
        deploy_notes.append(
            f"flash budget: {'fits' if fits_flash else 'does not fit'} "
            f"({format_bytes(tflite_info['size_bytes'])} / {format_bytes(args.flash_bytes)})"
        )
    if args.ram_bytes is not None:
        coarse_runtime_bytes = summary["parameter_bytes"] + summary["buffer_bytes"]
        fits_ram = coarse_runtime_bytes <= args.ram_bytes
        deploy_notes.append(
            f"coarse RAM check: {'fits' if fits_ram else 'does not fit'} "
            f"({format_bytes(coarse_runtime_bytes)} / {format_bytes(args.ram_bytes)})"
        )
        deploy_notes.append("note: real TFLite Micro arena usage is usually larger than parameter+buffer bytes")
    if summary["high_risk_modules"]:
        deploy_notes.append(
            "high-risk module types detected: " + ", ".join(summary["high_risk_modules"])
        )
    if not deploy_notes:
        deploy_notes.append("provide --flash-bytes and --ram-bytes for a board-specific deployment judgment")

    results.append(CheckResult("deployment_hint", True, "basic deployment assessment", {"notes": " | ".join(deploy_notes)}))

    for result in results:
        print_result(result)


if __name__ == "__main__":
    main()
