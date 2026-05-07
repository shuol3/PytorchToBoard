from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path


FRAMEWORK_TORCH = "torch"
FRAMEWORK_TF = "tensorflow"
FRAMEWORK_KERAS = "keras"
FRAMEWORK_UNKNOWN = "unknown"


class ModelPyInspector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.import_aliases: dict[str, str] = {}
        self.class_defs: list[str] = []
        self.function_defs: list[str] = []
        self.call_names: list[str] = []
        self.assignment_names: list[str] = []
        self.has_main_guard = False
        self.string_literals: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            self.imports.add(alias.name)
            self.import_aliases[alias.asname or root] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module:
            self.imports.add(module)
        root = module.split(".")[0] if module else ""
        for alias in node.names:
            if root:
                self.import_aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_defs.append(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_defs.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_defs.append(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignment_names.append(target.id)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self._is_main_guard(node.test):
            self.has_main_guard = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node.func)
        if name:
            self.call_names.append(name)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.string_literals.append(node.value)
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None

    @staticmethod
    def _is_main_guard(node: ast.AST) -> bool:
        if not isinstance(node, ast.Compare):
            return False
        if not isinstance(node.left, ast.Name) or node.left.id != "__name__":
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            return False
        if len(node.comparators) != 1:
            return False
        comparator = node.comparators[0]
        return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1 of the model pipeline: inspect a Python model file without executing it."
    )
    parser.add_argument("--model-py", type=Path, required=True, help="Path to the incoming Python file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to stdout only.",
    )
    return parser.parse_args()


def read_source(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")


def framework_score(inspector: ModelPyInspector) -> dict[str, int]:
    scores = Counter(
        {
            FRAMEWORK_TORCH: 0,
            FRAMEWORK_TF: 0,
            FRAMEWORK_KERAS: 0,
        }
    )

    for module in inspector.imports:
        lower = module.lower()
        if lower.startswith("torch"):
            scores[FRAMEWORK_TORCH] += 5
        if lower.startswith("tensorflow"):
            scores[FRAMEWORK_TF] += 5
        if lower == "keras" or lower.startswith("keras."):
            scores[FRAMEWORK_KERAS] += 5

    for call_name in inspector.call_names:
        lower = call_name.lower()
        if lower.startswith("torch.") or lower.startswith("nn."):
            scores[FRAMEWORK_TORCH] += 1
        if lower.startswith("tf.") or lower.startswith("tensorflow."):
            scores[FRAMEWORK_TF] += 1
        if lower.startswith("keras.") or ".keras." in lower:
            scores[FRAMEWORK_KERAS] += 1
        if "torch.onnx.export" in lower:
            scores[FRAMEWORK_TORCH] += 3
        if "tfliteconverter" in lower:
            scores[FRAMEWORK_TF] += 3

    for class_name in inspector.class_defs:
        lower = class_name.lower()
        if "model" in lower or "net" in lower:
            scores[FRAMEWORK_TORCH] += 1
            scores[FRAMEWORK_TF] += 1

    for text in inspector.string_literals:
        lower = text.lower()
        if ".tflite" in lower:
            scores[FRAMEWORK_TF] += 1
        if ".onnx" in lower:
            scores[FRAMEWORK_TORCH] += 1

    return dict(scores)


def detect_framework(inspector: ModelPyInspector) -> tuple[str, float, list[str]]:
    scores = framework_score(inspector)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_framework, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0

    reasons: list[str] = []
    if top_score == 0:
        return FRAMEWORK_UNKNOWN, 0.0, ["No strong framework-specific imports or calls were detected."]

    if top_framework == FRAMEWORK_TORCH:
        reasons.append("Detected torch-related imports or API calls.")
    elif top_framework == FRAMEWORK_TF:
        reasons.append("Detected tensorflow-related imports or API calls.")
    elif top_framework == FRAMEWORK_KERAS:
        reasons.append("Detected keras-related imports or API calls.")

    confidence = min(0.99, 0.45 + (top_score * 0.08) + max(0, top_score - second_score) * 0.03)
    return top_framework, round(confidence, 2), reasons


def detect_script_role(inspector: ModelPyInspector) -> str:
    call_set = {name.lower() for name in inspector.call_names}
    func_set = {name.lower() for name in inspector.function_defs}
    text_blob = " ".join(inspector.string_literals).lower()

    if any("fit" in name or "modelcheckpoint" in name for name in call_set) or "train" in func_set:
        return "training_script"
    if any("export" in name or "convert" in name for name in call_set):
        return "export_or_conversion_script"
    if "interpreter" in text_blob or any("interpreter" in name for name in call_set):
        return "inference_or_inspection_script"
    if inspector.class_defs:
        return "model_definition_or_mixed_script"
    return "general_python_script"


def detect_input_style(inspector: ModelPyInspector) -> str:
    text_blob = " ".join(inspector.string_literals + inspector.call_names + inspector.function_defs).lower()
    if "wav" in text_blob or "audio" in text_blob or "waveform" in text_blob:
        return "audio_or_waveform_related"
    if "mel" in text_blob or "stft" in text_blob or "spectrogram" in text_blob:
        return "feature_extraction_related"
    if "image" in text_blob or "conv2d" in text_blob:
        return "image_like_or_tensor_input"
    return "unknown"


def build_manifest(model_py: Path, inspector: ModelPyInspector) -> dict:
    framework, confidence, reasons = detect_framework(inspector)
    role = detect_script_role(inspector)
    input_style = detect_input_style(inspector)

    return {
        "step": "identify_model_py",
        "model_py": str(model_py.resolve()),
        "framework": framework,
        "framework_confidence": confidence,
        "script_role": role,
        "input_style_hint": input_style,
        "has_main_guard": inspector.has_main_guard,
        "class_definitions": inspector.class_defs,
        "function_definitions": inspector.function_defs,
        "top_calls": inspector.call_names[:40],
        "imports": sorted(inspector.imports),
        "assignment_names": inspector.assignment_names[:40],
        "reasons": reasons,
        "next_step_suggestion": suggest_next_step(framework, role),
    }


def suggest_next_step(framework: str, role: str) -> str:
    if framework == FRAMEWORK_TORCH:
        return "Use a Torch adapter to extract the model entry and convert it toward ONNX/TFLite."
    if framework in (FRAMEWORK_TF, FRAMEWORK_KERAS):
        return "Use a TensorFlow/Keras adapter to export a deployable TFLite model."
    if role == "training_script":
        return "Locate the model build function/class and the export section before conversion."
    return "Manual review is recommended before building the conversion pipeline."


def main() -> None:
    args = parse_args()
    model_py = args.model_py.resolve()
    if not model_py.is_file():
        raise FileNotFoundError(f"Python model file not found: {model_py}")

    source = read_source(model_py)
    tree = ast.parse(source, filename=str(model_py))
    inspector = ModelPyInspector()
    inspector.visit(tree)

    manifest = build_manifest(model_py, inspector)
    text = json.dumps(manifest, indent=2, ensure_ascii=False)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote identification report to {args.output}")

    print(text)


if __name__ == "__main__":
    main()
