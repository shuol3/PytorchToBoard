from __future__ import annotations

import argparse
import json
import pickletools
import zipfile
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "in"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1 of the pt-first pipeline: inspect a .pt file and report what is still needed."
    )
    parser.add_argument(
        "--model-pt",
        type=Path,
        default=None,
        help="Path to the incoming .pt file. If omitted, auto-detect a single .pt file under tools/in.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to tools/out/<stem>_pt_manifest.json.",
    )
    return parser.parse_args()


def resolve_pt_path(candidate: Path | None) -> Path:
    if candidate is not None:
        path = candidate.resolve()
        if not path.is_file():
            raise FileNotFoundError(f".pt file not found: {path}")
        return path

    pt_files = sorted(DEFAULT_INPUT_DIR.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {DEFAULT_INPUT_DIR}")
    if len(pt_files) > 1:
        names = ", ".join(path.name for path in pt_files)
        raise RuntimeError(
            "Multiple .pt files were found under tools/in. "
            f"Please specify one with --model-pt. Candidates: {names}"
        )
    return pt_files[0].resolve()


def find_archive_root(names: list[str]) -> str:
    if not names:
        return ""
    first = names[0]
    if "/" not in first:
        return ""
    return first.split("/", 1)[0]


def read_zip_entries(pt_path: Path) -> tuple[list[str], dict[str, bytes]]:
    with zipfile.ZipFile(pt_path, "r") as zf:
        names = zf.namelist()
        root = find_archive_root(names)
        interesting: dict[str, bytes] = {}
        for base_name in ("data.pkl", "constants.pkl", "version", "byteorder"):
            full_name = f"{root}/{base_name}" if root else base_name
            if full_name in names:
                interesting[base_name] = zf.read(full_name)
        return names, interesting


def pickle_strings(blob: bytes) -> list[str]:
    strings: list[str] = []
    for opcode, arg, _pos in pickletools.genops(blob):
        if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"} and isinstance(arg, str):
            strings.append(arg)
        elif opcode.name == "GLOBAL" and isinstance(arg, str):
            strings.append(arg)
    return strings


def detect_archive_type(entry_names: list[str]) -> str:
    normalized = {name.replace("\\", "/") for name in entry_names}
    has_constants = any(name.endswith("/constants.pkl") or name == "constants.pkl" for name in normalized)
    has_code = any("/code/" in name or name.startswith("code/") for name in normalized)
    has_data = any(name.endswith("/data.pkl") or name == "data.pkl" for name in normalized)

    if has_constants or has_code:
        return "torchscript_archive"
    if has_data:
        return "eager_checkpoint_or_state_dict"
    return "unknown_pt_archive"


def detect_checkpoint_kind(strings: list[str], archive_type: str) -> tuple[str, list[str]]:
    lower_strings = [text.lower() for text in strings]
    reasons: list[str] = []

    if archive_type == "torchscript_archive":
        reasons.append("Archive contains TorchScript-style entries such as constants/code.")
        return "torchscript", reasons

    state_dict_hits = sum(
        1
        for text in lower_strings
        if any(keyword in text for keyword in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"))
    )
    checkpoint_hits = sum(
        1
        for text in lower_strings
        if text in {"model_state_dict", "optimizer_state_dict", "epoch", "scheduler", "state_dict"}
    )

    if checkpoint_hits >= 2:
        reasons.append("Detected training-checkpoint keys such as epoch/optimizer/model_state_dict.")
        return "training_checkpoint", reasons
    if state_dict_hits >= 4:
        reasons.append("Detected many parameter-like tensor keys, which strongly suggests a state_dict.")
        return "state_dict", reasons

    reasons.append("Could not safely prove whether this is a full checkpoint or a raw state_dict.")
    return "unknown_pt_payload", reasons


def detect_architecture_hints(strings: list[str]) -> list[str]:
    hints = Counter()
    for text in strings:
        lower = text.lower()
        if "conv" in lower:
            hints["conv"] += 1
        if "bn" in lower or "batchnorm" in lower:
            hints["batchnorm"] += 1
        if "lstm" in lower:
            hints["lstm"] += 1
        if "gru" in lower:
            hints["gru"] += 1
        if "linear" in lower or "fc" in lower:
            hints["linear"] += 1
        if "embedding" in lower:
            hints["embedding"] += 1
        if "mel" in lower:
            hints["mel"] += 1
        if "audio" in lower:
            hints["audio"] += 1
    return [name for name, _count in hints.most_common(8)]


def find_companion_py_files(pt_path: Path) -> list[str]:
    parent = pt_path.parent
    siblings = sorted(path.name for path in parent.glob("*.py"))
    return siblings


def required_inputs(checkpoint_kind: str, companion_py_files: list[str]) -> list[str]:
    missing: list[str] = []
    if checkpoint_kind in {"state_dict", "training_checkpoint", "unknown_pt_payload"} and not companion_py_files:
        missing.append("model_definition_py")
    if checkpoint_kind == "unknown_pt_payload":
        missing.append("manual_pt_structure_review")
    return missing


def next_step_suggestion(checkpoint_kind: str, missing: list[str]) -> str:
    if "model_definition_py" in missing:
        return "Provide the matching model-definition .py file before attempting pt -> tflite conversion."
    if checkpoint_kind == "torchscript":
        return "Try a TorchScript-aware export path, then inspect whether direct TFLite conversion is feasible."
    if checkpoint_kind in {"state_dict", "training_checkpoint"}:
        return "Locate the matching .py model definition, load the weights, then continue to ONNX/TFLite export."
    return "Manual review is recommended before building the conversion pipeline."


def build_manifest(pt_path: Path) -> dict:
    manifest: dict = {
        "step": "identify_model_pt",
        "model_pt": str(pt_path),
        "file_name": pt_path.name,
        "file_size_bytes": pt_path.stat().st_size,
        "is_zip_serialized": zipfile.is_zipfile(pt_path),
    }

    if not manifest["is_zip_serialized"]:
        manifest.update(
            {
                "archive_type": "legacy_or_unknown_pt",
                "checkpoint_kind": "unknown_pt_payload",
                "companion_py_files": find_companion_py_files(pt_path),
                "missing_requirements": ["manual_pt_structure_review"],
                "next_step_suggestion": "This .pt file is not zip-serialized. Manual inspection is required first.",
            }
        )
        return manifest

    entry_names, interesting = read_zip_entries(pt_path)
    strings = pickle_strings(interesting.get("data.pkl", b""))
    archive_type = detect_archive_type(entry_names)
    checkpoint_kind, reasons = detect_checkpoint_kind(strings, archive_type)
    companion_py = find_companion_py_files(pt_path)
    missing = required_inputs(checkpoint_kind, companion_py)

    manifest.update(
        {
            "archive_type": archive_type,
            "checkpoint_kind": checkpoint_kind,
            "archive_entries_sample": entry_names[:40],
            "pickle_string_sample": strings[:80],
            "architecture_hints": detect_architecture_hints(strings),
            "companion_py_files": companion_py,
            "missing_requirements": missing,
            "reasons": reasons,
            "next_step_suggestion": next_step_suggestion(checkpoint_kind, missing),
        }
    )
    return manifest


def main() -> None:
    args = parse_args()
    pt_path = resolve_pt_path(args.model_pt)
    manifest = build_manifest(pt_path)

    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{pt_path.stem}_pt_manifest.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote identification report to {output_path}")
    print(text)


if __name__ == "__main__":
    main()
