from __future__ import annotations

from contextlib import contextmanager
import inspect
import pickletools
from pathlib import Path
import sys
from typing import Any
import zipfile

try:
    import torch
except ImportError:  # pragma: no cover - import guard
    torch = None


_PICKLE_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}
_BENIGN_GLOBAL_PREFIXES = (
    "builtins",
    "collections",
    "copyreg",
    "functools",
    "itertools",
    "math",
    "numpy",
    "operator",
    "torch",
    "typing",
)
_WEIGHTS_ONLY_RETRY_HINTS = (
    "weights only load failed",
    "weights_only",
    "unsupported global",
    "safe_globals",
    "can still be loaded",
    "torchscript archives",
)


def is_torchscript_archive(path: Path) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    entry_names, _interesting = _read_zip_entries(path)
    return _detect_archive_type(entry_names) == "torchscript_archive"


def extract_pickle_strings(path: Path) -> list[str]:
    if not zipfile.is_zipfile(path):
        return []
    _entry_names, interesting = _read_zip_entries(path)
    return _pickle_strings(interesting.get("data.pkl", b""))


def extract_serialized_global_refs(path: Path) -> list[tuple[str, str]]:
    if not zipfile.is_zipfile(path):
        return []
    _entry_names, interesting = _read_zip_entries(path)
    return _pickle_global_refs(interesting.get("data.pkl", b""))


def serialized_user_global_refs(path: Path) -> list[tuple[str, str]]:
    refs = []
    for module_name, global_name in extract_serialized_global_refs(path):
        if _looks_like_user_defined_global(module_name, global_name):
            refs.append((module_name, global_name))
    return refs


def extract_state_dict(loaded_object: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(loaded_object, dict):
        return None, None

    if all(isinstance(key, str) for key in loaded_object.keys()):
        tensor_like_values = sum(hasattr(value, "shape") for value in loaded_object.values())
        if tensor_like_values > 0:
            return loaded_object, "plain_state_dict"

    for key in ("model_state_dict", "state_dict"):
        value = loaded_object.get(key)
        if isinstance(value, dict):
            return value, key
    return None, None


def load_torch_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
    model_module: Any | None = None,
) -> tuple[Any, str]:
    # One entrypoint for eager checkpoints and TorchScript archives.
    if torch is None:
        raise RuntimeError("PyTorch is not installed in the current Python environment")

    if is_torchscript_archive(checkpoint_path):
        scripted_model = torch.jit.load(str(checkpoint_path), map_location=map_location)
        return scripted_model, "torchscript"

    alias_map = build_module_alias_map(checkpoint_path, model_module)
    if not _supports_weights_only_parameter():
        with _temporary_module_aliases(alias_map):
            return torch.load(checkpoint_path, map_location=map_location), "legacy_torch_load"

    try:
        with _temporary_module_aliases(alias_map):
            return (
                torch.load(checkpoint_path, map_location=map_location, weights_only=True),
                "weights_only",
            )
    except Exception as exc:
        if not _should_retry_with_unsafe_load(exc):
            raise

    with _temporary_module_aliases(alias_map):
        return (
            torch.load(checkpoint_path, map_location=map_location, weights_only=False),
            "compat_full_load",
        )


def build_module_alias_map(checkpoint_path: Path, model_module: Any | None) -> dict[str, Any]:
    # Reuse the loaded user module under serialized archive module names.
    if model_module is None:
        return {}

    module_classes = {
        name
        for name, value in vars(model_module).items()
        if isinstance(value, type)
    }
    alias_map: dict[str, Any] = {}
    for module_name, global_name in serialized_user_global_refs(checkpoint_path):
        if global_name in module_classes:
            alias_map[module_name] = model_module
    return alias_map


@contextmanager
def _temporary_module_aliases(alias_map: dict[str, Any]):
    previous: dict[str, Any] = {}
    missing = object()
    for module_name, module_obj in alias_map.items():
        previous[module_name] = sys.modules.get(module_name, missing)
        sys.modules[module_name] = module_obj
    try:
        yield
    finally:
        for module_name, old_value in previous.items():
            if old_value is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_value


def _supports_weights_only_parameter() -> bool:
    if torch is None:
        return False
    try:
        parameters = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False
    return "weights_only" in parameters


def _should_retry_with_unsafe_load(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _WEIGHTS_ONLY_RETRY_HINTS)


def _read_zip_entries(pt_path: Path) -> tuple[list[str], dict[str, bytes]]:
    with zipfile.ZipFile(pt_path, "r") as archive:
        names = archive.namelist()
        root = _find_archive_root(names)
        interesting: dict[str, bytes] = {}
        for base_name in ("data.pkl", "constants.pkl", "version", "byteorder"):
            full_name = f"{root}/{base_name}" if root else base_name
            if full_name in names:
                interesting[base_name] = archive.read(full_name)
    return names, interesting


def _find_archive_root(names: list[str]) -> str:
    if not names:
        return ""
    first = names[0]
    if "/" not in first:
        return ""
    return first.split("/", 1)[0]


def _pickle_strings(blob: bytes) -> list[str]:
    strings: list[str] = []
    for opcode, arg, _pos in pickletools.genops(blob):
        if opcode.name in _PICKLE_STRING_OPS and isinstance(arg, str):
            strings.append(arg)
        elif opcode.name == "GLOBAL" and isinstance(arg, str):
            strings.append(arg)
    return strings


def _pickle_global_refs(blob: bytes) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    recent_strings: list[str] = []

    for opcode, arg, _pos in pickletools.genops(blob):
        if opcode.name in _PICKLE_STRING_OPS and isinstance(arg, str):
            recent_strings.append(arg)
            if len(recent_strings) > 8:
                recent_strings = recent_strings[-8:]
            continue

        if opcode.name == "GLOBAL" and isinstance(arg, str):
            ref = _parse_global_arg(arg)
        elif opcode.name == "STACK_GLOBAL":
            ref = _parse_stack_global(recent_strings)
        else:
            ref = None

        if ref is not None and ref not in seen:
            seen.add(ref)
            refs.append(ref)

    return refs


def _parse_global_arg(arg: str) -> tuple[str, str] | None:
    parts = arg.replace("\n", " ").split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _parse_stack_global(recent_strings: list[str]) -> tuple[str, str] | None:
    if len(recent_strings) < 2:
        return None
    return recent_strings[-2], recent_strings[-1]


def _detect_archive_type(entry_names: list[str]) -> str:
    normalized = {name.replace("\\", "/") for name in entry_names}
    has_constants = any(name.endswith("/constants.pkl") or name == "constants.pkl" for name in normalized)
    has_code = any("/code/" in name or name.startswith("code/") for name in normalized)
    has_data = any(name.endswith("/data.pkl") or name == "data.pkl" for name in normalized)
    if has_constants or has_code:
        return "torchscript_archive"
    if has_data:
        return "eager_checkpoint_or_state_dict"
    return "unknown_pt_archive"


def _looks_like_user_defined_global(module_name: str, global_name: str) -> bool:
    del global_name
    return not module_name.startswith(_BENIGN_GLOBAL_PREFIXES)
