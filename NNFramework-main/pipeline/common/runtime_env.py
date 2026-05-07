from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


DEFAULT_ENV_ROOT = Path(r"D:\work\AnacondaEnvironment")
DEFAULT_PYTHON = DEFAULT_ENV_ROOT / "python.exe"
BOOTSTRAP_SENTINEL = "PYTORCH_TO_BOARD_ENV_BOOTSTRAPPED"
OVERRIDE_PYTHON_ENV = "PYTORCH_TO_BOARD_PYTHON"


def resolve_default_python_executable(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    override = os.environ.get(OVERRIDE_PYTHON_ENV)
    if override:
        candidates.append(Path(override))

    candidates.append(DEFAULT_PYTHON)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "python.exe")

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.append(Path(virtual_env) / "Scripts" / "python.exe")

    candidates.append(Path(sys.executable))

    for candidate in candidates:
        resolved = _normalize_path(candidate)
        if resolved.is_file():
            return resolved

    return _normalize_path(Path(sys.executable))


def ensure_default_python_for_script(script_path: str | os.PathLike[str]) -> Path:
    script = _normalize_path(Path(script_path))
    return _ensure_default_python([str(script), *sys.argv[1:]])


def ensure_default_python_for_module(module_name: str) -> Path:
    return _ensure_default_python(["-m", module_name, *sys.argv[1:]])


def _ensure_default_python(run_args: list[str]) -> Path:
    target_python = resolve_default_python_executable()
    current_python = _normalize_path(Path(sys.executable))

    if _same_path(current_python, target_python):
        return target_python
    if os.environ.get(BOOTSTRAP_SENTINEL) == "1":
        return current_python

    env = _build_bootstrap_env(target_python)
    completed = subprocess.run([str(target_python), *run_args], env=env)
    raise SystemExit(completed.returncode)


def _build_bootstrap_env(target_python: Path) -> dict[str, str]:
    env = os.environ.copy()
    env[BOOTSTRAP_SENTINEL] = "1"
    env[OVERRIDE_PYTHON_ENV] = str(target_python)

    env_root = target_python.parent
    env["CONDA_PREFIX"] = str(env_root)

    path_entries = [
        str(env_root),
        str(env_root / "Scripts"),
        str(env_root / "Library" / "bin"),
    ]
    existing_path = env.get("PATH", "")
    if existing_path:
        env["PATH"] = os.pathsep.join([*path_entries, existing_path])
    else:
        env["PATH"] = os.pathsep.join(path_entries)
    return env


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _same_path(left: Path, right: Path) -> bool:
    return str(left).lower() == str(right).lower()
