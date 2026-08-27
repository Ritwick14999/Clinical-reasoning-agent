#!/usr/bin/env python
"""Cross-platform task runner.

The Makefile assumes a POSIX shell and `.venv/bin/python`; on Windows neither
holds. This script does the same jobs with nothing but a system Python, so the
documented commands work identically on Windows, macOS and Linux:

    python tasks.py setup     install into .venv
    python tasks.py test      run the test suite
    python tasks.py demo      one agent episode, no credentials needed
    python tasks.py doctor    check the local environment and Ollama

Run `python tasks.py` with no arguments to list every task.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
IS_WINDOWS = os.name == "nt"
VENV = ROOT / ".venv"
# The one difference that breaks the Makefile on Windows.
VENV_PY = VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
REQUIRED_MODELS = ["qwen3:8b", "llama3.1:8b"]

TASKS: dict[str, str] = {}


def task(help_text: str):
    def wrap(fn):
        TASKS[fn.__name__.replace("_", "-")] = help_text
        return fn

    return wrap


def run(cmd: list[str], check: bool = True, **kw) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, **kw)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode


def venv_python() -> str:
    if not VENV_PY.exists():
        sys.exit(
            f"No virtualenv at {VENV}.\n"
            f"Run: python tasks.py setup"
        )
    return str(VENV_PY)


def _uv() -> str | None:
    return shutil.which("uv")


REQUIRED_PY = (3, 11)


def _version_of(cmd: list[str]) -> tuple[int, int] | None:
    """Return the (major, minor) an interpreter command reports, or None."""
    try:
        out = subprocess.run(
            [*cmd, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.stdout.split()
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def find_python311() -> list[str] | None:
    """Locate a 3.11 interpreter without uv.

    The project pins 3.11 because several pinned dependencies (torch and the
    dense-retrieval stack in particular) have no wheels for newer versions.
    Building the venv from whatever `python` happens to be on PATH produces a
    venv that only fails later, at install time, so this runs first.
    """
    candidates: list[list[str]] = []
    if IS_WINDOWS:
        # The py launcher is the reliable way to reach a specific version on Windows.
        candidates.append(["py", "-3.11"])
    candidates += [["python3.11"], ["python3.11.exe"], [sys.executable]]
    for cmd in candidates:
        if shutil.which(cmd[0]) is None and cmd[0] != sys.executable:
            continue
        if _version_of(cmd) == REQUIRED_PY:
            return cmd
    return None


UV_INSTALL_HINT = """
Install uv -- it can fetch Python 3.11 for you, so nothing else needs installing:

    pip install uv

or the standalone installer:
    Windows : powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

Then re-run:  python tasks.py setup

Alternatively, install Python 3.11 yourself from https://www.python.org/downloads/release/python-3119/
(tick "Add python.exe to PATH" on Windows) and re-run this command."""


def _existing_venv_version() -> tuple[int, int] | None:
    return _version_of([str(VENV_PY)]) if VENV_PY.exists() else None


def _running_inside_project_venv() -> bool:
    """True when this script is itself running on the project's virtualenv.

    Deleting that venv would mean deleting the running interpreter, which
    Windows refuses outright (the .exe is locked) and which leaves a
    half-removed directory behind on any platform.
    """
    try:
        return Path(sys.executable).resolve().is_relative_to(VENV.resolve())
    except (OSError, ValueError):
        return False


@task("Create .venv and install the package (add --extras dense,demo for optional groups)")
def setup(args: argparse.Namespace) -> None:
    extras = f"[{args.extras}]" if args.extras else "[dev]"

    existing = _existing_venv_version()
    if existing is not None and existing != REQUIRED_PY and not args.force:
        sys.exit(
            f"{VENV} already exists but runs Python {existing[0]}.{existing[1]}, "
            f"and this project needs {REQUIRED_PY[0]}.{REQUIRED_PY[1]}.\n"
            f"Re-run with --force to delete and rebuild it:\n"
            f"    python tasks.py setup --force"
        )
    if args.force and VENV.exists():
        if _running_inside_project_venv():
            sys.exit(
                "Cannot rebuild .venv while running from inside it "
                f"({sys.executable}).\n"
                "Deactivate it first, then re-run:\n"
                "    deactivate\n"
                "    python tasks.py setup --force"
            )
        print(f"Removing {VENV}")
        shutil.rmtree(VENV, ignore_errors=True)
        if VENV.exists():
            sys.exit(
                f"Could not fully remove {VENV} -- something is still using it.\n"
                "Close any terminal or editor using that environment and try again."
            )

    uv = _uv()
    if uv:
        # uv downloads a matching interpreter when the system has none.
        run([uv, "venv", "--python", "3.11", str(VENV)])
        run([uv, "pip", "install", "--python", str(VENV_PY), "-e", f".{extras}"])
    else:
        base = find_python311()
        if base is None:
            running = f"{sys.version_info[0]}.{sys.version_info[1]}"
            sys.exit(
                f"No Python {REQUIRED_PY[0]}.{REQUIRED_PY[1]} interpreter found "
                f"(you are running {running}), and uv is not installed.\n"
                f"{UV_INSTALL_HINT}"
            )
        print(f"uv not found; building the venv with {' '.join(base)} (slower than uv).")
        run([*base, "-m", "venv", str(VENV)])
        run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(VENV_PY), "-m", "pip", "install", "-e", f".{extras}"])

    built = _existing_venv_version()
    if built != REQUIRED_PY:
        sys.exit(f"Unexpected: the new venv reports Python {built}. Expected {REQUIRED_PY}.")

    print("\nInstalled. Optional extras:")
    print("  python tasks.py setup --extras dev,dense   # dense retrieval + local NLI entailment")
    print("  python tasks.py setup --extras dev,demo    # Gradio demo")
    print("\nNext:  python tasks.py doctor")


@task("Run the test suite (no network, no credentials)")
def test(args: argparse.Namespace) -> None:
    run([venv_python(), "-m", "pytest", *args.rest])


@task("Lint with ruff")
def lint(args: argparse.Namespace) -> None:
    run([venv_python(), "-m", "ruff", "check", "src", "tests", "scripts", "tasks.py"])


@task("Auto-fix lint issues")
def fmt(args: argparse.Namespace) -> None:
    run([venv_python(), "-m", "ruff", "check", "--fix", "src", "tests", "scripts", "tasks.py"])


@task("Run one agent episode on the mock client and print the trace")
def demo(args: argparse.Namespace) -> None:
    run([venv_python(), "scripts/demo_episode.py", *args.rest])


@task("Check Python, the virtualenv, and whether Ollama is serving the needed models")
def doctor(args: argparse.Namespace) -> None:
    ok = True

    major, minor = sys.version_info[:2]
    where = " [this is the project venv -- run `deactivate` to see your system Python]" if (
        _running_inside_project_venv()
    ) else ""
    print(f"Running Python     : {major}.{minor} ({sys.executable}){where}")

    if VENV_PY.exists():
        out = subprocess.run(
            [str(VENV_PY), "--version"], capture_output=True, text=True, check=False
        )
        print(f"Project virtualenv : {out.stdout.strip() or out.stderr.strip()} at {VENV_PY}")
        if "3.11" not in (out.stdout + out.stderr):
            print("  ! The project pins Python 3.11 (>=3.11,<3.12): several pinned deps have no")
            print("    wheels for newer versions. Rebuild with: python tasks.py setup --force")
            ok = False
    else:
        print("Project virtualenv : MISSING -- run `python tasks.py setup`")
        ok = False

    print(f"uv                 : {_uv() or 'not found (pip fallback will be used)'}")

    print(f"Ollama at {OLLAMA_URL}")
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read())
        installed = sorted(m.get("name", "") for m in tags.get("models", []))
        print(f"  reachable, {len(installed)} model(s) installed")
        for name in installed:
            print(f"    - {name}")
        for needed in REQUIRED_MODELS:
            # Ollama reports "qwen3:8b"; tolerate a missing ":latest" suffix.
            if not any(i == needed or i.startswith(needed) for i in installed):
                print(f"  ! {needed} not installed -- run: ollama pull {needed}")
                ok = False
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  ! not reachable ({exc}).")
        print("    Start it with `ollama serve`, or install from https://ollama.com/download")
        print("    Ollama is only needed for real rollouts; tests and the demo run without it.")
        ok = False

    print("\n" + ("All checks passed." if ok else "Some checks failed -- see the ! lines above."))
    sys.exit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", nargs="?", choices=sorted(TASKS), help="task to run")
    parser.add_argument("--extras", default="", help="setup only: extras, e.g. dev,dense")
    parser.add_argument(
        "--force", action="store_true", help="setup only: delete and rebuild an existing .venv"
    )
    # Not argparse.REMAINDER: as a positional it swallows everything after the
    # task name, declared flags included, so `setup --force` silently lost the
    # flag. parse_known_args keeps declared flags and passes the rest through.
    args, rest = parser.parse_known_args()
    args.rest = rest

    if not args.task:
        print(__doc__)
        print("Tasks:")
        for name, help_text in sorted(TASKS.items()):
            print(f"  {name:<10} {help_text}")
        sys.exit(0)

    globals()[args.task.replace("-", "_")](args)


if __name__ == "__main__":
    main()
