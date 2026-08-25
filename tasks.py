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


@task("Create .venv and install the package (add --extras dense,demo for optional groups)")
def setup(args: argparse.Namespace) -> None:
    extras = f"[{args.extras}]" if args.extras else "[dev]"
    uv = _uv()
    if uv:
        run([uv, "venv", "--python", "3.11", str(VENV)])
        run([uv, "pip", "install", "--python", str(VENV_PY), "-e", f".{extras}"])
    else:
        print("uv not found; falling back to venv + pip (slower).")
        run([sys.executable, "-m", "venv", str(VENV)])
        run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(VENV_PY), "-m", "pip", "install", "-e", f".{extras}"])
    print("\nInstalled. Optional extras:")
    print("  python tasks.py setup --extras dev,dense   # dense retrieval + local NLI entailment")
    print("  python tasks.py setup --extras dev,demo    # Gradio demo")


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
    print(f"System Python      : {major}.{minor} ({sys.executable})")

    if VENV_PY.exists():
        out = subprocess.run(
            [str(VENV_PY), "--version"], capture_output=True, text=True, check=False
        )
        print(f"Project virtualenv : {out.stdout.strip() or out.stderr.strip()} at {VENV_PY}")
        if "3.11" not in (out.stdout + out.stderr):
            print("  ! The project pins Python 3.11 (>=3.11,<3.12). Recreate the venv with 3.11.")
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
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="extra args passed through")
    args = parser.parse_args()

    if not args.task:
        print(__doc__)
        print("Tasks:")
        for name, help_text in sorted(TASKS.items()):
            print(f"  {name:<10} {help_text}")
        sys.exit(0)

    globals()[args.task.replace("-", "_")](args)


if __name__ == "__main__":
    main()
