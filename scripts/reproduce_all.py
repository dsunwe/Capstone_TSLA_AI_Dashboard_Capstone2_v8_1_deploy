"""Recompute reviewed RQ2-RQ4 modules in isolated virtual environments."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict


ROOT = Path(__file__).resolve().parents[1]
VENV_ROOT = ROOT / ".research-venvs"


class StageSettings(TypedDict, total=False):
    directory: Path
    environment: str
    requirements: Path
    commands: list[list[str]]


STAGES: dict[str, StageSettings] = {
    "rq2": {
        "directory": ROOT / "research" / "rq2",
        "commands": [
            ["run_rq2_experiment.py"],
            ["validate_rq2.py"],
            ["build_notebook.py"],
        ],
    },
    "rq3": {
        "directory": ROOT / "research" / "rq3",
        "commands": [
            ["run_rq3_experiment.py"],
            ["build_notebook.py"],
            ["validate_rq3.py", "--write-artifacts"],
        ],
    },
    "rq4": {
        "directory": ROOT / "research" / "rq4",
        "commands": [
            ["run_rq4_experiment.py"],
            ["validate_rq4.py", "--write-artifacts"],
            ["build_readme.py"],
            ["build_notebook.py"],
        ],
    },
    "rq4-cross-asset": {
        "directory": ROOT / "research" / "rq4",
        "environment": "rq4-cross-asset",
        "requirements": (
            ROOT / "research" / "rq4" / "requirements-cross-asset.txt"
        ),
        "commands": [
            ["run_cross_asset_predictions.py"],
            ["run_rq4_experiment.py"],
            ["validate_rq4.py", "--write-artifacts"],
            ["build_readme.py"],
            ["build_notebook.py"],
        ],
    },
}


def environment_python(environment: Path) -> Path:
    """Return the environment interpreter for the current operating system."""

    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def run(command: list[str], cwd: Path) -> None:
    """Run one command and stop on the first failure."""

    print(f"\n[{cwd.relative_to(ROOT)}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def validate_stage_layout(stage: str) -> None:
    """Fail before installation if a reviewed stage is incomplete."""

    settings = STAGES[stage]
    directory = settings["directory"]
    requirements = settings.get("requirements", directory / "requirements.txt")
    required = [
        requirements,
        *(directory / args[0] for args in settings["commands"]),
    ]
    missing = [path.relative_to(ROOT) for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Stage {stage!r} is incomplete; missing: {joined}")


def prepare_environment(stage: str, install: bool) -> Path:
    """Create or reuse the stage-specific environment."""

    settings = STAGES[stage]
    environment_name = settings.get("environment", stage)
    environment = VENV_ROOT / environment_name
    python = environment_python(environment)
    requirements = settings.get(
        "requirements",
        settings["directory"] / "requirements.txt",
    )

    if install:
        if not python.exists():
            run([sys.executable, "-m", "venv", str(environment)], ROOT)
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"], ROOT)
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements),
            ],
            ROOT,
        )

    if not python.exists():
        raise SystemExit(
            f"Missing environment: {environment.relative_to(ROOT)}. "
            f"Re-run with --install."
        )
    return python


def execute_stage(stage: str, install: bool) -> None:
    """Execute a reviewed stage in its isolated environment."""

    validate_stage_layout(stage)
    settings = STAGES[stage]
    python = prepare_environment(stage, install)
    for script_args in settings["commands"]:
        run([str(python), *script_args], settings["directory"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute one reviewed research module in its own virtual "
            "environment. The Streamlit Dashboard never runs this automatically."
        )
    )
    parser.add_argument(
        "--stage",
        choices=[*STAGES, "all"],
        required=True,
        help=(
            "Use rq4 for the packaged prediction ledger; rq4-cross-asset first "
            "refits AAPL and NVDA. all runs rq2, rq3, then standard rq4."
        ),
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Create the selected isolated environment and install its requirements.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = ["rq2", "rq3", "rq4"] if args.stage == "all" else [args.stage]
    for stage in stages:
        execute_stage(stage, args.install)
    print(
        "\nRecomputation completed. It may have changed frozen research files; "
        "review the regenerated evidence before rebuilding any release manifest."
    )


if __name__ == "__main__":
    main()
