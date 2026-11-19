#!/usr/bin/env python3
"""Run the full pipeline to produce evaluation/submission.csv, step by step."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


# ANSI bright colors
CYAN = "\033[96m"    # bright cyan
GREEN = "\033[92m"   # bright green
RED = "\033[91m"     # bright red
YELLOW = "\033[93m"  # bright yellow
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _as_lines(explanation: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize to one display line per entry (also splits embedded newlines)."""
    if isinstance(explanation, str):
        parts: list[str] = [explanation]
    else:
        parts = list(explanation)
    lines: list[str] = []
    for part in parts:
        chunk = str(part).splitlines() or [""]
        lines.extend(chunk)
    return tuple(lines)


@dataclass(frozen=True, init=False)
class Step:
    title: str
    explanation: tuple[str, ...]  # one display line each
    cmd: list[str]
    color: str = CYAN

    def __init__(
        self,
        title: str,
        explanation: str | list[str] | tuple[str, ...],
        cmd: list[str],
        color: str = CYAN,
    ):
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "explanation", _as_lines(explanation))
        object.__setattr__(self, "cmd", cmd)
        object.__setattr__(self, "color", color)


STEPS: list[Step] = [

    Step(
        title="Correct goalkeepers' trackers",
        explanation="Correct goalkeepers' trackers",
        cmd=[
            sys.executable,
            "data_correction/goalkeeper_trackers/run.py",
        ],
        color=YELLOW
    ),

    Step(
            title="Synchronize tactical and broadcast videos",
            explanation="Run the synchronization script",
            cmd=[
                sys.executable,
                "match_viewer/synchronize_videos.py",
            ],
            color=YELLOW
        ),

    Step(
        title="Annotate initial Dead-Ball intervals",
        explanation="Annotate initial Dead-Ball intervals",
        cmd=[
            sys.executable,
            "dead_ball_investigation/run.py",
        ],
        color=YELLOW
    ),

    Step(
        title="Build appearance banks",
        explanation=[
            "Learning the pixel information about the ball and teams (without goalkeepers) from the first 5 minutes of play",
            "Optionally if you want, you can create bigger banks from the whole match, but that's computationally heavy",
        ],
        cmd=[
            sys.executable,
            "-m",
            "ball_detection.build_banks",
            "--first",
            "11185",
            "--last",
            "18684",
        ],
        color=GREEN,
    ),
    Step(
        title="Run ball detection",
        explanation=[
            "Running the detection of the ball on the whole match (may take some time)",
            "Edit the --max-memory-mb flag to use more or less memory for the task (default 4GB)",
            "Checking every 10th kick, for faster detection, edit --kick-stride for striding",
        ],
        cmd=[
            sys.executable,
            "-m",
            "ball_detection.run_detect",
            "--first",
            "11185",
            "--last",
            "190197",
            "--max-link-dist-px",
            "28",
            "--foot-exclude-px",
            "14",
            "--min-bright-frac",
            "0.30",
            "--max-memory-mb",
            "4096",
            "--kick-stride",
            "10",
            "--kick-threshold",
            "4.0"
        ],
        color=GREEN,
    ),
    Step(
        title="Run kick attribution",
        explanation=[
            "Attributes to each ball detection event, which team kicked the ball",
            "Every event is with a given confidence between 0 and 1",
        ],
        cmd=[sys.executable, "-m", "ball_detection.run_attribute"],
        color=GREEN,
    ),
    Step(
        title="Run possession algorithm: lh_balldet_gk_dead",
        explanation=[
            "This is the LH algorithm with corrected goalkeepers and dead ball intervals",
            "It outputs lh_balldet_gk_dead.cvs in the evaluation/results folder",
        ],
        cmd=[sys.executable, "possesion_algorithms/run.py", "lh_balldet_gk_dead"],
        color=CYAN,
    ),
    Step(
        title="Run possession algorithm: lgbm_shape",
        explanation=[
            "This is the LiteGBM decision tree algorithm",
            "It outputs both, it's own intervals at lgbm_shape.csv and also it's confidence probabilities in lgbm_shape_proba.csv",
        ],
        cmd=[
            sys.executable,
            "possesion_algorithms/run.py",
            "lgbm_shape",
            "--name",
            "lgbm_shape",
        ],
        color=CYAN,
    ),
    Step(
        title="Combining LH and LGBM",
        explanation=[
            "Creates the final submission.csv in evaluation/results",
            "Combines intervals where LH and LGBM agree, and LGBM overwrites LH if it's confidence is above 0.7",
        ],
        cmd=[
            sys.executable,
            "make_submission.py",
            "evaluation/results/lh_balldet_gk_dead.csv",
            "evaluation/results/lgbm_shape.csv",
            "evaluation/results/lgbm_shape_proba.csv",
            "evaluation/results/submission.csv",
        ],
        color=CYAN,
    ),
]


def _print_banner(step: Step, index: int, total: int) -> None:
    sep = "─" * 72
    print()
    print(f"{step.color}{BOLD}{sep}{RESET}")
    print(f"{step.color}{BOLD}[{index}/{total}] {step.title}{RESET}")
    print("\n".join(f"{step.color}{line}{RESET}" for line in step.explanation))
    print(f"{DIM}$ {' '.join(step.cmd)}{RESET}")
    print(f"{step.color}{BOLD}{sep}{RESET}")
    print(flush=True)


def run_step(step: Step, index: int, total: int) -> None:
    _print_banner(step, index, total)
    # inherit stdout/stderr so command output streams live to the console
    result = subprocess.run(step.cmd, check=False)
    if result.returncode != 0:
        print(
            f"\n{BOLD}\033[91mCommand failed with exit code {result.returncode}."
            f" Stopping pipeline.{RESET}\n",
            flush=True,
        )
        sys.exit(result.returncode)
    print(f"{step.color}✓ Done: {step.title}{RESET}", flush=True)


def main() -> None:
    total = len(STEPS)
    print(f"{BOLD}Starting submission pipeline ({total} steps)…{RESET}", flush=True)
    for i, step in enumerate(STEPS, start=1):
        run_step(step, i, total)
    print()
    print(f"{GREEN}{BOLD}All steps finished successfully.{RESET}")


if __name__ == "__main__":
    main()
