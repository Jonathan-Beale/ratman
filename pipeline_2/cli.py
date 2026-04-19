"""
cli.py

Command-line interface for the pose -> character video pipeline.
"""

from __future__ import annotations

import argparse

from config import load_settings, Settings
from pipeline import run_pipeline
from utils_io import log


def cmd_run_all(settings: Settings, args: argparse.Namespace) -> None:
    run_pipeline(settings)


def cmd_generate(settings: Settings, args: argparse.Namespace) -> None:
    run_pipeline(
        settings=settings,
        pose_filter=args.pose,
        ref_filter=args.ref,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pose-video-driven AnimateDiff + ControlNet pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run-all", help="Run full pipeline on all inputs")

    gen = sub.add_parser("generate", help="Generate/refine video for a specific pose and/or reference")
    gen.add_argument("--pose", type=str, help="Pose video filename")
    gen.add_argument("--ref", type=str, help="Reference image filename")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings()

    if args.command == "run-all":
        cmd_run_all(settings, args)
    elif args.command == "generate":
        cmd_generate(settings, args)
    else:
        log(settings, "Unknown command")
        parser.error("Unknown command")


if __name__ == "__main__":
    main()