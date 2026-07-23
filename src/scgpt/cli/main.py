#!/usr/bin/env python3
from __future__ import annotations

import sys
import importlib


USAGE = (
    "Usage: scgpt <subcommand> [args]\n\n"
    "Subcommands:\n"
    "  embed       Run atlas embedding pipeline\n"
    "  benchmark   Run checkpoint benchmark pipeline\n"
    "  annotate    Run cell-type annotation audit pipeline\n"
    "  pipeline    Run full atlas pipeline orchestrator\n"
)


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    subcommand = argv[0]
    forwarded = argv[1:]
    commands = {
        "embed": "scgpt.pipelines.embed",
        "benchmark": "scgpt.pipelines.benchmark",
        "annotate": "scgpt.pipelines.annotate",
        "pipeline": "scgpt.pipelines.full_pipeline",
    }
    module_name = commands.get(subcommand)
    if module_name is None:
        print(f"Unknown subcommand: {subcommand}\n\n{USAGE}", file=sys.stderr)
        return 2

    previous_argv = sys.argv
    try:
        sys.argv = [f"scgpt {subcommand}", *forwarded]
        module = importlib.import_module(module_name)
        return int(module.main())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
