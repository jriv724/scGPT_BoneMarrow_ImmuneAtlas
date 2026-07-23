#!/usr/bin/env python3
from __future__ import annotations

import sys

from scgpt.pipelines.benchmark import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume cached stages.", file=sys.stderr)
        raise SystemExit(130)
