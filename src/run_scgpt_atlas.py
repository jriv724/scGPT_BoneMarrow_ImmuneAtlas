#!/usr/bin/env python3
from __future__ import annotations

import sys

from scgpt.pipelines.embed import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
