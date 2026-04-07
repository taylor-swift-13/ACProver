#!/usr/bin/env python3
"""Legacy entrypoint kept only to report that proving is disabled."""

from __future__ import annotations

import json
import sys


def main() -> None:
    payload = {
        "success": False,
        "error": "proving is disabled in this repository; use `build-stdlib-index` and `query-experience` instead.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
