#!/usr/bin/env python3
"""Small CLI wrapper for natural-language and SQL experience retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from src.experience_retrieval import (
        query_coqstoq_by_description,
        query_coqstoq_sql,
        query_stdlib_by_description,
        query_stdlib_sql,
    )
except ModuleNotFoundError:
    from experience_retrieval import (  # type: ignore
        query_coqstoq_by_description,
        query_coqstoq_sql,
        query_stdlib_by_description,
        query_stdlib_sql,
    )


def _query_nl(domain: str, description: str, k: int) -> Dict[str, Any]:
    if domain == "stdlib":
        hits = query_stdlib_by_description(description, limit=k)
    elif domain == "coqstoq":
        hits = query_coqstoq_by_description(description, limit=k)
    else:
        raise ValueError(f"unsupported domain: {domain}")
    return {
        "success": True,
        "mode": "natural_language",
        "domain": domain,
        "description": description,
        "k": k,
        "hits": hits,
    }


def _query_sql(domain: str, sql: str) -> Dict[str, Any]:
    if domain == "stdlib":
        result = query_stdlib_sql(sql)
    elif domain == "coqstoq":
        result = query_coqstoq_sql(sql)
    else:
        raise ValueError(f"unsupported domain: {domain}")
    result["success"] = True
    result["mode"] = "sql"
    result["domain"] = domain
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Query ACProver experience records.")
    parser.add_argument("--domain", choices=["stdlib", "coqstoq"], default="stdlib")
    parser.add_argument("--description", help="Natural-language query for semantic retrieval.")
    parser.add_argument("--sql", help="SQL query over metadata.db.")
    parser.add_argument("-k", type=int, default=5, help="Top-k hits for natural-language retrieval.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.description) == bool(args.sql):
        raise SystemExit("Pass exactly one of --description or --sql.")
    if args.description:
        result = _query_nl(args.domain, args.description, args.k)
    else:
        result = _query_sql(args.domain, args.sql)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
