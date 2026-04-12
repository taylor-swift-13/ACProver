#!/usr/bin/env python3
"""Local retrieval tools for standard-library and CoqStoq theorem records."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

try:
    from experience_retrieval import (
        query_coqstoq_by_description,
        query_coqstoq_sql,
        query_stdlib_by_description,
        query_stdlib_sql,
    )
    from experience_store import experience_domain_root, refresh_experience_indexes
    from stdlib_index import build_and_write
except ModuleNotFoundError:
    from .experience_retrieval import (
        query_coqstoq_by_description,
        query_coqstoq_sql,
        query_stdlib_by_description,
        query_stdlib_sql,
    )
    from .experience_store import experience_domain_root, refresh_experience_indexes
    from .stdlib_index import build_and_write


def cmd_query_stdlib(args: argparse.Namespace) -> Dict[str, Any]:
    hits = query_stdlib_by_description(args.description, limit=args.k)
    return {
        "success": True,
        "source": "stdlib",
        "description": args.description,
        "k": args.k,
        "hits": hits,
    }


def cmd_build_stdlib_index(args: argparse.Namespace) -> Dict[str, Any]:
    result = build_and_write(args.module_path, rebuild_indexes=not args.no_rebuild_indexes)
    result["source"] = "stdlib"
    return result


def cmd_query_coqstoq(args: argparse.Namespace) -> Dict[str, Any]:
    hits = query_coqstoq_by_description(args.description, limit=args.k)
    return {
        "success": True,
        "source": "coqstoq",
        "description": args.description,
        "k": args.k,
        "hits": hits,
    }


def cmd_query_stdlib_sql(args: argparse.Namespace) -> Dict[str, Any]:
    result = query_stdlib_sql(args.sql)
    result["success"] = True
    result["source"] = "stdlib"
    return result


def cmd_query_coqstoq_sql(args: argparse.Namespace) -> Dict[str, Any]:
    result = query_coqstoq_sql(args.sql)
    result["success"] = True
    result["source"] = "coqstoq"
    return result


def cmd_build_coqstoq_index(args: argparse.Namespace) -> Dict[str, Any]:
    root = experience_domain_root("coqstoq")
    refresh = refresh_experience_indexes(root)
    return {
        "success": True,
        "source": "coqstoq",
        "message": "Rebuilt CoqStoq metadata and FAISS indexes from existing records.",
        "experience_root": str(root),
        "refresh": refresh,
    }


def cmd_build_stdlib_from_existing(args: argparse.Namespace) -> Dict[str, Any]:
    root = experience_domain_root("stdlib")
    refresh = refresh_experience_indexes(root)
    return {
        "success": True,
        "source": "stdlib",
        "message": "Rebuilt stdlib metadata and FAISS indexes from existing records.",
        "experience_root": str(root),
        "refresh": refresh,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Local retrieval tools for standard-library and CoqStoq theorem records.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stdlib_query = subparsers.add_parser("query-stdlib")
    stdlib_query.add_argument("--description", required=True)
    stdlib_query.add_argument("-k", type=int, default=5)

    stdlib_sql = subparsers.add_parser("query-stdlib-sql")
    stdlib_sql.add_argument("--sql", required=True)

    stdlib_parser = subparsers.add_parser("build-stdlib-index")
    stdlib_parser.add_argument("--module-path", default="Coq.Lists.List")
    stdlib_parser.add_argument("--no-rebuild-indexes", action="store_true")

    subparsers.add_parser("build-stdlib-from-existing")

    coqstoq_query = subparsers.add_parser("query-coqstoq")
    coqstoq_query.add_argument("--description", required=True)
    coqstoq_query.add_argument("-k", type=int, default=5)

    coqstoq_sql = subparsers.add_parser("query-coqstoq-sql")
    coqstoq_sql.add_argument("--sql", required=True)

    subparsers.add_parser("build-coqstoq-index")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "query-stdlib":
        result = cmd_query_stdlib(args)
    elif args.command == "query-stdlib-sql":
        result = cmd_query_stdlib_sql(args)
    elif args.command == "build-stdlib-index":
        result = cmd_build_stdlib_index(args)
    elif args.command == "build-stdlib-from-existing":
        result = cmd_build_stdlib_from_existing(args)
    elif args.command == "query-coqstoq":
        result = cmd_query_coqstoq(args)
    elif args.command == "query-coqstoq-sql":
        result = cmd_query_coqstoq_sql(args)
    elif args.command == "build-coqstoq-index":
        result = cmd_build_coqstoq_index(args)
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
