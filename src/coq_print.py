#!/usr/bin/env python3
"""
Coq Print/Check 查询接口（对外公开）
==================================

用途:
- 执行不依赖具体 theorem context 的查询命令
- 例如: Print, Check, About, Locate, Search

示例:
  python coq_print.py "Print nat."
  python coq_print.py "Check plus."
  python coq_print.py "Locate \"++\"."
  python coq_print.py "Print unique_key_in." --repo ../CoqStoq/test-repos/huffman --compile-args "-R theories Huffman"
"""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _find_coqtop() -> str:
    env_coqtop = os.environ.get("COQTOP")
    if env_coqtop and os.path.isfile(env_coqtop) and os.access(env_coqtop, os.X_OK):
        return env_coqtop

    for path_dir in os.environ.get("PATH", "").split(":"):
        candidate = os.path.join(path_dir, "coqtop")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    opam_prefix = os.environ.get("OPAM_SWITCH_PREFIX")
    if opam_prefix:
        candidate = os.path.join(opam_prefix, "bin", "coqtop")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return "coqtop"


def execute_print_command(
    query_command: str,
    setup_commands: Optional[List[str]] = None,
    setup_script: Optional[str] = None,
    compile_args: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    执行 Coq 查询命令（Print / Check / About / Locate / Search）。
    """
    setup_commands = setup_commands or []
    compile_args = compile_args or []
    coqtop_path = _find_coqtop()

    lines: List[str] = []
    if setup_script and setup_script.strip():
        script_block = setup_script
        if not script_block.endswith("\n"):
            script_block += "\n"
        lines.append(script_block.rstrip("\n"))

    for cmd in setup_commands:
        cmd = cmd.strip()
        if cmd:
            lines.append(cmd if cmd.endswith(".") else cmd + ".")

    query = query_command.strip()
    if not query:
        return {
            "success": False,
            "error": "query_command 不能为空",
            "command": query_command,
            "output": "",
        }
    lines.append(query if query.endswith(".") else query + ".")
    lines.append("Quit.")
    script = "\n".join(lines) + "\n"

    # 注意: `-batch` 不会按预期处理 stdin 交互命令，因此这里使用 `-quiet`
    # 并通过 stdin 提交查询脚本。
    cmd = [coqtop_path] + compile_args + ["-quiet"]
    run_cwd = cwd if cwd else os.getcwd()

    try:
        proc = subprocess.run(
            cmd,
            input=script,
            text=True,
            capture_output=True,
            cwd=run_cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"执行超时（>{timeout}s）",
            "command": query_command,
            "output": "",
            "raw_output": "",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"执行失败: {e}",
            "command": query_command,
            "output": "",
            "raw_output": "",
        }

    raw = (proc.stdout or "") + (proc.stderr or "")
    clean_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("Welcome to Coq"):
            continue
        if s == "Coq <":
            continue
        # coqtop 交互输出通常以 "Coq < " 开头；去掉提示符保留内容
        if line.startswith("Coq < "):
            content = line[len("Coq < "):].rstrip()
            if content:
                clean_lines.append(content)
            continue
        clean_lines.append(line.rstrip())

    clean = "\n".join(clean_lines).strip()
    ok = proc.returncode == 0 and "Error:" not in raw
    return {
        "success": ok,
        "error": None if ok else (clean or raw or "未知错误"),
        "command": query_command,
        "output": clean,
        "raw_output": raw,
        "coqtop": coqtop_path,
        "cwd": run_cwd,
        "compile_args": compile_args,
    }


def _parse_setup_args(values: List[str]) -> List[str]:
    result = []
    for value in values:
        if not value:
            continue
        result.extend([x for x in value.split(";") if x.strip()])
    return [x.strip() for x in result if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Coq print/check query tool (independent from theorem context).")
    parser.add_argument("command", help='查询命令，如 "Print nat." / "Check plus."')
    parser.add_argument(
        "--setup",
        action="append",
        default=[],
        help='预执行命令，可重复，例如 --setup "From mathcomp Require Import ssreflect ssrnat."',
    )
    parser.add_argument(
        "--compile-args",
        default="",
        help='传给 coqtop 的参数字符串，例如: "-R theories Huffman"',
    )
    parser.add_argument(
        "--repo",
        default="",
        help="工作目录（可选），用于项目内查询",
    )
    parser.add_argument("-j", "--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    compile_args = shlex.split(args.compile_args) if args.compile_args else []
    setup_commands = _parse_setup_args(args.setup)
    cwd = str(Path(args.repo).resolve()) if args.repo else None

    result = execute_print_command(
        args.command,
        setup_commands=setup_commands,
        compile_args=compile_args,
        cwd=cwd,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=" * 80)
        print("COQ QUERY RESULT")
        print("=" * 80)
        print(f"Command: {args.command}")
        if setup_commands:
            print(f"Setup: {' ; '.join(setup_commands)}")
        if compile_args:
            print(f"Compile Args: {' '.join(compile_args)}")
        if cwd:
            print(f"CWD: {cwd}")
        print("-" * 80)
        print(result["output"] if result["output"] else "(No output)")
        print("-" * 80)
        if not result["success"]:
            print("Error:")
            print(result["error"])
        print("=" * 80)
