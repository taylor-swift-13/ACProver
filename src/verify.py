#!/usr/bin/env python3
"""
CoqStoq 证明验证器接口
========================

提供真正的编译验证方法。

主要功能:
- verify_proof: 替换原证明并使用 coqc 验证
- get_theorem_info: 获取定理信息
- list_theorems: 列出可用定理

依赖状态:
- mathcomp: 已安装（ssreflect 可用）
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TheoremInfo:
    """定理信息"""
    theorem_id: str
    project: str
    file: str
    statement: str
    theorem_line: int
    proof_start_line: Optional[int] = None
    proof_end_line: Optional[int] = None
    compile_args: List[str] = field(default_factory=list)


class CoqProofVerifier:
    """Coq 证明验证器"""

    def __init__(self, coqstoq_path: Optional[str] = None):
        """初始化验证器"""
        if coqstoq_path is None:
            coqstoq_path = self._resolve_coqstoq_path()

        self.coqstoq_path = coqstoq_path
        self.coqc_path = self._resolve_coqc_path()
        self.coqtop_path = self._resolve_coqtop_path()
        self._index_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _resolve_coqstoq_path(self) -> str:
        """解析 CoqStoq 路径，支持环境变量、当前目录、脚本目录和上级目录。"""
        env_path = os.environ.get("COQSTOQ_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        script_dir = Path(__file__).resolve().parent
        candidates = [
            os.path.join(os.getcwd(), "CoqStoq"),
            os.path.join(script_dir, "CoqStoq"),
            os.path.join(script_dir.parent, "CoqStoq"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _resolve_coqc_path(self) -> str:
        """解析 coqc 可执行文件路径。"""
        env_coqc = os.environ.get("COQC")
        if env_coqc and os.path.isfile(env_coqc) and os.access(env_coqc, os.X_OK):
            return env_coqc

        from_path = shutil.which("coqc")
        if from_path:
            return from_path

        opam_prefix = os.environ.get("OPAM_SWITCH_PREFIX")
        if opam_prefix:
            candidate = os.path.join(opam_prefix, "bin", "coqc")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        # 兼容历史路径，若存在则使用；否则返回 "coqc" 让后续错误更清晰
        legacy = "/root/.opam/coqswitch/bin/coqc"
        if os.path.isfile(legacy) and os.access(legacy, os.X_OK):
            return legacy

        return "coqc"

    def _resolve_coqtop_path(self) -> str:
        """解析 coqtop 可执行文件路径。"""
        env_coqtop = os.environ.get("COQTOP")
        if env_coqtop and os.path.isfile(env_coqtop) and os.access(env_coqtop, os.X_OK):
            return env_coqtop

        from_path = shutil.which("coqtop")
        if from_path:
            return from_path

        opam_prefix = os.environ.get("OPAM_SWITCH_PREFIX")
        if opam_prefix:
            candidate = os.path.join(opam_prefix, "bin", "coqtop")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        return "coqtop"

    def _parse_theorem_id(self, theorem_id: str) -> tuple:
        """解析定理ID，支持格式:
        - "test:N" (N 是 test, val 或 cutoff 定理的索引）
        - "val:N" (N 是 val 定理的索引）
        - "cutoff:N" (N 是 cutoff 定理的索引）
        - "N" (仅数字，使用 test 定理的索引）

        Returns:
            (split_name, index)
        """
        if ":" in theorem_id:
            parts = theorem_id.split(":")
            split_name = parts[0]
            try:
                index = int(parts[1])
            except ValueError:
                index = self._find_index_by_name(split_name, parts[1])
            return (split_name, index)
        else:
            return ("test", int(theorem_id))

    def _find_index_by_name(self, split: str, name: str) -> int:
        """根据定理名称查找索引"""
        test_index = self._load_split_index(split)
        for i, entry in enumerate(test_index):
            thm_path = entry.get("thm_path", "")
            if name in thm_path:
                return i
        return 0

    def _load_split_index(self, split: str) -> List[Dict[str, Any]]:
        """加载指定 split 的索引"""
        if split in self._index_cache:
            return self._index_cache[split]

        index_file = os.path.join(self.coqstoq_path, f"{split}-theorems.json")
        if not os.path.exists(index_file):
            return []

        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
            self._index_cache[split] = index
            return index

    def _load_theorem_definition(self, split: str, index: int) -> Optional[Dict[str, Any]]:
        """加载定理定义

        Args:
            split: "test", "val", or "cutoff"
            index: 定理索引（从 0 开始）

        Returns:
            定理定义字典或 None
        """
        split_index = self._load_split_index(split)

        if index >= len(split_index):
            return None

        entry = split_index[index]
        thm_path = entry.get("thm_path", "")
        thm_idx = entry.get("thm_idx", 0)

        thm_file = os.path.join(self.coqstoq_path, thm_path)

        if not os.path.exists(thm_file):
            return None

        with open(thm_file, "r", encoding="utf-8") as f:
            theorems = json.load(f)

        if thm_idx >= len(theorems):
            return None

        return theorems[thm_idx]

    def _extract_theorem_statement(self, repo_path: str, theorem_def: Dict[str, Any]) -> str:
        """从源文件提取定理陈述"""
        file_path = os.path.join(repo_path, theorem_def["path"])

        if not os.path.exists(file_path):
            return "无法读取源文件"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            start_line = theorem_def["theorem_start_pos"]["line"]
            end_line = theorem_def["theorem_end_pos"]["line"]

            statement = "".join(lines[start_line:end_line + 1])
            statement = statement.strip()

            if len(statement) > 300:
                statement = statement[:300] + "..."

            return statement
        except Exception:
            return "无法提取定理陈述"

    def verify_proof(
        self,
        theorem_id: str,
        proof_content: str,
        injected_prelude: str = "",
    ) -> Dict[str, Any]:
        """
        验证证明（替换原证明并验证）

        Args:
            theorem_id: 定理ID，格式如 "test:39" 或 "val:100"
            proof_content: 证明内容

        Returns:
            {
                "success": bool,
                "state": str,  # "proven", "in_progress", "failed", "error"
                "error_message": str | None,
                "theorem_info": dict,
                "proof_content": str,
                "proof_status": str,
                "verification_method": str  # "coqc" | "syntax"
            }
        """
        # 解析定理ID
        split_name, index = self._parse_theorem_id(theorem_id)

        # 获取定理信息
        theorem_def = self._load_theorem_definition(split_name, index)

        if theorem_def is None:
            return {
                "success": False,
                "state": "error",
                "error_message": f"定理 '{theorem_id}' 不存在",
                "theorem_info": None,
                "proof_content": proof_content,
                "proof_status": "定理不存在",
                "verification_method": "syntax"
            }

        # 构建文件路径
        repo_path = os.path.join(
            self.coqstoq_path,
            theorem_def["project"]["split"]["dir_name"],
            theorem_def["project"]["dir_name"]
        )
        full_file_path = os.path.join(repo_path, theorem_def["path"])

        if not os.path.exists(full_file_path):
            return {
                "success": False,
                "state": "error",
                "error_message": f"源文件不存在: {full_file_path}",
                "theorem_info": None,
                "proof_content": proof_content,
                "proof_status": "源文件不存在",
                "verification_method": "syntax"
            }

        # 读取原始源文件
        try:
            with open(full_file_path, "r", encoding="utf-8") as f:
                original_lines = f.readlines()
        except Exception as e:
            return {
                "success": False,
                "state": "error",
                "error_message": f"读取源文件失败: {e}",
                "theorem_info": None,
                "proof_content": proof_content,
                "proof_status": "读取失败",
                "verification_method": "syntax"
            }

        # Step 1: 首先编译原文件，确保原证明能编译通过
        original_compile_result = self._compile_original_file(repo_path, theorem_def, full_file_path)
        if not original_compile_result["success"]:
            return {
                "success": False,
                "state": "error",
                "error_message": f"原文件编译失败: {original_compile_result['error']}",
                "theorem_info": None,
                "proof_content": proof_content,
                "proof_status": "原文件编译失败",
                "verification_method": "coqc",
                "original_compile_error": original_compile_result["error"]
            }

        # 创建临时文件
        safe_id = theorem_id.replace(":", "_")
        temp_filename = f"coq_verify_{safe_id}.v"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)

        try:
            # Step 2: 使用精确的 column 位置构造新文件（参考 check.py 的 get_check_contents）
            new_content = self._construct_new_file(
                original_lines,
                theorem_def,
                proof_content,
                injected_prelude=injected_prelude,
            )
            
            with open(temp_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(new_content)

            # Step 3: 尝试使用 coqc 编译验证
            return self._verify_with_coqc(repo_path, theorem_def, proof_content, temp_path, theorem_id)

        except Exception as e:
            return {
                "success": False,
                "state": "error",
                "error_message": f"验证错误: {e}",
                "theorem_info": None,
                "proof_content": proof_content,
                "proof_status": "验证错误",
                "verification_method": "syntax"
            }

    def _verify_with_coqc(self, repo_path: str, theorem_def: Dict[str, Any],
                         proof_content: str, temp_path: str, theorem_id: str) -> Dict[str, Any]:
        """使用 coqc 编译验证"""
        cmd = [self.coqc_path] + theorem_def["project"].get("compile_args", []) + [temp_path]

        env = os.environ.copy()
        coqc_dir = str(Path(self.coqc_path).resolve().parent) if self.coqc_path != "coqc" else ""
        if coqc_dir and coqc_dir not in env.get("PATH", ""):
            env["PATH"] = coqc_dir + ":" + env.get("PATH", "")

        try:
            result_proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
                cwd=repo_path
            )

            stdout = result_proc.stdout
            stderr = result_proc.stderr
            combined = stdout + stderr

            # 分析结果
            is_success = result_proc.returncode == 0

            if is_success:
                state = "proven"
                status = "编译成功"
                proof_state = None
            else:
                # 检查证明是否包含 Qed
                has_qed = "Qed" in proof_content or "Qed." in proof_content
                if has_qed:
                    state = "failed"
                    status = "编译失败（证明有误）"
                    proof_state = None
                else:
                    state = "in_progress"
                    status = "编译失败（证明未完成）"
                    # 获取当前证明状态（内部调用 Show.）
                    proof_state = self._get_proof_state_with_show(
                        repo_path, theorem_def, theorem_id, proof_content
                    )

            return {
                "success": is_success,
                "state": state,
                "error_message": combined if not is_success else None,
                "theorem_info": {
                    "theorem_id": theorem_id,
                    "project": theorem_def["project"]["dir_name"],
                    "file": theorem_def["path"],
                    "statement": theorem_def.get("path", "")
                },
                "proof_content": proof_content,
                "proof_status": status,
                "verification_method": "coqc",
                "temp_file": temp_path,
                "compilation_output": combined,
                "compilation_error": combined if not is_success else None,
                "proof_state": proof_state
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "state": "failed",
                "error_message": "编译超时",
                "theorem_info": {
                    "theorem_id": theorem_id,
                    "project": theorem_def["project"]["dir_name"],
                    "file": theorem_def["path"],
                    "statement": theorem_def.get("path", "")
                },
                "proof_content": proof_content,
                "proof_status": "编译超时",
                "verification_method": "coqc",
                "temp_file": temp_path,
                "compilation_output": None,
                "compilation_error": "timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "state": "error",
                "error_message": f"编译错误: {e}",
                "theorem_info": {
                    "theorem_id": theorem_id,
                    "project": theorem_def["project"]["dir_name"],
                    "file": theorem_def["path"],
                    "statement": theorem_def.get("path", "")
                },
                "proof_content": proof_content,
                "proof_status": "编译错误",
                "verification_method": "coqc",
                "temp_file": temp_path,
                "compilation_output": None,
                "compilation_error": str(e)
            }

    def _compile_original_file(self, repo_path: str, theorem_def: Dict[str, Any], 
                                full_file_path: str) -> Dict[str, Any]:
        """
        编译原文件，确保原证明能编译通过
        
        参考 CoqStoq 的 compile_file 函数实现
        """
        compile_args = theorem_def["project"].get("compile_args", [])
        
        # 使用绝对路径
        full_path = Path(full_file_path).resolve()
        project_path = Path(repo_path).resolve()
        
        # 创建临时输出目录（避免在数据集目录写入）
        tmp_dir_path = tempfile.mkdtemp(prefix="tmp_verify_out_")
        tmp_dir = Path(tmp_dir_path)
        try:
            # 构建编译命令
            tmp_out_loc = tmp_dir / full_path.with_suffix(".vo").name
            cmd = [self.coqc_path, "-o", str(tmp_out_loc)] + compile_args + [str(full_path)]
            
            env = os.environ.copy()
            coqc_dir = str(Path(self.coqc_path).resolve().parent) if self.coqc_path != "coqc" else ""
            if coqc_dir and coqc_dir not in env.get("PATH", ""):
                env["PATH"] = coqc_dir + ":" + env.get("PATH", "")
            
            # 在 repo_path 目录下编译
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
                cwd=repo_path
            )
            
            if result.returncode == 0:
                return {"success": True, "error": None}
            else:
                return {"success": False, "error": result.stderr or result.stdout}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "原文件编译超时"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            # 清理临时目录
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def _construct_new_file(
        self,
        original_lines: List[str],
        theorem_def: Dict[str, Any],
        proof_content: str,
        injected_prelude: str = "",
    ) -> str:
        """
        使用精确的 column 位置构造新文件（参考 check.py 的 get_check_contents）
        
        构造方式：
        1. 定理声明及之前的内容（到 theorem_end_pos.column 为止）
        2. 新证明内容
        3. Qed. 
        4. 原证明之后的内容（从 proof_end_pos.column 开始）
        """
        theorem_end_pos = theorem_def.get("theorem_end_pos", {})
        proof_end_pos = theorem_def.get("proof_end_pos", {})
        
        theorem_start_pos = theorem_def.get("theorem_start_pos", {})
        theorem_start_line = theorem_start_pos.get("line", 0)
        theorem_end_line = theorem_end_pos.get("line", 0)
        theorem_end_column = theorem_end_pos.get("column", 0)
        proof_end_line = proof_end_pos.get("line", 0)
        proof_end_column = proof_end_pos.get("column", 0)

        # 构造前缀：可选注入引理后，再拼接原定理声明
        before_theorem = "".join(original_lines[:theorem_start_line])
        theorem_decl_lines = original_lines[theorem_start_line:theorem_end_line + 1].copy()
        if theorem_decl_lines:
            theorem_decl_lines[-1] = theorem_decl_lines[-1][:theorem_end_column]
        theorem_decl = "".join(theorem_decl_lines)

        injected = injected_prelude.strip()
        if injected:
            if before_theorem and not before_theorem.endswith("\n"):
                before_theorem += "\n"
            injected_block = injected + ("\n" if not injected.endswith("\n") else "")
            prefix_text = before_theorem + injected_block + theorem_decl
        else:
            prefix_text = before_theorem + theorem_decl
        
        # 构造后缀：原证明之后的内容
        suffix_lines = original_lines[proof_end_line:].copy()
        if suffix_lines:
            # 从第一行的 proof_end_pos.column 开始
            suffix_lines[0] = suffix_lines[0][proof_end_column:]
        
        # 准备证明内容（移除末尾的 Qed. 如果有的话，因为我们自己会添加）
        stripped_proof = proof_content.strip()
        if stripped_proof.endswith("Qed."):
            use_proof = stripped_proof[:-len("Qed.")].rstrip()
        else:
            use_proof = stripped_proof
        
        # 组合新文件内容
        # 注意：需要在证明前添加换行符，确保证明在新行开始
        new_content = ""
        new_content += prefix_text
        # 确保前缀以换行符结尾
        if prefix_text and not prefix_text.endswith("\n"):
            new_content += "\n"
        new_content += use_proof + "\n"
        new_content += "Qed."
        new_content += "".join(suffix_lines)

        return new_content

    def _extract_context_before_theorem(self, repo_path: str, theorem_def: Dict[str, Any]) -> str:
        """提取定理前的上下文。"""
        file_path = os.path.join(repo_path, theorem_def["path"])
        if not os.path.exists(file_path):
            return ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            theorem_start_line = theorem_def["theorem_start_pos"]["line"]
            return "".join(lines[:theorem_start_line])
        except Exception:
            return ""

    def _normalize_partial_proof(self, proof_content: str) -> str:
        """归一化未完成证明内容，移除 Proof./Qed. 包装。"""
        text = proof_content.strip()
        if text.startswith("Proof."):
            text = text[len("Proof."):].lstrip()
        if text.endswith("Qed."):
            text = text[:-len("Qed.")].rstrip()
        return text

    def _parse_show_state(self, raw: str) -> Optional[Dict[str, Any]]:
        """从 Show. 原始输出中提取 hypotheses/subgoals。"""
        lines = [ln.rstrip() for ln in raw.splitlines()]
        useful: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("Warning:", "File ", "Chars ")):
                continue
            if "Toplevel input" in s:
                continue
            if line.startswith("Coq < "):
                line = line[len("Coq < "):]
                if not line.strip():
                    continue
            useful.append(line)

        # 尝试定位 goals 段
        start = -1
        for i, line in enumerate(useful):
            ls = line.strip()
            if ls.endswith("goal") or ls.endswith("goals") or ls.startswith("subgoal"):
                start = i
                break
        if start == -1:
            # 找不到结构化目标就回退为原输出
            if useful:
                return {"subgoals": useful, "hypotheses": []}
            return None

        block = useful[start:]
        sep_idx = -1
        for i, line in enumerate(block):
            if "====" in line:
                sep_idx = i
                break

        if sep_idx == -1:
            return {"subgoals": block, "hypotheses": []}

        hyps = [x for x in block[:sep_idx] if x.strip() and "goal" not in x.strip()]
        goals = [x for x in block[sep_idx + 1:] if x.strip()]
        return {"subgoals": goals, "hypotheses": hyps}

    def _get_proof_state_with_show(
        self, repo_path: str, theorem_def: Dict[str, Any], theorem_id: str, proof_content: str
    ) -> Optional[Dict[str, Any]]:
        """
        in_progress 时内部执行 `Show.` 获取当前证明状态。
        """
        context = self._extract_context_before_theorem(repo_path, theorem_def)
        theorem_stmt = self._extract_theorem_statement(repo_path, theorem_def)
        partial_proof = self._normalize_partial_proof(proof_content)
        coq_script = f"""{context}

{theorem_stmt}
Proof.
{partial_proof}
Show.
"""

        compile_args = theorem_def["project"].get("compile_args", [])
        cmd = [self.coqtop_path] + compile_args + ["-batch", "-load-vernac-source"]

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{theorem_id.replace(':', '_')}_show.v", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(coq_script)
                tmp_path = tf.name

            run_cmd = cmd + [tmp_path]
            proc = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=60,
            )
            raw = (proc.stdout or "") + (proc.stderr or "")
            parsed = self._parse_show_state(raw)
            if parsed is None:
                return None
            parsed["raw_show_output"] = raw
            return parsed
        except Exception:
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def get_theorem_info(self, theorem_id: str) -> Optional[TheoremInfo]:
        """获取定理信息"""
        split_name, index = self._parse_theorem_id(theorem_id)

        theorem_def = self._load_theorem_definition(split_name, index)
        if theorem_def is None:
            return None

        repo_path = os.path.join(
            self.coqstoq_path,
            theorem_def["project"]["split"]["dir_name"],
            theorem_def["project"]["dir_name"]
        )

        statement = self._extract_theorem_statement(repo_path, theorem_def)
        compile_args = theorem_def["project"].get("compile_args", [])

        return TheoremInfo(
            theorem_id=theorem_id,
            project=theorem_def["project"]["dir_name"],
            file=theorem_def["path"],
            statement=statement,
            theorem_line=theorem_def["theorem_start_pos"]["line"],
            proof_start_line=theorem_def.get("proof_start_pos", {}).get("line"),
            proof_end_line=theorem_def.get("proof_end_pos", {}).get("line"),
            compile_args=compile_args
        )

    def list_theorems(self, split: str = "test", limit: int = 10) -> List[Dict[str, Any]]:
        """列出可用定理"""
        split_index = self._load_split_index(split)

        results = []
        for i, entry in enumerate(split_index[:limit]):
            thm_def = self._load_theorem_definition(split, i)
            if thm_def:
                repo_path = os.path.join(
                    self.coqstoq_path,
                    thm_def["project"]["split"]["dir_name"],
                    thm_def["project"]["dir_name"]
                )
                statement = self._extract_theorem_statement(repo_path, thm_def)
                results.append({
                    "index": i,
                    "theorem_id": f"{split}:{i}",
                    "thm_path": entry.get("thm_path", ""),
                    "thm_idx": entry.get("thm_idx", 0),
                    "project": thm_def["project"]["dir_name"],
                    "file": thm_def["path"],
                    "statement": statement
                })
        return results


# 便捷函数
def verify_proof(theorem_id: str, proof_content: str, injected_prelude: str = "") -> Dict[str, Any]:
    """
    验证证明（替换原证明并验证）

    Args:
        theorem_id: 定理ID，格式如 "test:39" 或 "val:100"
        proof_content: 证明内容

    Returns:
        验证结果字典
    """
    verifier = CoqProofVerifier()
    return verifier.verify_proof(theorem_id, proof_content, injected_prelude=injected_prelude)


def get_theorem_info(theorem_id: str) -> Optional[Dict[str, Any]]:
    """获取定理信息"""
    verifier = CoqProofVerifier()
    info = verifier.get_theorem_info(theorem_id)
    if info is None:
        return None
    return {
        "theorem_id": info.theorem_id,
        "project": info.project,
        "file": info.file,
        "statement": info.statement,
        "theorem_line": info.theorem_line,
        "proof_start_line": info.proof_start_line,
        "proof_end_line": info.proof_end_line
    }


def list_theorems(split: str = "test", limit: int = 10) -> List[Dict[str, Any]]:
    """列出可用定理"""
    verifier = CoqProofVerifier()
    return verifier.list_theorems(split, limit)


def format_verify_result(result: Dict[str, Any]) -> str:
    """
    Format verification result into a structured string for LLM consumption.
    
    Args:
        result: The dictionary returned by verify_proof()
        
    Returns:
        A formatted string with proof state and error information
    """
    state = result.get("state", "unknown")
    theorem_info = result.get("theorem_info") or {}
    proof_content = result.get("proof_content", "")
    error_message = result.get("error_message", "")
    proof_state = result.get("proof_state")
    
    # Build theorem identifier
    theorem_id = theorem_info.get("theorem_id", "unknown")
    project = theorem_info.get("project", "unknown")
    file_path = theorem_info.get("file", "unknown")
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"VERIFICATION RESULT: {state.upper()}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Theorem: {theorem_id} ({project}/{file_path})")
    lines.append("")
    
    if state == "proven":
        # PROVEN: Simple success message
        lines.append("Proof State: PROVEN")
        lines.append("")
        lines.append("Status: Proof verification successful")
        
    elif state == "in_progress":
        # IN_PROGRESS: Show proof content and current proof state
        lines.append("Proof Content:")
        lines.append("-" * 80)
        lines.append(proof_content)
        lines.append("-" * 80)
        lines.append("")
        
        if proof_state:
            lines.append("Current Proof State:")
            
            # Show hypotheses
            hypotheses = proof_state.get("hypotheses", [])
            if hypotheses:
                for hyp in hypotheses:
                    lines.append(f"  {hyp}")
            
            # Show separator and subgoals
            subgoals = proof_state.get("subgoals", [])
            if subgoals:
                lines.append("  " + "=" * 50)
                for i, goal in enumerate(subgoals, 1):
                    lines.append(f"  {goal}")
        else:
            lines.append("Current Proof State: (Unable to retrieve)")
        
        lines.append("")
        lines.append("Status: Proof incomplete - missing 'Qed.'")
        
    elif state == "failed":
        # FAILED: Show proof content and error
        lines.append("Proof State: FAILED")
        lines.append("")
        lines.append("Proof Content:")
        lines.append("-" * 80)
        lines.append(proof_content)
        lines.append("-" * 80)
        lines.append("")
        
        # Extract error from compilation output
        if error_message:
            lines.append("Error:")
            # Find the first "Error:" line
            for line in error_message.split('\n'):
                if 'Error:' in line:
                    lines.append(line.strip())
                    break
        
        lines.append("")
        lines.append("Status: Proof verification failed")
        
    elif state == "error":
        # ERROR: System/environment error
        lines.append("Proof State: ERROR")
        lines.append("")
        
        if error_message:
            lines.append("Error:")
            lines.append(error_message.strip())
        
        lines.append("")
        lines.append("Status: System error - check dependencies")
    
    lines.append("=" * 80)
    
    return '\n'.join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        theorem_id = sys.argv[1]
        proof_content = " ".join(sys.argv[2:])

        result = verify_proof(theorem_id, proof_content)
        print("=" * 60)
        print("验证结果:")
        print("=" * 60)
        print(f"状态: {result['state']}")
        print(f"描述: {result['proof_status']}")

        if result['error_message']:
            print(f"\n错误: {result['error_message'][:500]}")

        if result['theorem_info']:
            info = result['theorem_info']
            print(f"\n定理信息:")
            print(f"  ID: {info['theorem_id']}")
            print(f"  项目: {info['project']}")
            print(f"  文件: {info['file']}")
            print(f"  陈述: {info['statement']}")

        if result.get('temp_file'):
            print(f"\n临时文件: {result['temp_file']}")

        print("=" * 60)
