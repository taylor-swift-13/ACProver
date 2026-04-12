# Standard Library Retrieval

标准库 experience 已经落在 `experience/stdlib/` 下。检索分两条链路：

- 自然语言检索：走 `semantic_explanation` 的向量索引
- metadata 检索：走 `experience/stdlib/metadata.db`

## 1. 重建索引

当你已经写完 `experience/stdlib/*/metadata.json`，只想从现有记录重建索引时，用：

```bash
python3 src/coqstoq_tools.py build-stdlib-from-existing
```

这会重建：

- `experience/stdlib/metadata_index.json`
- `experience/stdlib/metadata.db`
- `experience/stdlib/semantic_explanations.faiss`
- `experience/stdlib/semantic_explanations.json`

如果你是在插入某个模块时顺手重建，也可以直接：

```bash
python3 src/coqstoq_tools.py build-stdlib-index --module-path Coq.Lists.List
```

## 2. 自然语言检索

自然语言检索只基于 `semantic_explanation` 做向量召回和重排。

命令行：

```bash
python3 src/coqstoq_tools.py query-stdlib --description "append with empty list on the right" -k 5
```

常见例子：

```bash
python3 src/coqstoq_tools.py query-stdlib --description "list append right identity" -k 5
python3 src/coqstoq_tools.py query-stdlib --description "permutation lemmas on lists" -k 10
python3 src/coqstoq_tools.py query-stdlib --description "induction principle for nat" -k 10
```

返回字段里最常用的是：

- `record_id`
- `module_path`
- `item_kind`
- `item_name`
- `semantic_explanation`
- `detail_path`
- `reasoning_path`

## 3. Metadata SQL 检索

SQL 检索直接查 `experience/stdlib/metadata.db`。

命令行：

```bash
python3 src/coqstoq_tools.py query-stdlib-sql --sql "select record_id, module_path from records limit 10"
```

推荐查询：

```bash
python3 src/coqstoq_tools.py query-stdlib-sql --sql "select record_id, item_kind, item_name from records where module_path = 'Coq.Lists.List' limit 20"
python3 src/coqstoq_tools.py query-stdlib-sql --sql \"select record_id, item_kind, item_name from records where item_kind = 'definition' limit 20\"
python3 src/coqstoq_tools.py query-stdlib-sql --sql \"select record_id, module_path, item_name from records where item_name like 'app_%' limit 20\"
python3 src/coqstoq_tools.py query-stdlib-sql --sql \"select record_id, semantic_explanation from records where normalized_theorem_types_json like '%equality%' limit 20\"
```

当前 `records` 表的关键列有：

- `record_id`
- `project`
- `file_path`
- `module_path`
- `item_kind`
- `item_name`
- `semantic_explanation`
- `normalized_theorem_types_json`
- `context`
- `proof`
- `related_json`
- `detail_path`
- `reasoning_path`
- `metadata_json`

限制：

- 只允许 `SELECT`
- 不允许多条 SQL

## 4. 检索脚本

仓库里提供了一个简单脚本：

```bash
python3 scripts/query_experience.py --domain stdlib --description "append with empty list on the right" -k 5
python3 scripts/query_experience.py --domain stdlib --sql "select record_id, item_kind from records where module_path = 'Coq.Lists.List' limit 10"
```

也支持 `coqstoq`：

```bash
python3 scripts/query_experience.py --domain coqstoq --description "rewrite equality lemma" -k 5
```

脚本规则：

- `--description` 和 `--sql` 二选一
- `--domain` 默认是 `stdlib`

## 5. 建议用法

如果你知道想找的语义，但不知道条目名，先用自然语言检索。

如果你已经知道要按模块、条目类型、名字前缀、标签过滤，直接用 SQL。自然语言检索负责召回，SQL 检索负责精确筛选。
