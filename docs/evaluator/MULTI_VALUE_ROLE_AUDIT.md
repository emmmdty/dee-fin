# 多值角色数据审计

本文档记录 `data/processed/` 中三个已处理金融事件抽取数据集的多值角色审计结果，用于说明为什么除历史 fixed-slot 指标外，还需要 `unified-strict` / canonical role-value 评价。

## 复现命令

从项目根目录运行：

```bash
python scripts/evaluator/audit_multi_value_roles.py --project-root .
python scripts/evaluator/compare_fixed_slot_vs_canonical_units.py --project-root .
```

上述复现命令只读取 `data/processed/` 与各数据集 `schema.json`，不修改 `data/`、`evaluator/`、`baseline/` 或任何模型/实验输出目录。若需要重新生成机器可读结果，显式运行：

```bash
python scripts/evaluator/audit_multi_value_roles.py --project-root . --write-json
```

该命令会更新 `docs/evaluator/multi_value_role_audit_results.json`。

## 计数定义

- `canonical raw role-value unit`：一个单位是 `(event_type, record, role, normalized value)`；同一角色可以贡献多个值。
- `canonical unique role-value unit`：在同一个事件记录、同一个角色内，对归一化后完全相同的值去重后再计数。
- `fixed-slot non-empty unit`：一个单位是非空 `(event_type, record, role_slot)`；一个 role slot 最多贡献一个参数。
- 多值角色 occurrence：同一事件记录中的同一 `(event_type, role)` 出现两个或更多不同的归一化非空值。
- `multi-value extra units`：每个多值角色的 `unique value count - 1`，即 fixed-slot 表示边界之外会被折叠的额外值数。

归一化仅使用确定性表面规范化：Unicode NFKC、去除首尾空白、折叠连续空白。不使用别名扩展、语义匹配、金额/日期推理或 LLM judge。

## 总体结果

| dataset | split | documents | event records | canonical raw units | canonical unique units | fixed-slot non-empty units | multi-value occurrences | docs with multi-value | records with multi-value | extra units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ChFinAnn-Doc2EDAG | train | 25632 | 38088 | 229848 | 229848 | 229848 | 0 | 0 | 0 | 0 |
| ChFinAnn-Doc2EDAG | dev | 3204 | 4987 | 30588 | 30588 | 30588 | 0 | 0 | 0 | 0 |
| ChFinAnn-Doc2EDAG | test | 3204 | 4749 | 29435 | 29435 | 29435 | 0 | 0 | 0 | 0 |
| DuEE-Fin-dev500 | train | 6515 | 8824 | 45425 | 45395 | 42969 | 1559 | 1258 | 1459 | 2426 |
| DuEE-Fin-dev500 | dev | 500 | 674 | 3466 | 3462 | 3296 | 119 | 91 | 107 | 166 |
| DuEE-Fin-dev500 | test | 1171 | 1533 | 7915 | 7907 | 7533 | 264 | 219 | 246 | 374 |
| DocFEE-dev1000 | train | 17244 | 32393 | 102718 | 102718 | not_applicable | 0 | 0 | 0 | 0 |
| DocFEE-dev1000 | dev | 1000 | 1846 | 5801 | 5801 | not_applicable | 0 | 0 | 0 | 0 |
| DocFEE-dev1000 | test | 800 | 1236 | 3973 | 3973 | not_applicable | 0 | 0 | 0 | 0 |

`DocFEE-dev1000` 的 `schema.json` 是 JSON Schema 风格的 `properties` 对象，不是历史 native fixed-slot evaluator 使用的有序角色槽 schema。因此本文对 DocFEE 的 fixed-slot 计数标为 `not_applicable`，不把对象属性顺序解释为 native fixed-slot 顺序。

## Fixed-Slot 与 Canonical 差异

| dataset | split | fixed-slot non-empty units | canonical unique units | canonical - fixed-slot |
|---|---:|---:|---:|---:|
| ChFinAnn-Doc2EDAG | train | 229848 | 229848 | 0 |
| ChFinAnn-Doc2EDAG | dev | 30588 | 30588 | 0 |
| ChFinAnn-Doc2EDAG | test | 29435 | 29435 | 0 |
| DuEE-Fin-dev500 | train | 42969 | 45395 | 2426 |
| DuEE-Fin-dev500 | dev | 3296 | 3462 | 166 |
| DuEE-Fin-dev500 | test | 7533 | 7907 | 374 |
| DocFEE-dev1000 | train | not_applicable | 102718 | not_applicable |
| DocFEE-dev1000 | dev | not_applicable | 5801 | not_applicable |
| DocFEE-dev1000 | test | not_applicable | 3973 | not_applicable |

DuEE-Fin-dev500 的 train/dev/test 均存在 canonical unique units 多于 fixed-slot non-empty units 的情况；test split 与此前审计一致：fixed-slot 非空 schema-role slots 为 7533，canonical unique role-value units 为 7907，额外多值单位为 374。

## 多值角色集中位置

ChFinAnn-Doc2EDAG 和 DocFEE-dev1000 的当前 processed splits 中没有检测到同一事件记录、同一角色下的多值角色。DuEE-Fin-dev500 中，多值角色主要集中在价格、投资方、公司/主体类角色。

### DuEE-Fin-dev500 train Top 5

| event_type | role | extra units |
|---|---|---:|
| 股份回购 | 每股交易价格 | 379 |
| 企业融资 | 投资方 | 339 |
| 股东减持 | 减持方 | 248 |
| 中标 | 中标公司 | 239 |
| 高管变动 | 高管职位 | 215 |

### DuEE-Fin-dev500 dev Top 5

| event_type | role | extra units |
|---|---|---:|
| 股份回购 | 每股交易价格 | 26 |
| 企业融资 | 投资方 | 25 |
| 中标 | 中标公司 | 19 |
| 质押 | 质押方 | 15 |
| 股东减持 | 减持方 | 15 |

### DuEE-Fin-dev500 test Top 5

| event_type | role | extra units |
|---|---|---:|
| 股份回购 | 每股交易价格 | 83 |
| 企业融资 | 投资方 | 50 |
| 质押 | 质押方 | 39 |
| 中标 | 中标公司 | 36 |
| 股东减持 | 减持方 | 27 |

## 具体例子

| dataset | split | document_id | event_type | role | values |
|---|---|---|---|---|---|
| DuEE-Fin-dev500 | train | 079cd8cfdb6013ced33623c26840bc15 | 被约谈 | 约谈机构 | 北京市人力社保局；市商务局；市委政法委；市市场监管局四部门 |
| DuEE-Fin-dev500 | train | e55ffdd8af2e77df58b3020c8ee5502a | 企业收购 | 被收购方 | Dark Sky；NextVR；Voysis |
| DuEE-Fin-dev500 | dev | 9fddaaf106878e78f789db03820d1425 | 质押 | 质押方 | 邵健伟；邵健锋 |
| DuEE-Fin-dev500 | dev | f4549686f316cca57d5a3e4cbd6344ea | 高管变动 | 高管职位 | 总经理；法定代表人 |
| DuEE-Fin-dev500 | test | 5259811fce42c3a4833e326fb8f847a9 | 中标 | 中标公司 | 华润；国药器械；国药控股；海王 |
| DuEE-Fin-dev500 | test | 23ba128e902745d21c41fdd6097fa18b | 股份回购 | 每股交易价格 | 3.57元；6.1297元 |

这些例子说明，真实中文金融公告数据中存在一个事件记录内同一角色对应多个真实取值的情况。fixed-slot 表示仍然适合复现历史 baseline，但它天然不能把一个 role slot 展开成多个 canonical role-value units。

## 结论

- Native Doc2EDAG/ProcNet fixed-slot 指标仍然必要，因为它们对应历史系统的训练、解码和报告协议。
- 对 ChFinAnn-Doc2EDAG，当前 processed splits 与其 ordered schema 在计数上没有出现多值角色差异。
- 对 DuEE-Fin-dev500，train/dev/test 均存在多值角色，fixed-slot non-empty units 小于 canonical unique units。
- 对 DocFEE-dev1000，当前 processed splits 没有检测到多值角色；但该数据集没有可直接解释为 native fixed-slot role order 的 schema，因此 fixed-slot 计数不应强行套用。
- 因此，跨 ChFinAnn、DuEE-Fin、DocFEE 的科学比较应另报 `unified-strict` canonical role-value 指标；历史 fixed-slot 指标与 `unified-strict` 指标必须分列呈现。
