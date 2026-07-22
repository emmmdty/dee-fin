## 第一章事件抽取：自建中文金融事件集合

| Metric | Value |
|---|---|
| Events extracted | 677 |
| Source documents | 429 |
| Events / document | 1.58 |
| Distinct event types | 13 |
| Mean arguments / event | 5.02 (max 9) |
| Temporal anchor coverage | 30% |
| Subject coverage | 100% (478 distinct subjects) |
| 一致性闭包后的图边数量 | 20683 |
| Raw extractor candidate edges | 498 |
| Raw candidate edges dropped as ungrounded | 1 / 498 |

| Event type | 中文 | Count |
|---|---|---|
| Share buyback | 股份回购 | 95 |
| Share pledge | 质押 | 75 |
| Loss | 亏损 | 74 |
| Bid won | 中标 | 73 |
| Acquisition | 企业收购 | 65 |
| Shareholder sell-down | 股东减持 | 58 |
| Exec change | 高管变动 | 57 |
| Pledge release | 解除质押 | 54 |
| IPO | 公司上市 | 32 |
| Financing | 企业融资 | 29 |
| Shareholder buy-in | 股东增持 | 27 |
| Regulatory inquiry | 被约谈 | 21 |
| Bankruptcy | 企业破产 | 17 |

> 说明：当前关系边来自证据接地的启发式图构建器。图边数量是在一致性求解器补充时序传递边之后的结果；原始候选边统计发生在证据过滤和闭包之前。上游 SARGE 预测当前仍未稳定导出触发词 span，因此触发词证据字段需要后续补齐。
