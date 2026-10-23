# WWW 消融与 Full-candidate 汇总

## 审计边界

- 本目录是追加式汇总；未重跑或覆盖已有 87 项 source-validation 消融。
- 新结果是 validation-only 的 fixed-subset full-candidate filtered ranking，不是 locked test。
- GraIL/NBFNet 为 entity-inductive；ComplEx 仅为 transductive compatibility diagnostic。
- 每个事实同时评价 head/tail，conservative ties，CI 以原事实为 cluster 做 10,000 次 paired bootstrap。
- 三 seed 行报告 mean/sample std；它不是把三个 seed 当作大样本计算的置信区间。

## 三 seed 核心结果

| Backbone | Track | Base MRR | Stage 0 | Stage 1-only | Sequential | Sequential - Base |
|---|---|---:|---:|---:|---:|---:|
| GraIL | entity_inductive | 0.262499 | 0.279080 | 0.273467 | 0.287780 | +0.025282 |
| NBFNet | entity_inductive | 0.295588 | 0.310398 | 0.332224 | 0.347648 | +0.052060 |
| ComplEx | transductive_diagnostic | 0.161631 | 0.188285 | 0.172706 | 0.198284 | +0.036653 |

## 参数合同

- Stage 0: 85,185；Stage 1-only: 6,194；Sequential: 91,379；Flat-Matched: 91,378。
- Flat-Matched 与 Sequential 的相对参数差为 0.001094%，严格通过预注册的 5% 匹配门槛。

## 解释限制

这些结果回答的是：三个预冻结验证单元中，51-candidate 信号是否在全实体候选空间完全消失。它们不能替代完整官方查询、跨 split 或 one-shot locked test，也不能把 ComplEx 写成归纳模型。
