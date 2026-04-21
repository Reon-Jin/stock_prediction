# 训练样本结构说明

本文档对应当前项目里的正式训练链路：

- 训练数据准备代码：`train/data.py`
- 字段分组定义：`datasets/pytorch_dataset.py`
- 训练样本构建：`jobs/build_training_samples.py`
- 时间切分与导出：`datasets/splitter.py`

当前项目已经不再使用随机行切分，正式训练/验证/测试集统一采用 `time_purged` 时间切分，并带 `gap_days` 隔离。

## 1. 当前训练样本的真实结构

`python -m train.run` 最终拿到的单样本由 `train.data.FastStockDataset.__getitem__()` 生成，结构如下：

```python
{
  "X_seq": torch.FloatTensor[20, 42],
  "X_tab": torch.FloatTensor[13],
  "X_event": torch.FloatTensor[256],
  "X_mkt": torch.FloatTensor[16],
  "X_company_ids": {
    "symbol_id": torch.LongTensor[],
    "industry_id": torch.LongTensor[],
    "board_id": torch.LongTensor[],
  },
  "X_company_profile": torch.FloatTensor[12],
  "neighbors": {
    "neighbor_symbol_ids": torch.LongTensor[10],
    "neighbor_scores": torch.FloatTensor[10],
  },
  "y": {
    "p_win": torch.FloatTensor[5],
    "ret_mu": torch.FloatTensor[5],
    "risk_dd": torch.FloatTensor[5],
    "rank_score": torch.FloatTensor[1],
  },
}
```

DataLoader 拼 batch 后的形状为：

```python
{
  "X_seq": [B, 20, 42],
  "X_tab": [B, 13],
  "X_event": [B, 256],
  "X_mkt": [B, 16],
  "X_company_ids": {
    "symbol_id": [B],
    "industry_id": [B],
    "board_id": [B],
  },
  "X_company_profile": [B, 12],
  "neighbors": {
    "neighbor_symbol_ids": [B, 10],
    "neighbor_scores": [B, 10],
  },
  "y": {
    "p_win": [B, 5],
    "ret_mu": [B, 5],
    "risk_dd": [B, 5],
    "rank_score": [B, 1],
  },
}
```

默认参数：

- `seq_length = 20`
- `neighbor_topk = 10`

## 2. parquet 导出和训练样本不是一回事

当前 `data/exports/*.parquet` 导出的是按 `symbol, trade_date` 排序后的宽表行，不是已经展开好的 `X_seq`。

训练时会再做两步：

1. 按 `symbol, trade_date` 排序后，为每只股票动态切 20 日窗口，生成 `X_seq`
2. 按日期读取市场特征和事件特征，组装 `X_event` / `X_mkt`

所以：

- `train.parquet / valid.parquet / test.parquet` 是样本行级宽表
- 真正的训练输入是在 `train/data.py` 中二次拼装的

## 3. 当前导出文件结构

当前正式导出文件为：

- `data/exports/train.parquet`
- `data/exports/valid.parquet`
- `data/exports/test.parquet`
- `data/exports/train_events.parquet`
- `data/exports/valid_events.parquet`
- `data/exports/test_events.parquet`
- `data/exports/split_summary.json`
- `data/exports/split_manifest.json`

其中：

- 主 parquet 保存股票级宽表样本
- `*_events.parquet` 保存按 `trade_date` 去重后的日级事件向量

### 3.1 为什么要单独导出 `*_events.parquet`

`event_embedding` 只和日期有关，同一天所有股票共享同一个向量。

为了避免在主样本表中重复存储 256 维事件向量，当前导出逻辑采用 sidecar 形式：

- 主表不再重复写 `event_embedding`
- sidecar 中每个 `trade_date` 只存一条 `event_embedding`
- 训练加载时按 `trade_date` 回连

这样能显著降低 parquet 体积，同时不改变模型的 `X_event` 输入结构。

## 4. 各输入块字段定义

### 4.1 `X_seq`

`X_seq` 对应 `DEFAULT_SEQ_COLUMNS`，当前共 42 个字段：

1. `open`
2. `high`
3. `low`
4. `close`
5. `adj_close`
6. `volume`
7. `amount`
8. `turnover_rate`
9. `pct_chg`
10. `ret_1`
11. `ret_3`
12. `ret_5`
13. `ret_10`
14. `ret_20`
15. `ma5_gap`
16. `ma10_gap`
17. `ma20_gap`
18. `ma60_gap`
19. `rsi_6`
20. `rsi_14`
21. `macd_dif`
22. `macd_dea`
23. `macd_hist`
24. `atr_14`
25. `boll_pos`
26. `volatility_5`
27. `volatility_20`
28. `vol_ratio_5`
29. `vol_ratio_20`
30. `intraday_range`
31. `candle_body`
32. `upper_shadow`
33. `lower_shadow`
34. `gap_open_to_prev_close`
35. `close_to_prev_close`
36. `high_to_prev_close`
37. `low_to_prev_close`
38. `amount_log1p`
39. `volume_log1p`
40. `turnover_rate_delta`
41. `ret_spread_5_20`
42. `volatility_ratio_5_20`

说明：

- 其中后 13 个是 `train/data.py::_augment_aligned_features()` 动态补出的派生特征
- 旧版资金流序列列已经从正式训练输入中移除

### 4.2 `X_tab`

`X_tab` 对应 `DEFAULT_TAB_COLUMNS`，当前共 13 个字段：

1. `list_days`
2. `turnover_rank_industry`
3. `amount_rank_market`
4. `ret5_vs_hs300`
5. `ret10_vs_hs300`
6. `ret20_vs_industry`
7. `stock_rank_in_industry`
8. `industry_rank_5d`
9. `industry_rank_20d`
10. `roe`
11. `revenue_yoy`
12. `profit_yoy`
13. `industry_roe_percentile`

说明：

- `pe_ttm`、`pb`、`industry_pe_percentile` 已从正式训练输入移除
- `ret20_vs_industry` 和行业 rank 现在直接基于股票表内行业分组计算

### 4.3 `X_event`

`X_event` 维度固定为 256。

当前正式路径中：

- 主 parquet 不保存逐行展开后的 `event_embedding_000 ~ event_embedding_255`
- sidecar `*_events.parquet` 中保存 `trade_date + event_embedding`
- 训练时在 `train/data.py` 中解析并标准化为 `[256]`

说明：

- 同一交易日的所有股票共享同一个 `X_event`
- 当前不再使用旧版股票级事件计数和事件打分列

### 4.4 `X_mkt`

`X_mkt` 对应 `DEFAULT_MKT_COLUMNS`，当前共 16 个字段：

1. `up_limit_count`
2. `down_limit_count`
3. `broken_limit_rate`
4. `consecutive_limit_height`
5. `market_turnover`
6. `hs300_ret_1`
7. `cyb_ret_1`
8. `market_volatility_5`
9. `sector_hotness_top1`
10. `sector_hotness_top3_mean`
11. `risk_on_flag`
12. `risk_off_flag`
13. `limit_count_spread`
14. `limit_count_ratio`
15. `sector_hotness_spread`
16. `market_regime_score`

说明：

- 后 4 个是 `train/data.py::_augment_aligned_features()` 动态补出的市场派生特征

### 4.5 `X_company_ids`

`X_company_ids` 共 3 个离散 ID：

1. `symbol_id`
2. `industry_id`
3. `board_id`

说明：

- 这些列当前用于 embedding 编码
- 不建议改成 one-hot

### 4.6 `X_company_profile`

`X_company_profile` 对应 `DEFAULT_COMPANY_PROFILE_COLUMNS`，当前共 12 个字段：

1. `market_cap_log`
2. `volatility_120`
3. `beta_120`
4. `turnover_mean_120`
5. `amount_mean_120`
6. `ret_20`
7. `ret_60`
8. `roe`
9. `revenue_yoy`
10. `profit_yoy`
11. `debt_ratio`
12. `gross_margin`

说明：

- `company_profiles` 表是按 `symbol + asof_date` 存储的时点画像，不是一家公司一行
- `amount_mean_120` 表示过去 120 个交易日的平均成交额
- `pe_ttm`、`pb`、事件频率列已从正式公司画像输入中移除

### 4.7 `neighbors`

- `neighbor_symbol_ids`: `[10]`
- `neighbor_scores`: `[10]`

如果源 parquet 缺少相似公司列，当前实现会自动补零。

## 5. 标签结构

标签分组来自 `DEFAULT_LABEL_GROUPS`：

- `p_win`: `label_win_3`, `label_win_5`, `label_win_10`, `label_win_20`, `label_win_40`
- `ret_mu`: `label_ret_3`, `label_ret_5`, `label_ret_10`, `label_ret_20`, `label_ret_40`
- `risk_dd`: `label_maxdd_3`, `label_maxdd_5`, `label_maxdd_10`, `label_maxdd_20`, `label_maxdd_40`
- `rank_score`: `label_rank_score`

对应维度：

- `p_win`: 5
- `ret_mu`: 5
- `risk_dd`: 5
- `rank_score`: 1

## 6. 标准化与数值处理

当前训练链路中，以下输入都会在 `FeatureNormalizer` 中做标准化：

- `X_seq`
- `X_tab`
- `X_event`
- `X_mkt`
- `X_company_profile`

说明：

- 训练时 normalizer 由 train split 拟合
- 推理时复用 checkpoint 中保存的 normalizer 状态
- 事件向量 `X_event` 现在也会一起标准化

## 7. 当前正式切分方案

当前正式方案是 `time_purged`，配置来自 `configs/config.yaml`：

```yaml
project:
  split_method: time_purged
  split:
    train_start: 2023-01-01
    train_end: 2024-12-31
    valid_start: 2025-03-01
    valid_end: 2025-06-30
    test_start: 2025-09-01
    test_end: 2025-12-31
    gap_days: 60
```

关键规则：

- 只允许按时间区间切分
- gap 区间数据不会导出到任何 split
- 同一合法交易日保留全市场横截面
- 严格校验 train/valid/test 间没有时间泄露

## 8. `inspect_sample.py` / `inspect_samples.py`

当前样本检查脚本已经改成直接复用训练真实拼装链路，而不是自己手写一套旧逻辑。

可用命令示例：

```bash
python scripts/inspect_sample.py --split train --index 0
python scripts/inspect_samples.py --split valid --symbol 000001.SZ --trade-date 2025-03-03
python scripts/inspect_sample.py --split train --view schema
python scripts/inspect_sample.py --split train --view raw --index 0
```

说明：

- `dataset` 视图查看的是训练真实会吃到的样本
- 会自动走 `build_or_load_split + FastStockDataset + FeatureNormalizer`
- 能正确处理 event sidecar、cache 和标准化

## 9. 当前和旧版实现的关键差异

以下内容已经不再属于正式训练输入：

- random row split
- 旧版股票级事件计数和事件打分类
- 旧版资金流派生特征列
- `pe_ttm`
- `pb`
- `industry_pe_percentile`
- 公司画像中的事件频率列

如果你在数据库里还看见这些旧列，多半是历史遗留结构，正式链路已经不再依赖它们。
