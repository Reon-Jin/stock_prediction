# A 股数据提取与训练样本构建项目

这是一个面向 A 股日频机器学习任务的数据管道项目。它的目标不是只做原始数据抓取，而是稳定地产出：

- 可复现的 MySQL 训练样本库
- 严格时间切分的 `train / valid / test` parquet
- 可直接送入 PyTorch 模型的训练样本结构
- 面向当日推理的 `today` 预测样本

当前正式评估方案已经统一切换为 `time_purged` 时间切分，不再使用 random row split。

## 1. 项目产物

项目当前有两类主要产物：

- 训练数据
  - MySQL 表：`training_samples`
  - 导出文件：`data/exports/train.parquet`、`valid.parquet`、`test.parquet`
  - 事件 sidecar：`train_events.parquet`、`valid_events.parquet`、`test_events.parquet`
- 预测数据
  - `data/today*.parquet`
  - `data/today_infer*.parquet`
  - `data/predictions/` 下的归档副本

## 2. 当前正式训练方案

### 2.1 切分策略

当前正式切分为 `Purged Time Series Split`：

- 只允许按时间切分
- train / valid / test 各自使用独立时间区间
- split 边界之间保留 `gap_days`
- 默认 `gap_days = seq_length + max_horizon = 60`
- gap 区间数据不会进入任何导出 split

默认配置见 [config.yaml](/D:/python_object/股市数据提取预处理/configs/config.yaml)。

### 2.2 事件存储策略

当前事件特征是按日期共享的日级向量，不再为每只股票重复存储：

- 主样本 parquet 不重复写 `event_embedding`
- `*_events.parquet` 按 `trade_date` 去重保存事件向量
- 训练时按 `trade_date` 回连，组装成 `X_event`

### 2.3 当前模型输入维度

当前正式输入维度为：

- `X_seq`: `[20, 42]`
- `X_tab`: `[13]`
- `X_event`: `[256]`
- `X_mkt`: `[16]`
- `X_company_ids`: `3`
- `X_company_profile`: `[12]`
- `neighbors`: top-k `10`

更详细的字段定义见 [training_sample_schema.md](/D:/python_object/股市数据提取预处理/docs/training_sample_schema.md)。

## 3. 项目结构

```text
configs/
  config.yaml

data/
  exports/
  predictions/
  parquet/
  raw_cache/

datasets/
  pytorch_dataset.py
  splitter.py
  check_split.py

docs/
  training_sample_schema.md

features/
  feature_builder.py
  event_features.py
  price_features.py
  volume_features.py
  relative_features.py
  fundamental_features.py
  market_features.py
  company_profile_builder.py
  company_similarity_builder.py
  sample_finalize.py

jobs/
  sync_securities.py
  sync_daily_bars.py
  sync_index_bars.py
  sync_sector_daily.py
  sync_capital_flow.py
  sync_financial_snapshot.py
  sync_news.py
  sync_announcements.py
  build_event_features.py
  build_company_profiles.py
  build_company_similarity.py
  build_training_samples.py
  build_prediction_samples.py

scripts/
  init_db.py
  run_full_pipeline.py
  inspect_sample.py
  inspect_samples.py
  reset_training_data.py

train/
  data.py
  model.py
  predict.py
  run.py

warehouse/
  db.py
  models.py
  repository.py
  schema_init.py
```

## 4. 核心数据表

项目会使用或构建以下核心表：

- `securities`
- `trade_calendar`
- `daily_bars`
- `index_bars`
- `sector_daily`
- `capital_flow_daily`
- `financial_snapshot`
- `news_raw`
- `news_norm`
- `announcement_raw`
- `announcement_norm`
- `event_features_daily`
- `company_profiles`
- `company_similarity`
- `training_samples`
- `job_runs`

其中比较关键的是：

- `event_features_daily`
  - 当前只保存 `trade_date + event_embedding`
- `company_profiles`
  - 保存按 `symbol + asof_date` 的时点公司画像
  - 不是一家公司一行
- `training_samples`
  - 保存最终训练样本宽表

## 5. 环境准备

建议使用 Python 3.10+。

示例：

```bash
conda create -n dataproccess python=3.12 -y
conda activate dataproccess
pip install -r requirements.txt
```

## 6. 数据库准备

默认数据库连接配置在 [config.yaml](/D:/python_object/股市数据提取预处理/configs/config.yaml)：

```yaml
database:
  url: mysql+pymysql://root:185258@127.0.0.1:3306/a_share_ml?charset=utf8mb4
```

初始化 schema：

```bash
python -m warehouse.schema_init
```

或：

```bash
python scripts/init_db.py
```

## 7. 推荐使用方式

### 7.1 全流程构建

统一入口是 [run_full_pipeline.py](/D:/python_object/股市数据提取预处理/scripts/run_full_pipeline.py)：

```bash
python scripts/run_full_pipeline.py --start 2023-01-01 --end 2025-12-31
```

标准流程会依次执行：

1. 初始化 schema
2. 同步证券主数据和交易日历
3. 同步个股行情、指数行情、板块行情
4. 同步资金流和财务快照
5. 同步新闻和公告
6. 构建日级事件向量
7. 构建公司画像
8. 构建公司相似度
9. 构建训练样本
10. 导出 `train / valid / test` parquet

说明：

- 当前资金流数据仍会同步入库，但正式训练特征不再使用旧版资金流派生列
- `pe_ttm`、`pb` 会保留在原始财务快照表里，但不会进入正式训练输入

### 7.2 today 预测模式

全市场：

```bash
python scripts/run_full_pipeline.py --today
```

指定日期：

```bash
python scripts/run_full_pipeline.py --today --end 2026-04-02
```

单股票：

```bash
python scripts/run_full_pipeline.py --today 000001.SZ
```

说明：

- today 模式只构建预测样本，不构建训练标签
- 单股票 today 会尽量复用已有市场上下文和相似度
- today 模式下公告同步会跳过，因为当前日级事件向量只依赖新闻

## 8. 单独执行关键任务

### 8.1 初始化数据库结构

```bash
python -m warehouse.schema_init
```

### 8.2 重建公司画像

```bash
python -m jobs.build_company_profiles --config configs/config.yaml
```

### 8.3 重建公司相似度

```bash
python -m jobs.build_company_similarity --config configs/config.yaml
```

### 8.4 构建并导出训练样本

```bash
python -m jobs.build_training_samples --config configs/config.yaml --export
```

只导出现有 `training_samples`：

```bash
python -m jobs.build_training_samples --config configs/config.yaml --export-only
```

### 8.5 校验时间切分和泄露风险

```bash
python -m datasets.check_split --config configs/config.yaml --export-dir data/exports
```

### 8.6 清理旧训练数据

```bash
python scripts/reset_training_data.py --config configs/config.yaml
```

## 9. inspect_sample / inspect_samples

当前样本检查脚本已经接入真实训练拼装链路，不再依赖旧版 `MultiInputTrainingDataset` 的单独拼装逻辑。

常用示例：

```bash
python scripts/inspect_sample.py --split train --index 0
python scripts/inspect_samples.py --split valid --symbol 000001.SZ --trade-date 2025-03-03
python scripts/inspect_sample.py --split train --view raw --index 0
python scripts/inspect_sample.py --split train --view schema
```

它适合检查：

- parquet 原始行
- 训练真实会读到的样本结构
- event sidecar 是否正确生效
- 当前 split 的输入维度和标签维度

## 10. 当前实现里已经废弃或不再推荐的内容

以下内容已经不属于正式训练方案：

- `random` split
- 旧版股票级事件计数和事件打分列
- 旧版资金流派生特征列
- `pe_ttm`
- `pb`
- `industry_pe_percentile`
- 公司画像中的事件频率列

如果看到旧数据库里还有历史遗留列，不代表当前训练链路还在使用它们。

## 11. 常见说明

### 11.1 为什么 `company_profiles` 一家公司有很多行

因为它是时点画像表，不是静态主数据表。

- 唯一键是 `symbol + asof_date`
- `market_cap_log`、`volatility_120`、`beta_120`、`turnover_mean_120`、`amount_mean_120` 都是时变的
- 财务字段会按“截至当日可见的最近快照”向前对齐

### 11.2 `amount_mean_120` 是什么

表示过去 120 个交易日的平均成交额，是公司画像中的一个慢变量，用来描述长期交易活跃度和资金容量。

## 12. 相关文档

- [训练样本结构说明](/D:/python_object/股市数据提取预处理/docs/training_sample_schema.md)
- [当前模型架构说明](/D:/python_object/股市数据提取预处理/docs/model_architecture.md)
