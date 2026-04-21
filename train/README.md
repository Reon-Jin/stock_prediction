# 训练流水线

该目录提供一个可直接运行的极简深度学习训练入口，面向 `data/exports/train.parquet`、`valid.parquet`、`test.parquet`。

模型结构：

- `X_seq` 用多尺度时序编码器：`TemporalConv + BiGRU + Self-Attention Pooling`
- `X_tab`、`X_event`、`X_mkt`、`X_company_profile` 用带残差门控的深层 `MLP` 编码
- `X_company_ids` 走 embedding
- `neighbors` 走相似公司 symbol embedding 的加权池化
- `event + market` 先融合成上下文向量，再对序列、截面、公司表示做门控调制
- 最后进入共享融合层和多任务专用头，输出 `p_win / ret_mu / risk_dd / rank_score`

数据侧优化：

- 首次运行时把 parquet 预处理为缓存张量，保存到 `train/cache/`
- `event_embedding` 按交易日去重解析，避免对整表重复解码 256 维新闻向量
- 训练阶段只做张量切片，不再反复做 pandas 切片与 JSON 解析

示例：

```bash
python -m train.run
```

当前默认参数已经按常见 `RTX 4060 Laptop GPU` 做了偏实战化调整：

- `batch_size` 默认会按显存自动取值，8GB 档默认 `384`
- `epochs=12`
- `lr=5e-4`
- `early_stop_patience=4`
- 分类头自动使用训练集正负样本比例估计的 `pos_weight`
- 多周期 horizon 会使用不同损失权重，重点照顾 `5/10/20` 日预测
- CUDA 下自动开启 AMP、`cudnn.benchmark` 和高精度 matmul

快速 smoke test：

```bash
python -m train.run --epochs 1 --max-train-batches 10 --max-eval-batches 3
```

预测入口：

```bash
python -m train.predict --checkpoint-path train/artifacts/你的实验目录/best_model.pt --input-path data/today_infer.parquet
```

也支持直接读取带历史上下文的原始 parquet：

```bash
python -m train.predict --checkpoint-path train/artifacts/你的实验目录/best_model.pt --input-path data/exports/test.parquet
```

为了避免大表测试过慢：

- raw parquet 默认只预测“最新交易日”的样本
- 可用 `--limit 64` 之类的小样本参数做 2 分钟内 smoke test
- 若确实要跑 raw 文件的全部可预测样本，再显式加 `--all-dates`

输出内容默认写到 `train/artifacts/run_时间戳/`：

- `config.json`
- `history.csv`
- `history.json`
- `training_curves.png`
- `best_model.pt`
- `test_metrics.json`

预测输出默认写到输入文件同目录，文件名形如 `*_pred.parquet` 和 `*_pred.csv`。
