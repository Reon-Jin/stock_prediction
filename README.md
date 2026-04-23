# A 股智能分析系统用户使用手册

本项目不是一个“只负责构建数据集”的脚本集合，而是一套围绕 A 股量化分析构建的完整系统，包含：

- 数据采集与清洗
- 特征工程与训练样本构建
- 模型训练与批量推理
- 单股分析与全市场选股 Web 应用
- 分析结果、模型产物、预测文件的统一落盘

如果你是第一次接触这个仓库，建议按下面的顺序阅读和使用：

1. 先看“系统能做什么”
2. 再看“系统结构”
3. 然后按“部署”和“启动”完成环境准备
4. 最后按“典型使用流程”运行数据、训练模型、启动 Web 端

---

## 1. 系统概览

这套系统面向 A 股股票分析场景，核心目标是把“原始行情/新闻/财务数据”逐步加工为“可训练、可预测、可解释、可在前端交互”的分析结果。

### 1.1 系统主要能力

当前仓库已经具备以下核心能力：

- **市场数据同步**
  - 证券基础信息
  - 交易日历
  - 个股日线行情
  - 指数行情
  - 板块行情
  - 资金流
  - 财务快照
  - 新闻
  - 公告

- **特征与样本构建**
  - 构建价格、成交量、相对强弱、财务、市场环境等特征
  - 构建事件向量 `event_features_daily`
  - 构建公司画像 `company_profiles`
  - 构建公司相似度 `company_similarity`
  - 生成 `training_samples` 训练样本表
  - 导出 `train / valid / test` Parquet 文件

- **模型训练**
  - 读取导出的训练集进行 PyTorch 训练
  - 自动生成 `best_model.pt`、训练曲线、指标 JSON、历史 CSV
  - 支持验证集和测试集评估

- **模型推理**
  - 使用训练好的模型对 `today` 样本或打包后的推理样本进行预测
  - 输出概率、收益、回撤、排序分数等字段
  - 生成 Parquet / CSV 预测结果

- **Web 分析平台**
  - 用户注册 / 登录
  - 首页总览
  - 单股分析
  - 全市场推荐 / 快速推荐
  - 历史会话记录
  - AI 对话式追问分析

### 1.2 这套系统适合谁用

- 想维护 A 股机器学习训练数据的开发者
- 想训练自己的选股模型的研究人员
- 想把预测结果通过 Web 页面提供给终端用户的产品开发者
- 想快速做“单只股票分析”或“全市场候选股推荐”的内部使用者

---

## 2. 系统整体结构

可以把本项目理解为 4 层：

### 2.1 数据层

负责从外部数据源拉取原始数据，并落到 MySQL。

对应目录：

- `jobs/`：各类同步任务
- `providers/`：数据提供方适配
- `warehouse/`：数据库模型、连接、仓储访问

### 2.2 特征与样本层

负责把原始数据变成模型可消费的特征和样本。

对应目录：

- `features/`：特征构建逻辑
- `labels/`：标签生成逻辑
- `datasets/`：切分、检查、PyTorch 数据集封装

### 2.3 模型层

负责训练、评估、推理。

对应目录：

- `train/model.py`：模型定义
- `train/data.py`：训练数据加载与归一化
- `train/run.py`：训练入口
- `train/predict.py`：推理入口
- `decision/`：决策与排序逻辑

### 2.4 应用层

负责把结果提供给最终用户。

对应目录：

- `webapp/`：FastAPI 后端
- `frontend/`：React + Vite 前端

---

## 3. 目录说明

下面是你最常用的目录：

```text
configs/            系统配置
data/               原始缓存、Parquet、中间导出、预测结果
datasets/           训练/推理数据集封装与切分校验
decision/           选股打分与决策逻辑
docs/               补充文档
features/           特征工程
frontend/           前端页面（React + Vite）
jobs/               数据同步和样本构建任务
models/             模型相关辅助代码
providers/          外部数据源适配
scripts/            常用脚本入口
train/              训练、评估、推理、模型产物
utils/              配置、日志、工具函数
warehouse/          数据库模型与仓储访问
webapp/             FastAPI 后端服务
```

### 3.1 最重要的入口文件

- `configs/config.yaml`
  - 主配置文件，数据库地址、数据源、切分参数、事件配置都在这里

- `scripts/run_full_pipeline.py`
  - 全量数据管道入口
  - 用于同步数据、构建特征、导出训练集或生成当日预测样本

- `train/run.py`
  - 模型训练入口

- `train/predict.py`
  - 模型推理入口

- `webapp/main.py`
  - FastAPI 后端启动入口

- `frontend/package.json`
  - 前端启动与构建入口

---

## 4. 功能模块说明

## 4.1 数据同步模块

负责把外部数据写入 MySQL，对应 `jobs/` 中的任务，例如：

- `sync_securities.py`：同步证券列表与基础信息
- `sync_daily_bars.py`：同步个股日线
- `sync_index_bars.py`：同步指数行情
- `sync_sector_daily.py`：同步板块行情
- `sync_capital_flow.py`：同步资金流
- `sync_financial_snapshot.py`：同步财务快照
- `sync_news.py`：同步新闻
- `sync_announcements.py`：同步公告

这些任务通常不需要单独逐个调用，推荐统一走 `scripts/run_full_pipeline.py`。

## 4.2 特征工程模块

`features/` 负责从同步后的数据中提取模型需要的输入：

- `price_features.py`：价格类特征
- `volume_features.py`：量能类特征
- `relative_features.py`：相对强弱 / 相对位置
- `fundamental_features.py`：财务类特征
- `market_features.py`：市场环境特征
- `event_features.py`：新闻 / 公告事件特征
- `company_profile_builder.py`：公司画像
- `company_similarity_builder.py`：公司相似度

## 4.3 训练样本模块

训练样本最终进入 `training_samples` 表，并导出为：

- `data/exports/train.parquet`
- `data/exports/valid.parquet`
- `data/exports/test.parquet`

同时还会有事件 sidecar：

- `train_events.parquet`
- `valid_events.parquet`
- `test_events.parquet`

当前正式切分方式是 **time_purged 时间切分**，不是随机切分。默认配置：

- `train`: `2023-01-01 ~ 2024-12-31`
- `valid`: `2025-03-01 ~ 2025-06-30`
- `test`: `2025-09-01 ~ 2025-12-31`
- `gap_days`: `60`

## 4.4 模型训练模块

`train/run.py` 会：

- 读取训练 / 验证 / 测试 Parquet
- 构建多输入模型
- 执行训练、验证、早停
- 保存最优模型
- 生成训练曲线和指标文件

训练产物默认落在：

- `train/artifacts/run_时间戳/`

典型文件包括：

- `best_model.pt`
- `history.csv`
- `history.json`
- `test_metrics.json`
- `training_curves.png`
- `business_curves.png`
- `topk_metrics.png`

如果你要给 Web 端使用模型，通常需要确保有一个可用的最佳模型：

- `train/artifacts/best/best_model.pt`

## 4.5 模型推理模块

`train/predict.py` 支持对预测样本进行推理，输出字段包含：

- `p_win_prob_*`：不同持有周期下的胜率概率
- `ret_mu_pred_*`：预期收益
- `risk_dd_pred_*`：风险 / 回撤
- `decision_score_*`：决策分数
- `rank_score_pred`：排序分数
- `signal_score`：最终信号分数

## 4.6 Web 应用模块

当前 Web 端分为前后端：

- **后端**：FastAPI，提供认证、分析、推荐、历史会话等接口
- **前端**：React + Vite，提供用户界面

已接入页面能力：

- 用户注册 / 登录
- 首页总览
- 单股分析
- 全市场推荐
- 快速推荐
- 历史分析会话查看和删除
- 分析后继续追问

---

## 5. 运行前准备

## 5.1 软件要求

建议环境：

- Python `3.10+`
- MySQL `8.x`
- Node.js `18+`
- npm `9+`

如果需要 GPU 训练，还需要：

- CUDA 环境
- 可被 PyTorch 正确识别的显卡驱动

## 5.2 安装 Python 依赖

推荐用 Conda 或 venv。

### Conda 示例

```bash
conda create -n stock_prediction python=3.12 -y
conda activate stock_prediction
pip install -r requirements.txt
```

### venv 示例

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 5.3 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

---

## 6. 配置说明

主配置文件为：

- [configs/config.yaml](/D:/python_object/stock_prediction/configs/config.yaml)

### 6.1 必改项

至少需要确认以下内容：

#### 数据库连接

```yaml
database:
  url: mysql+pymysql://root:password@127.0.0.1:3306/a_share_ml?charset=utf8mb4
```

请改成你自己的 MySQL 用户名、密码、库名。

#### 数据源配置

`providers` 段控制行情、财务、新闻等来源。

#### LLM 配置

`llm` 段用于 Web 分析时的大模型总结与对话能力。

> 建议不要把真实数据库密码和 API Key 直接提交到代码仓库，生产环境最好改成环境变量或外部密钥管理方案。

### 6.2 关键参数

#### 项目参数

- `project.default_start` / `default_end`：默认数据时间范围
- `project.seq_length`：序列长度，当前默认 `20`
- `project.split_method`：当前为 `time_purged`

#### 切分参数

- `train_start` / `train_end`
- `valid_start` / `valid_end`
- `test_start` / `test_end`
- `gap_days`

#### 标签参数

- `holding_periods: [3, 5, 10, 20, 40]`
- `benchmark_index`
- `bigloss_threshold`

---

## 7. 数据库初始化

第一次使用时，先初始化数据库表结构。

### 方式 1

```bash
python -m warehouse.schema_init
```

### 方式 2

```bash
python scripts/init_db.py
```

完成后，系统会创建训练、预测、用户、分析会话等所需表。

---

## 8. 部署与启动

## 8.1 仅部署数据与训练环境

如果你暂时不需要 Web 页面，只需要：

1. 安装 Python 依赖
2. 配置 MySQL
3. 初始化数据库
4. 运行全量数据管道
5. 训练模型

## 8.2 部署 Web 应用环境

如果你需要前端页面，需要同时启动：

1. MySQL
2. FastAPI 后端
3. React 前端

### 后端启动

在项目根目录执行：

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

启动后访问：

- 服务地址：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- 接口文档：[http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)

开发模式可用：

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000 --reload --reload-exclude frontend/node_modules --reload-exclude frontend/dist
```

### 前端启动

```bash
cd frontend
npm run dev
```

默认地址：

- [http://127.0.0.1:5173](http://127.0.0.1:5173)

当前前端通过 Vite 代理把 `/api` 请求转发到 `127.0.0.1:8000`，所以前后端本地联调时通常不需要额外改接口地址。

---

## 9. 系统启动顺序建议

第一次完整启动，建议按这个顺序：

1. 启动 MySQL
2. 检查并修改 `configs/config.yaml`
3. 初始化数据库
4. 运行全量数据管道，生成训练数据
5. 训练模型
6. 准备 `best_model.pt`
7. 启动 FastAPI 后端
8. 启动前端

如果只是日常使用 Web 页面：

1. 保证数据库可连接
2. 保证已有模型文件
3. 启动后端
4. 启动前端

---

## 10. 典型使用流程

## 10.1 全量构建训练数据

统一入口：

- [scripts/run_full_pipeline.py](/D:/python_object/stock_prediction/scripts/run_full_pipeline.py)

示例：

```bash
python scripts/run_full_pipeline.py --start 2023-01-01 --end 2025-12-31
```

这个流程通常会执行：

1. 初始化 schema
2. 同步证券与交易日历
3. 同步行情、指数、板块
4. 同步资金流与财务
5. 同步新闻与公告
6. 构建事件特征
7. 构建公司画像
8. 构建公司相似度
9. 构建训练样本
10. 导出 `train / valid / test` Parquet

## 10.2 构建当日预测样本

### 全市场 today 样本

```bash
python scripts/run_full_pipeline.py --today
```

### 指定日期

```bash
python scripts/run_full_pipeline.py --today --end 2026-04-02
```

### 单只股票

```bash
python scripts/run_full_pipeline.py --today 000001.SZ
```

说明：

- `--today` 模式只构建预测样本，不构建训练标签
- 单股模式会尽量复用已有市场上下文和相似度数据
- today 模式下公告同步会跳过，当前事件向量主要依赖新闻

## 10.3 训练模型

最简单的训练方式：

```bash
python -m train.run
```

常见自定义参数示例：

```bash
python -m train.run --epochs 30 --batch-size 512 --device cuda
```

训练默认读取：

- `data/exports/train.parquet`
- `data/exports/valid.parquet`
- `data/exports/test.parquet`

训练结束后会在 `train/artifacts/run_*/` 下生成产物。

## 10.4 运行模型推理

```bash
python -m train.predict --checkpoint-path train/artifacts/best/best_model.pt --input-path data/today_infer.parquet
```

可选参数示例：

```bash
python -m train.predict --checkpoint-path train/artifacts/best/best_model.pt --input-path data/today_infer.parquet --output-path data/predictions/today_pred.parquet --device cpu
```

推理完成后会同时输出：

- Parquet 文件
- 同名 CSV 文件

## 10.5 启动 Web 平台并使用

### 第一步：启动后端

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

### 第二步：启动前端

```bash
cd frontend
npm run dev
```

### 第三步：打开浏览器

访问：

- [http://127.0.0.1:5173](http://127.0.0.1:5173)

### 第四步：注册并登录

系统支持：

- 用户注册
- 用户登录
- 登录态校验

### 第五步：使用功能

#### 首页总览

可查看：

- 最新交易日
- 覆盖股票数
- 当前启用模块
- 最新模型检查点

#### 单股分析

输入：

- 股票代码
- 风险偏好
- 是否持有
- 已持有天数

系统会：

1. 构建该股票可用样本
2. 调用模型推理
3. 结合决策引擎输出建议动作
4. 生成 AI 分析内容
5. 支持继续追问

#### 全市场推荐

输入：

- 推荐数量 `top_n`
- 推荐模式

推荐模式：

- `market`：全市场批量推荐
- `quick`：快速抽样推荐

系统会：

1. 准备候选股票池
2. 批量推理
3. 使用决策逻辑排序
4. 返回推荐列表
5. 支持继续追问为什么推荐这些股票

---

## 11. 常用命令清单

## 11.1 数据相关

初始化数据库：

```bash
python -m warehouse.schema_init
```

全量跑数据管道：

```bash
python scripts/run_full_pipeline.py --start 2023-01-01 --end 2025-12-31
```

重建公司画像：

```bash
python -m jobs.build_company_profiles --config configs/config.yaml
```

重建公司相似度：

```bash
python -m jobs.build_company_similarity --config configs/config.yaml
```

构建并导出训练样本：

```bash
python -m jobs.build_training_samples --config configs/config.yaml --export
```

只导出已有 `training_samples`：

```bash
python -m jobs.build_training_samples --config configs/config.yaml --export-only
```

校验时间切分：

```bash
python -m datasets.check_split --config configs/config.yaml --export-dir data/exports
```

清理旧训练数据：

```bash
python scripts/reset_training_data.py --config configs/config.yaml
```

## 11.2 样本检查相关

查看单个样本：

```bash
python scripts/inspect_sample.py --split train --index 0
```

按股票和日期检查样本：

```bash
python scripts/inspect_samples.py --split valid --symbol 000001.SZ --trade-date 2025-03-03
```

查看原始视图：

```bash
python scripts/inspect_sample.py --split train --view raw --index 0
```

查看 schema：

```bash
python scripts/inspect_sample.py --split train --view schema
```

## 11.3 模型相关

训练：

```bash
python -m train.run
```

推理：

```bash
python -m train.predict --checkpoint-path train/artifacts/best/best_model.pt --input-path data/today_infer.parquet
```

## 11.4 Web 相关

启动后端：

```bash
uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

启动前端：

```bash
cd frontend
npm run dev
```

---

## 12. 核心数据与产物位置

## 12.1 数据库中的关键表

系统会使用或生成以下关键表：

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

Web 端还会使用用户和会话相关表。

## 12.2 文件系统中的关键产物

### 数据产物

- `data/raw_cache/`：原始缓存
- `data/parquet/`：中间 Parquet
- `data/exports/`：训练导出集
- `data/predictions/`：预测结果归档

### 模型产物

- `train/artifacts/run_*/`：每次训练的独立产物目录
- `train/artifacts/best/`：供 Web 或线上推理优先使用的最佳模型目录

---

## 13. 使用注意事项

## 13.1 Web 端依赖模型文件

Web 分析接口会优先尝试加载最新模型检查点。如果没有可用模型，部分分析能力可能退化或无法达到预期效果。

## 13.2 Web 端依赖数据库

后端启动时会初始化 schema，并在分析时直接访问数据库。如果数据库未启动、配置错误或数据不足，接口会报错。

## 13.3 today 模式不是训练模式

`--today` 只用于预测样本构建，不会生成训练标签，也不会替代完整训练流程。

## 13.4 前后端都要启动

只启动 `frontend` 看不到完整功能；只启动 `webapp` 没有可视化页面。本地使用时一般要两个都开。

## 13.5 训练耗时与资源

- CPU 可以训练，但速度较慢
- GPU 更适合正式训练
- 全量市场数据同步和样本构建也会花费较长时间

---

## 14. 常见问题

## 14.1 启动前端后提示无法连接后端

请检查：

1. `uvicorn webapp.main:app --host 127.0.0.1 --port 8000` 是否已启动
2. 后端是否报数据库连接错误
3. 前端是否运行在 `5173`

## 14.2 后端能启动，但单股分析或推荐报错

通常是以下原因：

- 数据库里没有对应交易日的数据
- 还没有训练好模型
- `train/artifacts/best/best_model.pt` 不存在
- today 样本或上下文样本不完整

## 14.3 训练报找不到 Parquet

请先执行训练数据构建：

```bash
python scripts/run_full_pipeline.py --start 2023-01-01 --end 2025-12-31
```

或至少执行：

```bash
python -m jobs.build_training_samples --config configs/config.yaml --export
```

## 14.4 想只看数据，不启动 Web

完全可以。这个项目的底层能力本来就支持只做：

- 数据同步
- 样本构建
- 模型训练
- 批量推理

---

## 15. 相关文档

- [训练样本结构说明](/D:/python_object/stock_prediction/docs/training_sample_schema.md)
- [模型架构说明](/D:/python_object/stock_prediction/docs/model_architecture.md)
- [Web App 启动说明](/D:/python_object/stock_prediction/docs/web_app.md)
- [Web 训练控制台说明](/D:/python_object/stock_prediction/docs/web_training_console.md)

---

## 16. 推荐上手路径

如果你是新用户，建议按下面这个最小闭环走一遍：

1. 修改 [configs/config.yaml](/D:/python_object/stock_prediction/configs/config.yaml) 的数据库和数据源配置
2. 初始化数据库
3. 执行 `python scripts/run_full_pipeline.py --start 2023-01-01 --end 2025-12-31`
4. 执行 `python -m train.run`
5. 准备 `train/artifacts/best/best_model.pt`
6. 启动后端和前端
7. 先试“单股分析”，再试“全市场推荐”

这样你就能把这套系统从“采数据”一路跑到“页面里看分析结果”。
