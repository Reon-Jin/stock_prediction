# 当前模型架构说明

本文档描述当前项目默认训练入口 `python -m train.run` 所使用的模型架构，对应代码：

- 模型主体：[train/model.py](/D:/python_object/股市数据提取预处理/train/model.py)
- 公司编码器：[models/company_encoder.py](/D:/python_object/股市数据提取预处理/models/company_encoder.py)
- 默认训练超参：[train/run.py](/D:/python_object/股市数据提取预处理/train/run.py)

本文档中的维度默认基于当前正式训练样本结构：

- `X_seq`: `[B, 20, 42]`
- `X_tab`: `[B, 13]`
- `X_event`: `[B, 256]`
- `X_mkt`: `[B, 16]`
- `X_company_ids`:
  - `symbol_id`: `[B]`
  - `industry_id`: `[B]`
  - `board_id`: `[B]`
- `X_company_profile`: `[B, 12]`
- `neighbors.neighbor_symbol_ids`: `[B, 10]`
- `neighbors.neighbor_scores`: `[B, 10]`

默认模型超参来自 [train/run.py](/D:/python_object/股市数据提取预处理/train/run.py)：

- `seq_model_dim = 96`
- `seq_output_dim = 128`
- `seq_attn_heads = 4`
- `branch_hidden_dim = 192`
- `context_dim = 96`
- `company_dim = 48`
- `fusion_dim = 256`
- `task_hidden_dim = 128`
- `dropout = 0.2`

## 1. 总体结构

当前模型类名是 `ProfessionalFinancialModel`，训练时通过别名 `TinyMultiInputModel` 实例化。

整体可以分成 6 个部分：

1. 时序分支：`X_seq -> TemporalFinancialEncoder`
2. 截面分支：`X_tab -> DeepMLPEncoder`
3. 事件分支：`X_event -> DeepMLPEncoder`
4. 市场分支：`X_mkt -> DeepMLPEncoder`
5. 公司分支：`X_company_ids + X_company_profile -> CompanyEncoder`
6. 邻居分支：`neighbor_symbol_ids + neighbor_scores -> neighbor pooling`

其中：

- 事件分支和市场分支先融合成 `context_repr`
- `context_repr` 再通过 `CrossGate` 调制时序、截面、公司、邻居分支
- 最后拼接后进入融合干路和多任务输出头

## 2. 输入维度总览

| 输入块 | 默认维度 |
|---|---:|
| `X_seq` | `[B, 20, 42]` |
| `X_tab` | `[B, 13]` |
| `X_event` | `[B, 256]` |
| `X_mkt` | `[B, 16]` |
| `X_company_profile` | `[B, 12]` |
| `symbol_id` | `[B]` |
| `industry_id` | `[B]` |
| `board_id` | `[B]` |
| `neighbor_symbol_ids` | `[B, 10]` |
| `neighbor_scores` | `[B, 10]` |

## 3. 时序分支

时序分支由 `TemporalFinancialEncoder` 实现。

### 3.1 输入

- 输入：`X_seq`
- 形状：`[B, 20, 42]`

### 3.2 层级结构与维度

1. `LayerNorm(42)`
   - 输入：`[B, 20, 42]`
   - 输出：`[B, 20, 42]`

2. `Linear(42 -> 96)`
   - 输入：`[B, 20, 42]`
   - 输出：`[B, 20, 96]`

3. 可学习位置编码 `position_embedding`
   - 参数形状：`[1, 20, 96]`
   - 广播相加后输出：`[B, 20, 96]`

4. `TemporalConvBlock(kernel_size=3)`
   - 内部结构：
     - `LayerNorm(96)`
     - depthwise `Conv1d(96 -> 96, kernel=3, groups=96)`
     - pointwise `Conv1d(96 -> 96, kernel=1)`
     - dropout
     - residual
   - 输出：`[B, 20, 96]`

5. `TemporalConvBlock(kernel_size=5)`
   - 结构同上
   - 输出：`[B, 20, 96]`

6. 双向 GRU
   - 配置：
     - `input_size = 96`
     - `hidden_size = 48`
     - `num_layers = 2`
     - `bidirectional = True`
   - 输出：`[B, 20, 96]`

7. 多头自注意力
   - `MultiheadAttention(embed_dim=96, num_heads=4, batch_first=True)`
   - 输出：`[B, 20, 96]`

8. 残差 + `LayerNorm(96)`
   - 输出：`[B, 20, 96]`

9. 三路池化
   - `AttentionPooling` 输出：`[B, 96]`
   - 最近 5 步平均 `recent`：`[B, 96]`
   - 最后一步 `last`：`[B, 96]`

10. 拼接
   - `concat([pooled, recent, last])`
   - 输出：`[B, 288]`

11. 输出投影
   - `Linear(288 -> 128)`
   - `LayerNorm(128)`
   - `GELU`
   - `Dropout`
   - 最终输出：`seq_repr = [B, 128]`

### 3.3 时序分支总结

| 阶段 | 输出维度 |
|---|---:|
| 输入 | `[B, 20, 42]` |
| 投影后 | `[B, 20, 96]` |
| 卷积块后 | `[B, 20, 96]` |
| GRU 后 | `[B, 20, 96]` |
| 自注意力后 | `[B, 20, 96]` |
| 三路池化拼接 | `[B, 288]` |
| 最终时序表示 | `[B, 128]` |

## 4. 截面分支

截面分支使用 `DeepMLPEncoder(input_dim=13, hidden_dim=192, output_dim=64, depth=2)`。

### 4.1 输入

- 输入：`X_tab`
- 形状：`[B, 13]`

### 4.2 层级结构与维度

1. 输入层
   - `Linear(13 -> 192)`
   - `LayerNorm(192)`
   - `GELU`
   - `Dropout`
   - 输出：`[B, 192]`

2. 两个 `GatedResidualBlock(192, 384)`
   - 每个 block 内部：
     - `LayerNorm(192)`
     - `Linear(192 -> 384)`
     - `GELU`
     - `Dropout`
     - `Linear(384 -> 192)`
     - `Dropout`
     - 门控残差
   - 每个 block 输出：`[B, 192]`

3. 输出层
   - `LayerNorm(192)`
   - `Linear(192 -> 64)`
   - `GELU`
   - 输出：`tab_repr = [B, 64]`

## 5. 事件分支

事件分支使用 `DeepMLPEncoder(input_dim=256, hidden_dim=192, output_dim=96, depth=2)`。

### 5.1 输入

- 输入：`X_event`
- 形状：`[B, 256]`

### 5.2 层级结构与维度

1. `Linear(256 -> 192)` + `LayerNorm` + `GELU` + `Dropout`
   - 输出：`[B, 192]`

2. 两个 `GatedResidualBlock(192, 384)`
   - 输出保持：`[B, 192]`

3. `LayerNorm(192)` + `Linear(192 -> 96)` + `GELU`
   - 输出：`event_repr = [B, 96]`

## 6. 市场分支

市场分支使用 `DeepMLPEncoder(input_dim=16, hidden_dim=96, output_dim=48, depth=2)`。

### 6.1 输入

- 输入：`X_mkt`
- 形状：`[B, 16]`

### 6.2 层级结构与维度

1. `Linear(16 -> 96)` + `LayerNorm` + `GELU` + `Dropout`
   - 输出：`[B, 96]`

2. 两个 `GatedResidualBlock(96, 192)`
   - 输出保持：`[B, 96]`

3. `LayerNorm(96)` + `Linear(96 -> 48)` + `GELU`
   - 输出：`mkt_repr = [B, 48]`

## 7. 上下文融合分支

事件分支和市场分支先融合成一个公共上下文表示 `context_repr`。

### 7.1 拼接

- `concat([event_repr, mkt_repr])`
- 维度：`96 + 48 = 144`
- 输出：`[B, 144]`

### 7.2 上下文投影

- `Linear(144 -> 96)`
- `LayerNorm(96)`
- `GELU`
- `Dropout`
- 输出：`context_repr = [B, 96]`

## 8. 公司分支

公司分支由 `CompanyEncoder` 实现。

### 8.1 输入

- `symbol_id`: `[B]`
- `industry_id`: `[B]`
- `board_id`: `[B]`
- `X_company_profile`: `[B, 12]`

### 8.2 离散 embedding 维度

当前模型里写死为：

- `symbol_emb_dim = 24`
- `industry_emb_dim = 12`
- `board_emb_dim = 8`

所以：

- `symbol_emb`: `[B, 24]`
- `industry_emb`: `[B, 12]`
- `board_emb`: `[B, 8]`

### 8.3 公司画像 MLP

公司画像输入维度是 `12`，`profile_hidden_dims=(96, 48)`，所以 `profile_mlp` 结构为：

1. `Linear(12 -> 96)`
2. `ReLU`
3. `Dropout`
4. `Linear(96 -> 48)`

输出：

- `profile_emb = [B, 48]`

### 8.4 拼接与输出

1. 拼接四部分
   - `24 + 12 + 8 + 48 = 92`
   - 输出：`[B, 92]`

2. `LayerNorm(92)`
   - 输出：`[B, 92]`

3. `output_mlp`
   - 结构：
     - `Linear(92 -> 48)`
     - `ReLU`
     - `Dropout`
     - `Linear(48 -> 48)`
   - 输出：`company_repr = [B, 48]`

## 9. 邻居分支

邻居分支复用 `symbol_embedding`，再按相似度加权池化。

### 9.1 输入

- `neighbor_symbol_ids`: `[B, 10]`
- `neighbor_scores`: `[B, 10]`

### 9.2 邻居 embedding

- 使用 `company_encoder.symbol_embedding`
- 单个邻居 embedding 维度：`24`
- 输出：`embedded = [B, 10, 24]`

### 9.3 加权池化

1. `weights = neighbor_scores.unsqueeze(-1)`
   - 输出：`[B, 10, 1]`

2. 加权求和
   - `(embedded * weights).sum(dim=1)`
   - 输出：`[B, 24]`

3. 用权重和归一化
   - 输出仍是：`[B, 24]`

### 9.4 邻居投影

- `Linear(24 -> 48)`
- `LayerNorm(48)`
- `GELU`
- 输出：`neighbor_repr = [B, 48]`

## 10. CrossGate 调制层

时序、截面、公司、邻居四个分支都会被 `context_repr` 调制。

### 10.1 CrossGate 结构

对任意输入 `x: [B, d]` 和 `context: [B, 96]`：

1. 拼接
   - `[B, d + 96]`

2. 门控网络
   - `Linear(d + 96 -> d)`
   - `GELU`
   - `Dropout`
   - `Linear(d -> d)`
   - `Sigmoid`
   - 输出 `gate`: `[B, d]`

3. 上下文投影
   - `Linear(96 -> d)`
   - 输出 `context_term`: `[B, d]`

4. 残差输出
   - `x + gate * context_term`
   - 输出：`[B, d]`

### 10.2 四个 gated 分支维度

| 分支 | 输入维度 | 输出维度 |
|---|---:|---:|
| `seq_gate` | `128` | `[B, 128]` |
| `tab_gate` | `64` | `[B, 64]` |
| `company_gate` | `48` | `[B, 48]` |
| `neighbor_gate` | `48` | `[B, 48]` |

## 11. 融合干路

### 11.1 拼接

模型将以下 5 个表示拼接：

- `seq_repr`: `[B, 128]`
- `tab_repr`: `[B, 64]`
- `context_repr`: `[B, 96]`
- `company_repr`: `[B, 48]`
- `neighbor_repr`: `[B, 48]`

拼接后：

- `128 + 64 + 96 + 48 + 48 = 384`
- 输出：`[B, 384]`

### 11.2 输入投影

- `Linear(384 -> 256)`
- `LayerNorm(256)`
- `GELU`
- `Dropout`
- 输出：`[B, 256]`

### 11.3 三个融合残差块

每个 `GatedResidualBlock(256, 512)` 结构：

- `LayerNorm(256)`
- `Linear(256 -> 512)`
- `GELU`
- `Dropout`
- `Linear(512 -> 256)`
- `Dropout`
- 门控残差

3 个 block 串联后输出仍为：

- `[B, 256]`

### 11.4 共享输出层

- `LayerNorm(256)`
- `Linear(256 -> 256)`
- `GELU`
- 输出：`shared_repr = [B, 256]`

## 12. 多任务输出头

模型有 4 个 head：

- `p_win`
- `ret_mu`
- `risk_dd`
- `rank_score`

每个 head 结构相同：

1. `Linear(256 -> 128)`
2. `LayerNorm(128)`
3. `GELU`
4. `Dropout`
5. `Linear(128 -> head_dim)`

### 12.1 各 head 输出维度

| head | 输出维度 |
|---|---:|
| `p_win` | `[B, 5]` |
| `ret_mu` | `[B, 5]` |
| `risk_dd` | `[B, 5]` |
| `rank_score` | `[B, 1]` |

## 13. 完整前向路径维度汇总

### 13.1 主干表示

| 模块 | 输出维度 |
|---|---:|
| `X_seq` 输入 | `[B, 20, 42]` |
| `TemporalFinancialEncoder` | `[B, 128]` |
| `X_tab` 输入 | `[B, 13]` |
| `tab_encoder` | `[B, 64]` |
| `X_event` 输入 | `[B, 256]` |
| `event_encoder` | `[B, 96]` |
| `X_mkt` 输入 | `[B, 16]` |
| `mkt_encoder` | `[B, 48]` |
| `context_fusion` | `[B, 96]` |
| `CompanyEncoder` | `[B, 48]` |
| `neighbor_proj` | `[B, 48]` |
| 拼接后 | `[B, 384]` |
| `fusion_in` | `[B, 256]` |
| 3 个融合残差块后 | `[B, 256]` |
| `shared_out` | `[B, 256]` |

### 13.2 最终输出

| head | 输出维度 |
|---|---:|
| `p_win` | `[B, 5]` |
| `ret_mu` | `[B, 5]` |
| `risk_dd` | `[B, 5]` |
| `rank_score` | `[B, 1]` |

## 14. 训练时的损失函数

虽然这不是结构层的一部分，但理解输出头通常需要一起看。

当前默认损失定义在 [train/run.py](/D:/python_object/股市数据提取预处理/train/run.py)：

- `p_win`
  - `binary_cross_entropy_with_logits`
  - 带 `pos_weight`
  - 带 horizon 权重
  - 支持 label smoothing
- `ret_mu`
  - `smooth_l1_loss`
  - 带 horizon 权重
- `risk_dd`
  - `smooth_l1_loss`
  - 带 horizon 权重
- `rank_score`
  - 先对输出做 `sigmoid`
  - 再做 `smooth_l1_loss`

默认总损失：

```text
total_loss =
  1.0 * loss_p_win +
  1.0 * loss_ret_mu +
  1.0 * loss_risk_dd +
  0.5 * loss_rank_score
```

## 15. 需要注意的两个“维度来源”

### 15.1 输入维度来自数据

以下维度由实际训练样本决定：

- `f_seq = 42`
- `f_tab = 13`
- `f_event = 256`
- `f_mkt = 16`
- `f_company_profile = 12`
- `seq_length = 20`

这些值来自 `PreparedSplit.input_dims`。

### 15.2 离散 embedding 的词表大小来自数据

以下大小不是写死的，而是由当前数据集里的 ID 上界决定：

- `num_symbols`
- `num_industries`
- `num_boards`

也就是说：

- embedding 输出维度固定
- embedding 表行数随当前训练集 vocab size 变化

## 16. 一句话总结

当前模型本质上是一个“多分支异构输入编码器 + 上下文门控融合 + 多任务头”的金融预测网络：

- `X_seq` 负责时序模式
- `X_tab` 负责截面静态信息
- `X_event + X_mkt` 负责日级上下文
- `X_company_ids + X_company_profile + neighbors` 负责公司层面的结构化先验
- 最终在 256 维共享表示上输出 4 组任务结果
