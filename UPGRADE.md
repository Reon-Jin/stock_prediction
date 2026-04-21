
---
# 一、先明确当前方案的核心问题
问题：

### 1. `p_win` 被其他头硬加硬减，耦合过强

现在：

* `ret_mu` 给 `p_win` 加分
* `risk_dd` 给 `p_win` 扣分
* `bigloss` 给 `p_win` 扣分
* `rank_score` 给 `p_win` 加分

这会导致：

* 辅助头学偏，主头一起偏
* 梯度目标互相拉扯
* 训练早期很不稳定
* 解释性也变差

---

### 2. 排序目标太弱

你现在的 `rank_score` 本质上还是回归一个分数，不是真正按“每天选前 K 只股票”的目标来优化。

但你的最终业务目标明显更像：

* 每天从全市场选 Top-K
* 希望这批票未来 (k) 天收益更高
* 同时控制大亏和回撤

所以你需要**横截面排序损失**，而不是简单回归 rank 分数。

---

### 3. 训练最优模型的判据不够业务化

你现在主要看：

* `valid_loss`
* `valid_p_win_acc`

这不够。
金融里“准确率高”并不等于“赚钱多”。

你真正该选的是更接近业务目标的综合指标，例如：

* 未来 (k) 天 top-k 平均收益
* top-k 大亏率
* top-k 风险调整后收益
* top-k 收益/回撤比
* calibrated precision / recall / balanced acc

---

### 4. 公司/邻居建模还有提升空间

当前 `CompanyEncoder` 是可用的，但偏轻。
邻居模块只是加权平均 symbol embedding，信息利用偏浅。

---

---

# 二、完整模型结构升级方案

我建议你把整个模型升级成一个更清晰的结构：

## 总体结构：从“多头互相硬修正”改成“共享表征 + 决策头分层”

分成四层：

### A. 多模态编码层

分别编码：

* **时序分支**：`X_seq`
* **静态财务分支**：`X_tab`
* **事件分支**：`X_event`
* **市场分支**：`X_mkt`
* **公司身份分支**：`X_company_ids + X_company_profile`
* **邻居分支**：`neighbors`

---

### B. 条件融合层

把事件、市场、公司、邻居作为上下文，对时序和静态特征进行条件调制。

---

### C. 任务共享表示层

得到一个共享 latent，再分裂成两类头：

#### 1）基础预测头

* `ret_mu_head`
* `risk_dd_head`
* `bigloss_head`
* `upside_head`（新增，可理解为未来上行潜力）
* `p_win_head`

#### 2）最终决策头

新增一个**决策分数头 `decision_score_head`**

这个头不再手工写规则加减，而是输入以下特征做最终决策：

* shared latent
* `ret_mu_pred`
* `risk_dd_pred`
* `bigloss_prob`
* `p_win_prob`
* `upside_pred`

由一个小 MLP 输出最终排序分数。

这样就把“金融规则先验”和“模型自学习决策”结合起来了。

---

## 2.1 时序编码器升级

你当前时序编码器已经可以，但我建议改成：

### 升级方向

用**轻量 Temporal Mixer + GRU + Attention Pooling**，保留轻量性，增强建模能力。

### 具体改法

#### 原方案

* input proj
* temporal conv x2
* bi-GRU
* MHA
* pooling

#### 升级方案

* `input_norm + linear`
* **Feature gating**：对每个时间步的 42 维特征先做通道门控
* **Temporal mixer block x2~3**

  * depthwise temporal conv
  * channel mixing MLP
  * residual
* **单层或双层 BiGRU**
* **attention pooling + recent pooling + last pooling**
* 输出 `seq_repr`

### 这样做的好处

* 比纯 Transformer 更轻
* 比现在的卷积块表达更强
* 对 20 天长度很合适
* 通道门控可以让不同特征在不同 regime 下权重不同

---

## 2.2 事件分支升级

你现在 `X_event` 是 256 维 dense vector，直接过 MLP。

建议升级为：

### 事件质量门控

增加一个 `event_confidence_gate`

输出：

* `event_repr`
* `event_strength`

即让模型自己学“当前事件向量是否可信、是否有信息量”。

然后在融合时：
[
event_repr = event_strength \cdot event_repr
]

### 原因

金融事件特征噪声很大，很多时候：

* 没新闻 ≠ 利空
* 有公告 ≠ 有交易意义
* 高维 embedding 里有很多无效成分

所以要让模型能“学会忽略弱信息”。

---

## 2.3 公司编码器升级

当前 `CompanyEncoder` 保留总体思路，但升级如下：

### 新版 CompanyEncoder 结构

#### 输入

* `symbol_emb`
* `industry_emb`
* `board_emb`
* `profile_mlp(profile_features)`

#### 升级点

1. **profile_mlp 改成 GELU + LayerNorm**
2. **symbol embedding 做 dropout**
3. **加入 industry-profile 交互门控**
4. **加入 board-profile 交互门控**
5. 输出前再过一个 residual fusion block

### 目标

避免模型过度记忆 symbol id，而是更多利用：

* 股票所属行业
* 板块制度属性
* 公司基本面画像

---

## 2.4 邻居模块升级

当前邻居模块只是对邻居 symbol embedding 加权平均。建议升级为：

### 邻居注意力聚合器

对每个样本的 10 个邻居做：

* 邻居 symbol embedding
* 邻居得分 `neighbor_scores`
* 与当前公司 embedding 的相似性交互
* attention pooling

即：

[
\alpha_i = \text{softmax}(MLP([neighbor_emb_i, company_repr, score_i]))
]

然后加权求和得到 `neighbor_repr`

### 好处

比简单平均更合理：

* 不是所有邻居都同等重要
* 当前股票和邻居之间的关系可以动态变化
* 不同市场环境下，邻居影响不同

---

## 2.5 融合层升级

当前是直接 concat 后过若干残差块。可以升级为：

### 双阶段融合

#### 第一阶段：上下文条件调制

用：

* `event_repr`
* `mkt_repr`
* `company_repr`
* `neighbor_repr`

去门控：

* `seq_repr`
* `tab_repr`

#### 第二阶段：融合器

把所有分支拼接后，通过：

* `fusion_in`
* `3~4层 gated residual block`
* **1层 cross-feature attention / feature mixer**
* `shared_repr`

这样会比纯 MLP 融合更强一点。

---

## 2.6 输出头升级：从“硬规则修正 p_win”改为“分层决策”

我建议把输出改成下面这些头：

### 基础头

* `ret_mu`: 各 horizon 预期收益
* `ret_sigma`: 各 horizon 收益不确定性（新增）
* `p_win`: 各 horizon 上涨概率
* `risk_dd`: 各 horizon 未来最大下行/回撤风险
* `bigloss`: 各 horizon 大亏概率
* `upside`: 各 horizon 上行潜力（新增）

### 决策头

* `decision_score`: 各 horizon 最终选股打分

### 关键变化

**不要再直接用 `ret_mu/risk_dd/bigloss` 去手工改 `p_win`。**

改成：

* 所有基础头独立训练
* 最终 `decision_score` 由这些头的预测结果 + shared latent 一起输入 MLP 学出来

即：

[
decision_score = MLP([shared, ret_mu, p_win, risk_dd, bigloss, upside, ret_sigma])
]

### 这比原来更合理的原因

* 不会把辅助头的误差直接注入主头
* 模型自己学“收益、胜率、风险怎么权衡”
* 更贴近最终选股目标
* 解释性也更强

---

# 三、损失函数全面升级：从“准确率导向”改成“收益-风险-进攻性平衡导向”

你的目标有三条：

1. **提高未来 (k) 天预期收益率**
2. **避免大幅亏损**
3. **不要过于保守，不敢预测上涨**

所以损失必须围绕这三条来构建。

---

## 3.1 总损失结构

我建议总损失：

[
L = L_{ret} + \lambda_{win}L_{p_win} + \lambda_{dd}L_{dd} + \lambda_{big}L_{bigloss} + \lambda_{rank}L_{rank} + \lambda_{up}L_{upside} + \lambda_{cal}L_{calibration} + \lambda_{adv}L_{anti_conservative}
]

下面分别说。

---

## 3.2 收益损失：核心主目标

### `ret_mu` 不再只用 SmoothL1

建议改成**分位数回归 + Huber**的混合方式。

#### 原因

金融收益分布：

* 重尾
* 偏态
* 极端值多

单纯 `SmoothL1` 太“平均化”，容易学得保守。

### 方案

让模型预测：

* `ret_q50`
* `ret_q70`
* `ret_q85`

或者至少：

* `ret_mu`
* `upside`

#### 对应损失

* `ret_mu`: Huber loss
* `upside`: 针对正收益尾部的 pinball loss / asymmetric loss

### 作用

这样模型不只是学平均收益，还学：

* 典型收益
* 偏乐观但合理的上行空间

这会减少“过于保守”的问题。

---

## 3.3 `p_win` 损失升级

### 当前问题

现在只是 BCE + 类别权重，还是太普通。

### 升级方案

使用：

### 1）`Focal BCE` 或加难例增强

对于难分样本、少数类样本给予更高权重。

### 2）保留 downside 权重

你现在这点是对的，继续保留。

### 3）加入**校准损失**

让输出概率更可用，而不只是对错。

可加一个简单的校准项：
[
L_{calibration} = \left|\text{mean}(\sigma(logits)) - \text{mean}(y)\right|
]
按 batch 或按日期分组做都可以。

### 目标

让 `p_win` 变成真正可用于决策的概率，而不是只是分类 logit。

---

## 3.4 风险损失 `risk_dd`

这一项很重要，因为你明确要求避免大幅亏损。

建议：

### 1）保持 Huber / SmoothL1

用于总体拟合。

### 2）对“坏样本”额外加大惩罚

比如当真实未来回撤超过阈值时：

* `drawdown < -0.05`
* `drawdown < -0.08`

则额外加权。

[
w_{dd} = 1 + \alpha \cdot \mathbf{1}(dd < -\tau)
]

### 目的

让模型更认真学习深回撤样本，而不是只拟合平均小波动。

---

## 3.5 大亏损失 `bigloss`

这一项建议升级成更重要的安全头。

### 定义建议

比如定义未来 (k) 天：

* 最大跌幅超过 6% / 8% / 10% 为 bigloss

### 损失

继续用 BCE/Focal BCE，但正类权重更大。

### 同时加入和决策分数的联动约束

对于真实 bigloss 样本，希望 `decision_score` 更低。

可以加一个 margin loss：

[
L_{bigloss_margin} = \max(0, decision_score - m)
]
对 bigloss 正类样本施加。

这样模型会学会：
**危险样本就不要排到前面。**

---

## 3.6 排序损失：最重要的新增项

这是整个升级方案的核心之一。

你最终不是要“所有样本平均预测都不错”，而是要：

**每天选前 K 个最值得买的股票。**

所以需要加横截面排序损失。

---

### 方案 A：分日期 Pairwise Ranking Loss

在同一天的股票截面内，构造股票对：

若股票 A 的未来收益显著高于 B，则希望：
[
decision_score_A > decision_score_B
]

损失可用：
[
L_{rank} = \log(1 + \exp(-(s_A - s_B)))
]

其中：

* 只对同一天样本构造 pair
* 只选收益差超过阈值的 pair
* 可按收益差大小加权

### 这是最推荐的方案

---

### 方案 B：Listwise Top-K Proxy

如果你愿意更进一步，可以对每个交易日：

* 用 softmax over decision scores
* 加权真实未来收益
* 直接优化“高分样本的加权收益更高”

但实现更复杂。
所以先上 Pairwise 就很好。

---

## 3.7 防止“过于保守”的损失

你明确提出：

> 不要过于保守不敢预测上涨

这是金融模型非常常见的问题。
因为一旦模型过于惩罚亏损，它很容易学成：

* 少给高分
* 少预测上涨
* 平均很稳，但赚不到钱

所以你必须显式加入“防保守机制”。

---

### 方法 1：上行捕捉损失 `upside_loss`

新增 `upside_head` 预测未来正向弹性，比如：
[
upside = \max(\text{future max return over k days}, 0)
]

对强上涨样本加强学习。

### 方法 2：正例召回约束

对于未来收益显著为正的样本：

* 如果 `decision_score` 太低，要额外惩罚

例如：
[
L_{anti_conservative} = \mathbf{1}(ret > \tau_{up}) \cdot \max(0, m - decision_score)
]

这样可以鼓励模型在真正好的样本上敢于给高分。

### 方法 3：预测上涨率约束

你现在已经统计了：

* target 正类比例
* pred 正类比例

可以增加一个轻量正则，约束模型不要长期把 `pred_pos_rate` 压得过低。

不是强行等于 target，而是防止塌缩到特别保守。

---

## 3.8 决策分数损失：最终交易目标

最终 `decision_score` 应该同时受到三种信号监督：

### 1）排序监督

高收益样本排前面

### 2）风险抑制监督

高 bigloss / 深回撤样本排后面

### 3）进攻性监督

真实强上涨样本不要被压低

所以 `decision_score` 的总损失可以写成：

[
L_{decision} = \lambda_1 L_{pairwise_rank} + \lambda_2 L_{bigloss_margin} + \lambda_3 L_{anti_conservative}
]

这会比单纯 `rank_score regression` 强得多。

---

# 四、训练过程升级方案

---

## 4.1 每轮结束保存当前最优模型

你要求：

> 每一轮结束都保存当前最优模型

这个要明确成两件事：

### 1）每轮都保存一个 last checkpoint

比如：

* `epoch_01.pt`
* `epoch_02.pt`

### 2）如果当前指标优于历史最佳，则覆盖 `best_model.pt`

### 推荐保存内容

* model state
* optimizer state
* scheduler state
* scaler state
* ema state
* current epoch
* best metric
* config
* calibration thresholds
* validation summary

---

## 4.2 最优模型不再只按 acc 选

建议新增一个**业务化综合指标**，例如：

[
score = \text{TopKReturn} - \alpha \cdot \text{TopKBigLossRate} - \beta \cdot \text{TopKDrawdownPenalty} + \gamma \cdot \text{BalancedAcc}
]

比如针对主 horizon（如 10d）：

* `valid_topk_ret_10d`
* `valid_topk_bigloss_rate_10d`
* `valid_topk_dd_10d`
* `valid_p_win_bal_acc_10d`

综合成一个 `valid_business_score`

### 最优模型就按这个选

这样保存出来的 best model 更贴近最终选股目标。

---

## 4.3 加入进度条可视化

你已经用 `tqdm` 了，但还可以更好。

### 训练时进度条显示这些信息

每个 batch 显示：

* 当前 loss
* p_win loss
* ret loss
* dd loss
* bigloss loss
* rank loss
* lr

### 每个 epoch 结束后打印：

* train / valid 总 loss
* 主 horizon 的 p_win_bal_acc
* 主 horizon 的 top-k 平均收益
* 主 horizon 的 top-k 大亏率
* 当前 best business score
* 是否保存 best model

### 额外保存

每轮更新：

* `history.csv`
* `history.json`
* `training_curves.png`

建议增加：

* `business_curves.png`
* `topk_metrics.png`

---

## 4.4 建议加入梯度监控与异常保护

为了训练更稳：

### 建议增加

* 梯度范数日志
* loss nan/inf 检查
* 如果某 batch 输出异常，跳过并记录

---

# 五、推荐的新评估指标体系

不要只看 acc。建议每个 horizon 都统计：

## 分类类

* `p_win_acc`
* `p_win_bal_acc`
* `precision`
* `recall`
* `pred_pos_rate`
* `target_pos_rate`

## 回归类

* `ret_mu_mae`
* `risk_dd_mae`
* `bigloss_bce`

## 决策类（最重要）

按每个交易日，对 `decision_score` 排序，取 top-k：

* `topk_avg_ret`
* `topk_median_ret`
* `topk_win_rate`
* `topk_bigloss_rate`
* `topk_avg_drawdown`
* `topk_sharpe_like = mean(ret) / std(ret)`
* `topk_utility = mean(ret) - α*bigloss_rate - β*abs(avg_drawdown)`

### 最佳模型选择建议

默认用：

* 主 horizon: `10d`
* `valid_topk_utility_10d`

---

# 六、推荐的新训练逻辑

---

## 阶段 1：基础多任务预训练

先训练这些头：

* `ret_mu`
* `p_win`
* `risk_dd`
* `bigloss`
* `upside`

先不让 `decision_score` 参与或权重很小。

### 目的

先学会基础金融表征，稳定。

---

## 阶段 2：加入决策头微调

再引入：

* `decision_score`
* pairwise ranking loss
* anti-conservative loss
* bigloss margin loss

### 目的

让模型从“会预测”进化为“会选股”。

---

## 阶段 3：阈值与业务决策校准

不再只校准 `p_win`，还可以校准：

* `decision_score` 的 top-k 规则
* 不同 horizon 下的风险过滤阈值

例如：

* `decision_score` 前 K
* 且 `bigloss_prob < x`
* 且 `risk_dd > y`

在验证集上搜索一组最优决策规则。

---

# 七、建议的默认权重

可以先从下面开始：

```python
loss_weights = {
    "ret_mu": 1.0,
    "p_win": 0.9,
    "risk_dd": 1.1,
    "bigloss": 1.2,
    "upside": 0.6,
    "rank_pairwise": 1.0,
    "bigloss_margin": 0.6,
    "anti_conservative": 0.5,
    "calibration": 0.15,
}
```

如果你特别重视防大亏，可以进一步提高：

* `risk_dd`
* `bigloss`

但不要把 `anti_conservative` 压太低，否则模型还是会变保守。

---

# 八、建议的代码结构调整

建议把项目代码拆成下面几个模块：

## `models/`

* `temporal_encoder.py`
* `company_encoder.py`
* `neighbor_encoder.py`
* `fusion.py`
* `heads.py`
* `decision_model.py`

## `train/`

* `losses.py`
* `metrics.py`
* `ranking.py`
* `checkpoint.py`
* `trainer.py`

## 好处

这样以后你调：

* 结构
* loss
* 指标
* 保存逻辑

都会很清晰。

---

# 九、你当前代码最应该立刻改的地方

按优先级：

## 第一优先级

**删掉 `p_win` 被其他头手工加减修正的逻辑**
改成单独的 `decision_score_head`

---

## 第二优先级

**把 `rank_score` 换成真正的 `decision_score + pairwise rank loss`**

---

## 第三优先级

**best model 按 `valid_topk_utility` 保存，而不是按 acc**

---

## 第四优先级

**升级邻居模块为注意力聚合**

---

## 第五优先级

**升级 CompanyEncoder 的 profile / interaction 结构**

---

# 十、训练保存与可视化的明确落地要求

你要求的第 3 点，我给你一个明确标准：

## 每轮训练中

进度条显示：

* `loss`
* `ret`
* `pwin`
* `dd`
* `bigloss`
* `rank`
* `lr`

## 每轮结束

保存：

* `best_model.pt`
* `epoch_{epoch:02d}.pt`

如果优于历史最佳，再保存：

* `best_model.pt`

并更新：

* `history.csv`
* `history.json`
* `training_curves.png`
* `business_curves.png`

---

# 十一、最终推荐的升级版模型定义

推荐你把最终模型理解成：

### 输入层

* 时序
* 静态财务
* 事件
* 市场
* 公司身份/画像
* 邻居图信息

### 编码层

* Temporal Mixer + GRU
* Event gate
* Mkt encoder
* Upgraded CompanyEncoder
* Neighbor attention encoder

### 融合层

* Context gating
* Shared fusion blocks

### 输出层

基础预测头：

* `ret_mu`
* `ret_sigma`
* `p_win`
* `risk_dd`
* `bigloss`
* `upside`

决策头：

* `decision_score`

### 损失层

* 收益拟合
* 胜率分类
* 回撤控制
* 大亏抑制
* 上行捕捉
* 横截面排序
* 防保守约束
* 校准约束



