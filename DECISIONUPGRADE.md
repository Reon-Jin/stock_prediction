---

# 一、先明确 v1.0 的问题

当前的 `decision_v1.0` 作为第一代是合理的，但有这些结构性问题：

### 1. 主评分函数过于线性

现在本质还是：

* 胜率
* 收益
* 回撤
* 排序分

做线性加权。
这不够表达真实交易中的非线性关系。

---

### 2. `bigloss` 没有进入核心效用函数

它更多在风控二次审查中起作用，而不是主决策目标的一部分。
这会导致主评分和最终风控目标分裂。

---

### 3. 周期冲突处理过于刚性

只要短长周期冲突，就倾向 `WATCH/REDUCE`，容易错杀：

* 启动初期
* 中继调整
* 短期洗盘后中期继续向上

---

### 4. `best_horizon` 定义过于单一

只看 `ret_mu / risk_dd`，没同时考虑：

* 胜率
* 大亏概率
* 市场状态
* 风险偏好
* 当前是否已持仓

---

### 5. 批量推荐排序仍偏启发式

没有真正围绕：

* top-k 平均收益
* top-k 大亏率
* 行业拥挤风险
* 相似股票冗余

去做组合层选择。

---

### 6. 风控规则很多，但缺乏层级化

现在更像“不断降级动作”，容易把系统压得过保守。

---

所以 `v2.0` 的核心升级方向应该是：

> **从“规则打分引擎”升级为“收益-风险-尾部损失联合优化的效用决策引擎”。**

---

# 二、`decision_v2.0` 总体架构

我建议把引擎拆成 6 层：

## Layer 1：输入标准化层

把模型输出统一转成可决策的标准字段。

## Layer 2：单周期效用层

对 3/5/10/20/40 每个周期分别计算：

* 预期收益效用
* 胜率效用
* 回撤惩罚
* 大亏惩罚
* 排序信号
* 上行弹性奖励
* 周期置信度

得到每个 horizon 的 `U_k`

## Layer 3：跨周期聚合层

把各 horizon 的效用按市场状态、风险偏好、是否持仓、持仓天数动态聚合，得到：

* `U_final`
* `consistency_score`
* `conflict_penalty`
* `best_horizon`

## Layer 4：动作决策层

根据：

* `U_final`
* `best_horizon`
* `market_regime`
* `holding_status`
* `confidence`
* `risk_flags`

映射到动作：

* 未持仓：`STRONG_BUY / BUY / WATCH / AVOID / STRONG_AVOID`
* 已持仓：`ADD / HOLD / REDUCE / SELL / STOP_LOSS`

## Layer 5：风险控制层

只处理：

* 极端风险
* 数据质量问题
* 覆盖不足
* 市场熔断级风险
* 流动性硬限制

而不是无限制接管主决策。

## Layer 6：批量推荐与组合构建层

从全市场候选中：

* 过滤
* 排序
* 分散
* 去相似
* 组合层约束

---

# 三、输入输出协议升级

---

## 3.1 输入字段升级

你当前输入已经够多了，但建议补充以下字段，形成统一协议：

```python
record = {
    "symbol": "...",
    "name": "...",
    "trade_date": "...",

    # multi-horizon probability
    "p_win_prob_3": ...,
    "p_win_prob_5": ...,
    "p_win_prob_10": ...,
    "p_win_prob_20": ...,
    "p_win_prob_40": ...,

    # expected return
    "ret_mu_pred_3": ...,
    "ret_mu_pred_5": ...,
    "ret_mu_pred_10": ...,
    "ret_mu_pred_20": ...,
    "ret_mu_pred_40": ...,

    # predicted downside / drawdown
    "risk_dd_pred_3": ...,
    "risk_dd_pred_5": ...,
    "risk_dd_pred_10": ...,
    "risk_dd_pred_20": ...,
    "risk_dd_pred_40": ...,

    # predicted tail risk
    "bigloss_prob_3": ...,
    "bigloss_prob_5": ...,
    "bigloss_prob_10": ...,
    "bigloss_prob_20": ...,
    "bigloss_prob_40": ...,

    # optional: upside / uncertainty
    "upside_pred_3": ...,
    "upside_pred_5": ...,
    "upside_pred_10": ...,
    "upside_pred_20": ...,
    "upside_pred_40": ...,

    "ret_sigma_pred_3": ...,
    "ret_sigma_pred_5": ...,
    "ret_sigma_pred_10": ...,
    "ret_sigma_pred_20": ...,
    "ret_sigma_pred_40": ...,

    # cross-sectional ranking
    "rank_score_pred": ...,

    # market regime
    "market_regime_prob": ...,
    "market_regime_score": ...,
    "risk_on_flag": ...,
    "risk_off_flag": ...,

    # market state details
    "limit_count_ratio": ...,
    "sector_hotness_spread": ...,
    "market_volatility_5": ...,

    # tradability / eligibility
    "is_st": ...,
    "is_suspended": ...,
    "list_days": ...,
    "pct_chg_1d": ...,
    "pct_chg_5d": ...,
    "amount_ma5": ...,

    # quality / reliability
    "feature_missing_rate": ...,
    "sample_coverage_score": ...,
    "model_drift_score": ...,
    "ranker_quality_score": ...,
}
```

---

## 3.2 外部上下文输入升级

```python
context = {
    "is_holding": False,
    "holding_days": 0,
    "entry_price": None,
    "risk_preference": "balanced",  # conservative / balanced / aggressive
    "strategy_style": "short_mid",  # short / short_mid / mid
    "position_size_hint": None,
}
```

---

## 3.3 输出协议升级

建议输出：

```python
{
    "symbol": "...",
    "symbol_name": "...",
    "trade_date": "...",

    "decision": {
        "action": "BUY",
        "action_cn": "推荐买入",
        "confidence": 0.81,
        "priority": 2,
        "path": "not_holding",
        "best_horizon": "k=10",
        "suggested_hold_days": 10,
        "position_hint": "half",   # light / half / full / reduce
    },

    "utility": {
        "U_3": ...,
        "U_5": ...,
        "U_10": ...,
        "U_20": ...,
        "U_40": ...,
        "U_final": ...,
        "consistency_score": ...,
        "conflict_penalty": ...,
        "confidence_score": ...,
    },

    "horizon_analysis": {
        "best_horizon": "k=10",
        "best_horizon_score": ...,
        "horizon_ranking": [...],
        "short_term_view": "...",
        "mid_term_view": "...",
        "conflict_type": "...",
    },

    "model_output": {...},

    "market_regime": "neutral",
    "market_regime_detail": {...},

    "risk_review": {
        "risk_level": "medium",
        "risk_flags": [...],
        "warnings": [...],
        "hard_blocks": [...],
        "downgraded": True,
        "original_action": "...",
        "final_action": "...",
    },

    "reasons": [...],

    "metadata": {
        "engine_version": "decision_v2.0",
        "risk_preference": "balanced",
        "strategy_style": "short_mid",
    }
}
```

---

# 四、核心评分体系升级：从 `S_final` 到 `U_final`

这是整个升级方案的核心。

---

## 4.1 每个周期不再用“简单线性分”，改成“效用函数”

对于每个 horizon (k)：

[
U_k = R_k + W_k + Q_k + X_k - D_k - B_k - V_k - C_k
]

其中：

* (R_k)：收益效用
* (W_k)：胜率效用
* (Q_k)：排序信号
* (X_k)：上行弹性奖励
* (D_k)：回撤惩罚
* (B_k)：大亏惩罚
* (V_k)：不确定性惩罚
* (C_k)：周期冲突/稳定性惩罚（后面再加）

---

## 4.2 各分项的建议定义

---

### 4.2.1 收益效用 `R_k`

不要直接拿 `ret_mu_pred_k` 原值，而应做尺度归一化和非线性拉伸。

建议：

[
R_k = w_r \cdot \tanh(ret_mu_k / s_k)
]

其中：

* `s_k` 是该 horizon 的经验尺度，如 3日、5日、10日不同
* `tanh` 防止极端值过度支配

---

### 4.2.2 胜率效用 `W_k`

建议不直接用 `p_win` 原值，而是做中心化：

[
W_k = w_w \cdot (p_win_k - 0.5) \cdot 2
]

这样：

* 0.5 附近接近 0
* 高于 0.5 才加分
* 低于 0.5 才减分

比直接用概率本身更合理。

---

### 4.2.3 排序信号 `Q_k`

因为 `rank_score_pred` 是跨 horizon 共享的，所以它可以作为一个全局加成项，但建议权重较小：

[
Q_k = w_q \cdot rank_score_norm
]

不要让它压过收益和风险。

---

### 4.2.4 上行弹性奖励 `X_k`

如果模型有 `upside_pred_k`，它应作为“不要太保守”的奖励项：

[
X_k = w_x \cdot \tanh(upside_k / u_k)
]

这能鼓励模型在真正强势机会面前更敢给买入。

---

### 4.2.5 回撤惩罚 `D_k`

回撤不应线性惩罚，建议非线性。

设 `dd_k = abs(min(risk_dd_pred_k, 0))`

[
D_k = w_d \cdot \left(\frac{dd_k}{d_k}\right)^\alpha
]

其中：

* (\alpha > 1)，例如 1.3~1.8
* 大回撤惩罚加速上升

这样比单纯 `1 - normalized_dd` 更合理。

---

### 4.2.6 大亏惩罚 `B_k`

这是 v2.0 必须加强的一项。

建议：

[
B_k = w_b \cdot \text{softplus}(\beta \cdot (bigloss_k - \tau_b))
]

即：

* 大亏概率在低位时影响小
* 超过阈值后惩罚迅速增加

这比简单线性减分更符合风控需求。

---

### 4.2.7 不确定性惩罚 `V_k`

如果模型未来有 `ret_sigma_pred_k`，建议引入：

[
V_k = w_v \cdot \tanh(ret_sigma_k / \sigma_k)
]

即：

* 在收益接近时，更偏好不确定性低的样本

如果没有 `ret_sigma`，可以先不启用。

---

# 五、跨周期聚合升级

---

## 5.1 时间权重不再固定死板

当前你是固定：

```python
3: 0.30
5: 0.30
10: 0.20
20: 0.12
40: 0.08
```

v2.0 建议做成**动态时间权重**，由以下因素决定：

* `risk_preference`
* `strategy_style`
* `market_regime`
* `is_holding`
* `holding_days`

### 例如

#### balanced + short_mid + not holding

```python
3: 0.24
5: 0.28
10: 0.24
20: 0.16
40: 0.08
```

#### aggressive + risk_on

增加短期：

```python
3: 0.30
5: 0.30
10: 0.22
20: 0.12
40: 0.06
```

#### conservative + risk_off

增加中期稳定性：

```python
3: 0.18
5: 0.22
10: 0.26
20: 0.22
40: 0.12
```

---

## 5.2 一致性指标改造

当前公式不够稳。建议改成两部分：

### 1）方向一致性 `direction_consistency`

看各 horizon 是否整体同向。

例如：

* 把 `U_k > 0` 记为正
* 统计正负方向一致程度

### 2）强度一致性 `magnitude_consistency`

看 `U_k` 的离散程度是否过大。

### 最终：

[
consistency_score = 0.6 \cdot direction_consistency + 0.4 \cdot magnitude_consistency
]

这样更合理。

---

## 5.3 周期冲突改成“软惩罚 + 分型”

不要一出现冲突就直接 `WATCH`。

建议把冲突分成：

### A. 良性错位

* 短期一般，中期强
* 短期强，中期也不弱

处理：只轻微降低置信度

### B. 过热型冲突

* 短期极强，中长期偏弱，且涨幅过热

处理：降级，倾向 `WATCH` 或 `REDUCE`

### C. 反转型冲突

* 短期很弱，中长期很强

处理：未持仓可 `WATCH`，已持仓可 `HOLD` 不急于卖

### D. 恶性冲突

* 短期、长期方向完全对立，且风险高

处理：较强惩罚甚至 veto

定义一个：

[
conflict_penalty \in [0, 1]
]

最终：

[
U_{final} = \sum_k \omega_k U_k - \lambda_c \cdot conflict_penalty
]

---

## 5.4 `best_horizon` 升级

不要再用单纯 `ret_mu / risk_dd`。

建议定义 horizon utility：

[
H_k =
a \cdot R_k

* b \cdot W_k
* c \cdot X_k

- d \cdot D_k
- e \cdot B_k
- f \cdot V_k
  ]

`best_horizon = argmax(H_k)`

### 好处

会更符合真实持仓周期选择，而不是被极小分母误导。

---

# 六、动作决策升级

---

## 6.1 动作集合保留，但映射逻辑重写

### 未持仓动作

* `STRONG_BUY`
* `BUY`
* `WATCH`
* `AVOID`
* `STRONG_AVOID`

### 已持仓动作

* `ADD_POSITION`
* `HOLD`
* `REDUCE`
* `SELL`
* `STOP_LOSS`

---

## 6.2 决策不再只看阈值，还看“效用 + 置信度 + 风险层级”

建议先定义几个核心量：

* `U_final`
* `confidence_score`
* `risk_level`
* `consistency_score`
* `best_horizon`
* `regime_state`

---

## 6.3 未持仓路径建议

### `STRONG_BUY`

满足：

* `U_final >= strong_buy_threshold`
* `confidence_score >= 0.75`
* `consistency_score >= 0.65`
* `risk_level == low`
* `bigloss_penalty` 不高
* 非严重冲突

### `BUY`

满足：

* `U_final >= buy_threshold`
* `confidence_score >= 0.55`
* `risk_level <= medium`

### `WATCH`

满足：

* `U_final` 中等
* 或存在轻中度冲突
* 或市场风险较高但非极端
* 或收益有吸引力但风险不完全匹配

### `AVOID`

满足：

* `U_final` 偏低
* 或风险偏高
* 或模型质量信号弱

### `STRONG_AVOID`

满足：

* `U_final` 极低
* 或 severe risk
* 或极端熔断级风险

---

## 6.4 已持仓路径建议

已持仓时不能只看当前 `U_final`，还应看：

* 当前 `best_horizon`
* `holding_days`
* `holding_days / best_horizon`
* 效用变化趋势
* 风险变化
* 市场状态

### `ADD_POSITION`

* `U_final` 很高
* `risk_level` 低
* `holding_days` 未接近最优周期上限
* 没有过热信号

### `HOLD`

* `U_final` 仍为正且稳定
* 风险未恶化
* `holding_days` 未明显超期

### `REDUCE`

* `U_final` 下降
* 或冲突增强
* 或短期过热
* 或已接近最优周期末端

### `SELL`

* `U_final` 转负
* 或收益风险比明显恶化
* 或已超过最佳持有周期较多

### `STOP_LOSS`

* 极端风险触发
* 或 `bigloss` / `risk_dd` 显著恶化
* 或市场 risk_off 且个股同步走坏

---

# 七、持仓管理升级

这个是 v2.0 很重要的一块。

---

## 7.1 引入“持仓阶段”概念

已持仓时把状态分为：

* `early_hold`：刚买入不久
* `mid_hold`：持有中段
* `late_hold`：接近最佳周期尾部
* `over_hold`：明显超出最佳周期

### 作用

不同阶段动作阈值不同。

例如：

* `early_hold` 不应太容易卖出
* `late_hold` 可更敏感地减仓/卖出
* `over_hold` 应显著提高卖出倾向

---

## 7.2 持仓天数相对化

定义：

[
hold_progress = \frac{holding_days}{best_horizon_days}
]

例如：

* `< 0.4`：early
* `0.4~0.9`：mid
* `0.9~1.2`：late
* `> 1.2`：over

这比直接看绝对天数更合理。

---

## 7.3 加仓逻辑要更严格

加仓不应只是“分高”。建议额外要求：

* 当前不在过热状态
* `bigloss_prob` 低
* 市场不是 risk_off
* 流动性足够
* 个股并非行业内过度拥挤

---

# 八、风险控制层升级

风险控制层要保留，但必须更分层。

---

## 8.1 风险分为三类

### 1）硬阻断风险 `hard_block`

一旦触发，直接拦截：

* ST / 停牌
* 上市不足天数
* 成交额严重不足
* 特征缺失严重
* 模型漂移严重
* 市场极端熔断级风险
* 大亏概率极高

### 2）强降级风险 `hard_downgrade`

触发后动作大幅降级：

* 5/10日回撤很高
* 短期涨幅过热
* risk_off + 高波动
* 覆盖不足
* ranker 质量弱

### 3）软惩罚风险 `soft_penalty`

只是降低 `U_final` 或置信度：

* 轻度冲突
* 中等流动性
* 轻度过热
* 行业风险分位较高

---

## 8.2 风控层不再“一路只会降级”，而是分两种作用方式

### A. 前置阻断

直接不让进入候选池。

### B. 后置惩罚

降低效用或置信度，而不是一票否决。

这样能避免系统过度保守。

---

## 8.3 建议新增风险评分

定义：

[
risk_score \in [0, 1]
]

由以下组成：

* 回撤风险
* 大亏风险
* 市场风险
* 流动性风险
* 数据质量风险
* 过热风险

最终给出：

* `low`
* `medium`
* `high`
* `severe`

动作映射依赖这个风险等级，而不是只依赖离散规则。

---

# 九、批量推荐与组合构建升级

这是非常关键的。

---

## 9.1 候选过滤条件升级

先保留硬条件：

* 非 ST
* 非停牌
* 上市天数 >= 120
* 非一字涨停临界
* 有效数据齐全
* 流动性过关

再加：

* `U_final >= buy_threshold`
* `confidence_score >= min_confidence`
* `risk_level <= medium`
* `bigloss_prob_5/10 <= threshold`
* `consistency_score >= min_consistency`

---

## 9.2 排序分数升级：从 `ranking_score` 到 `selection_utility`

建议每个候选的组合选择分数：

[
selection_utility =
0.32 \cdot U_{final}

* 0.18 \cdot rank_score
* 0.16 \cdot consistency
* 0.14 \cdot upside_bonus

- 0.12 \cdot drawdown_risk
- 0.08 \cdot bigloss_risk
  ]

如果无 `upside_pred`，可暂时把权重分给 `U_final`。

---

## 9.3 组合层动态分散约束

现在你已有：

* 单行业上限
* 单板块上限
* 去相似

v2.0 建议升级为动态约束：

### risk_on

允许更集中一些：

* 行业上限略高
* 板块上限略高

### risk_off

要求更分散：

* 行业上限更低
* 高相似股票更严格去重
* 高风险行业降权

---

## 9.4 相似股票去重升级

不要只做简单“避免过于相似”，建议给每个已选股票维护一个 `selected_symbols` 集合，后续候选增加相似性惩罚：

[
adjusted_utility = selection_utility - \lambda_{sim} \cdot max_similarity
]

这样是软去重，而不是生硬剔除。

---

## 9.5 最终排序输出建议增加组合解释

输出不仅要给单只股票原因，还要给组合层原因：

* 为什么入选
* 为什么某热门股被放弃（例如行业拥挤、风险高、与已选股票太像）

这对前端解释很有帮助。

---

# 十、校准与可配置化

---

## 10.1 所有阈值必须配置化

不要硬写在代码里。

建议 `decision/config_v2.yaml` 里配置：

* 风险偏好参数
* 市场 regime 阈值
* 动作阈值
* 风控阈值
* 候选池过滤阈值
* 组合分散阈值

---

## 10.2 阈值分成三类

### 固定逻辑阈值

如：

* ST、停牌、上市天数

### 可校准阈值

如：

* `buy_threshold`
* `bigloss_prob_threshold`
* `consistency_threshold`

### 动态阈值

根据市场状态和风险偏好动态调整。

---

## 10.3 建议做历史回测校准

用历史验证集/回测样本自动搜索：

* 各 horizon utility 权重
* 买入阈值
* 风险等级阈值
* top-n 组合分散参数

目标不是最大化准确率，而是最大化：

[
topk_utility = mean(return) - \alpha \cdot bigloss_rate - \beta \cdot drawdown
]

---

# 十一、代码结构重构建议

建议把 `decision/engine.py` 拆成下面这些文件：

```text
decision/
├─ engine.py                 # 主入口
├─ config.py                 # 配置读取
├─ schemas.py                # 输入输出数据结构
├─ market_regime.py          # 市场状态判断
├─ normalizers.py            # 收益/回撤/概率等标准化
├─ utility.py                # 单周期效用函数、跨周期聚合
├─ horizon.py                # best_horizon 选择逻辑
├─ actions.py                # 动作映射
├─ holding.py                # 持仓管理与持有阶段逻辑
├─ risk_controls.py          # 风控
├─ ranking.py                # 批量推荐排序
├─ diversification.py        # 分散与去相似
├─ explain.py                # reasons / warnings / metadata 生成
└─ config_v2.yaml            # 所有参数配置
```

---

## 11.1 `engine.py` 推荐主流程

```python
def evaluate_stock_decision(record, context, config):
    normalized = normalize_record(record, config)
    regime = infer_market_regime(record, config)

    per_horizon = compute_horizon_utilities(
        normalized=normalized,
        regime=regime,
        context=context,
        config=config,
    )

    horizon_summary = aggregate_horizon_utilities(
        per_horizon=per_horizon,
        regime=regime,
        context=context,
        config=config,
    )

    base_action = decide_action(
        horizon_summary=horizon_summary,
        regime=regime,
        context=context,
        config=config,
    )

    risk_review = apply_risk_controls(
        record=record,
        horizon_summary=horizon_summary,
        regime=regime,
        base_action=base_action,
        context=context,
        config=config,
    )

    result = build_explanation_and_output(
        record=record,
        regime=regime,
        per_horizon=per_horizon,
        horizon_summary=horizon_summary,
        risk_review=risk_review,
        context=context,
        config=config,
    )
    return result
```

---

## 11.2 `rank_market_candidates()` 推荐流程

```python
def rank_market_candidates(records, context, config, top_n=20):
    evaluated = [evaluate_stock_decision(r, context, config) for r in records]

    eligible = filter_candidates(evaluated, config)
    scored = score_candidates_for_selection(eligible, config)

    selected = diversify_and_select(scored, top_n=top_n, config=config)

    return selected
```

---

# 十二、默认参数建议

下面给你一套可作为初始版的默认参数思路。

---

## 12.1 风险偏好权重建议

### conservative

```python
{
    "ret": 0.22,
    "p_win": 0.24,
    "rank": 0.10,
    "upside": 0.08,
    "drawdown_penalty": 0.20,
    "bigloss_penalty": 0.12,
    "uncertainty_penalty": 0.04,
}
```

### balanced

```python
{
    "ret": 0.26,
    "p_win": 0.22,
    "rank": 0.12,
    "upside": 0.10,
    "drawdown_penalty": 0.16,
    "bigloss_penalty": 0.10,
    "uncertainty_penalty": 0.04,
}
```

### aggressive

```python
{
    "ret": 0.30,
    "p_win": 0.18,
    "rank": 0.16,
    "upside": 0.14,
    "drawdown_penalty": 0.12,
    "bigloss_penalty": 0.06,
    "uncertainty_penalty": 0.04,
}
```

---

## 12.2 动作阈值建议（以 `U_final` 为基准）

### risk_on

```python
strong_buy: 0.42
buy:        0.20
watch_low: -0.05
avoid:     -0.20
```

### neutral

```python
strong_buy: 0.48
buy:        0.24
watch_low: -0.08
avoid:     -0.24
```

### risk_off

```python
strong_buy: 0.58
buy:        0.32
watch_low: -0.12
avoid:     -0.28
```

这里不要照搬旧版的 0~1 逻辑，因为新效用分可以是正负混合。

---

## 12.3 风险等级阈值建议

```python
low:    risk_score < 0.25
medium: 0.25 <= risk_score < 0.50
high:   0.50 <= risk_score < 0.75
severe: risk_score >= 0.75
```

---

# 十三、和模型训练目标的对齐关系

这个决策引擎要和你前面准备升级的模型一一对齐：

### 模型输出

* `ret_mu`
* `p_win`
* `risk_dd`
* `bigloss`
* `upside`
* `rank_score`
* `market_regime`

### 决策引擎使用方式

* `ret_mu`：收益主效用
* `p_win`：方向置信度
* `risk_dd`：下行惩罚
* `bigloss`：尾部风险惩罚
* `upside`：防过度保守
* `rank_score`：横截面竞争力
* `market_regime`：动态调整阈值与权重

这样训练目标和引擎目标就不会脱节。

---

# 十四、建议实施顺序

不要一次全改完，建议分三步。

---

## 第一步：低风险升级

先不改整体接口，只做这些：

* 把 `S_k` 改成 `U_k`
* 把 `bigloss_prob` 纳入主评分
* 把 `best_horizon` 改成 horizon utility
* 把冲突从硬否决改成 soft penalty
* 输出增加 `risk_score`、`confidence_score`

这一步就已经会明显提升合理性。

---

## 第二步：动作与持仓管理升级

加入：

* `holding_stage`
* `hold_progress`
* 更精细的 `ADD/HOLD/REDUCE/SELL`
* position hint

---

## 第三步：批量推荐升级

加入：

* `selection_utility`
* 动态行业/板块约束
* 相似性软惩罚
* 组合层解释

---

