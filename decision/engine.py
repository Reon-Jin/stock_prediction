from __future__ import annotations

import math
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ENGINE_VERSION = "decision_v2.0"
EPS = 1e-6
HORIZONS = (3, 5, 10, 20, 40)
BIGLOSS_HORIZONS = HORIZONS
QUICK_RECOMMEND_MIN_WIN_RATE = 0.50

DEFAULT_CONFIG: dict[str, Any] = {
    "return_scales": {3: 0.020, 5: 0.030, 10: 0.045, 20: 0.065, 40: 0.090},
    "drawdown_scales": {3: 0.040, 5: 0.060, 10: 0.090, 20: 0.120, 40: 0.160},
    "upside_scales": {3: 0.030, 5: 0.045, 10: 0.065, 20: 0.090, 40: 0.120},
    "sigma_scales": {3: 0.025, 5: 0.035, 10: 0.050, 20: 0.075, 40: 0.100},
    "utility_weights": {
        "conservative": {
            "ret": 0.22,
            "p_win": 0.24,
            "rank": 0.10,
            "upside": 0.08,
            "drawdown_penalty": 0.20,
            "bigloss_penalty": 0.12,
            "uncertainty_penalty": 0.04,
        },
        "balanced": {
            "ret": 0.26,
            "p_win": 0.22,
            "rank": 0.12,
            "upside": 0.10,
            "drawdown_penalty": 0.16,
            "bigloss_penalty": 0.10,
            "uncertainty_penalty": 0.04,
        },
        "aggressive": {
            "ret": 0.30,
            "p_win": 0.18,
            "rank": 0.16,
            "upside": 0.14,
            "drawdown_penalty": 0.12,
            "bigloss_penalty": 0.06,
            "uncertainty_penalty": 0.04,
        },
    },
    "time_weights": {
        "short": {3: 0.36, 5: 0.32, 10: 0.18, 20: 0.10, 40: 0.04},
        "short_mid": {3: 0.24, 5: 0.28, 10: 0.24, 20: 0.16, 40: 0.08},
        "mid": {3: 0.14, 5: 0.20, 10: 0.28, 20: 0.24, 40: 0.14},
    },
    "action_thresholds": {
        "risk_on": {"strong_buy": 0.42, "buy": 0.20, "watch_low": -0.05, "avoid": -0.20},
        "neutral": {"strong_buy": 0.48, "buy": 0.24, "watch_low": -0.08, "avoid": -0.24},
        "risk_off": {"strong_buy": 0.58, "buy": 0.32, "watch_low": -0.12, "avoid": -0.28},
    },
    "risk_thresholds": {
        "low": 0.25,
        "medium": 0.50,
        "high": 0.75,
        "hard_bigloss": 0.72,
        "buy_bigloss": 0.45,
        "feature_missing_hard": 0.35,
        "feature_missing_warn": 0.15,
        "model_drift_hard": 0.75,
        "liquidity_hard": 5_000_000,
        "liquidity_warn": 20_000_000,
        "list_days_min": 120,
    },
    "utility_shape": {
        "drawdown_alpha": 1.45,
        "bigloss_beta": 8.0,
        "bigloss_tau": 0.35,
        "conflict_lambda": 0.35,
    },
    "risk_score_weights": {
        "drawdown": 0.25,
        "bigloss": 0.24,
        "market": 0.16,
        "liquidity": 0.12,
        "data_quality": 0.13,
        "overheat": 0.10,
    },
    "selection": {
        "min_confidence": 0.55,
        "min_consistency": 0.50,
        "max_bigloss_5_10": 0.45,
        "similarity_penalty": 0.10,
        "risk_on_industry_fraction": 0.45,
        "neutral_industry_fraction": 0.34,
        "risk_off_industry_fraction": 0.25,
        "risk_on_board_fraction": 0.70,
        "neutral_board_fraction": 0.60,
        "risk_off_board_fraction": 0.50,
    },
}

ACTION_META = {
    "STRONG_BUY": {"action_cn": "强烈推荐买入", "priority": 1, "path": "not_holding"},
    "BUY": {"action_cn": "推荐买入", "priority": 2, "path": "not_holding"},
    "RECOMMEND_BUY": {"action_cn": "推荐买入", "priority": 2, "path": "not_holding"},
    "WATCH": {"action_cn": "观望", "priority": 3, "path": "general"},
    "AVOID": {"action_cn": "建议回避", "priority": 4, "path": "not_holding"},
    "STRONG_AVOID": {"action_cn": "强烈回避", "priority": 5, "path": "not_holding"},
    "ADD_POSITION": {"action_cn": "建议加仓", "priority": 1, "path": "holding"},
    "HOLD": {"action_cn": "建议继续持有", "priority": 2, "path": "holding"},
    "REDUCE": {"action_cn": "建议逢高减仓", "priority": 3, "path": "holding"},
    "SELL": {"action_cn": "建议卖出", "priority": 4, "path": "holding"},
    "STOP_LOSS": {"action_cn": "建议止损", "priority": 5, "path": "holding"},
}
BUY_ACTIONS = {"STRONG_BUY", "BUY", "RECOMMEND_BUY", "ADD_POSITION"}
BUY_PATH_ORDER = ["STRONG_BUY", "BUY", "WATCH", "AVOID", "STRONG_AVOID"]
HOLD_PATH_ORDER = ["ADD_POSITION", "HOLD", "REDUCE", "SELL", "STOP_LOSS"]

_CONFIG_CACHE: dict[str, Any] | None = None


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_config_keys(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[Any, Any] = {}
        for key, item in value.items():
            normalized_key: Any = key
            if isinstance(key, str) and key.isdigit():
                normalized_key = int(key)
            output[normalized_key] = _normalize_config_keys(item)
        return output
    if isinstance(value, list):
        return [_normalize_config_keys(item) for item in value]
    return value


def _load_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        config_path = Path(__file__).with_name("config_v2.yaml")
        loaded: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        _CONFIG_CACHE = _deep_merge(DEFAULT_CONFIG, _normalize_config_keys(loaded))
    if config:
        return _deep_merge(_CONFIG_CACHE, _normalize_config_keys(dict(config)))
    return deepcopy(_CONFIG_CACHE)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _drawdown_magnitude(value: Any) -> float:
    numeric = _safe_float(value)
    return abs(numeric) if numeric < 0 else max(0.0, numeric)


def _normalize_probability(value: Any) -> float:
    return _clip(_safe_float(value))


def _normalize_preference(value: str | None) -> str:
    raw = (value or "balanced").strip().lower()
    aliases = {
        "conservative_cn": "conservative",
        "conservative": "conservative",
        "balanced_cn": "balanced",
        "balanced": "balanced",
        "aggressive_cn": "aggressive",
        "aggressive_cn2": "aggressive",
        "aggressive": "aggressive",
    }
    return aliases.get(raw, "balanced")


def _normalize_style(value: str | None) -> str:
    raw = (value or "short_mid").strip().lower()
    aliases = {
        "short_cn": "short",
        "short": "short",
        "short_mid_cn": "short_mid",
        "short_mid": "short_mid",
        "mid_cn": "mid",
        "mid": "mid",
    }
    return aliases.get(raw, "short_mid")


def _legacy_score_from_utility(value: float) -> float:
    return _clip(0.5 + value)


def _mean(values: Iterable[float]) -> float:
    seq = list(values)
    return float(np.mean(seq)) if seq else 0.0


def _resolve_context(
    context: Mapping[str, Any] | None,
    *,
    is_holding: bool,
    holding_days: int,
    risk_preference: str,
    strategy_style: str,
) -> dict[str, Any]:
    resolved = {
        "is_holding": bool(is_holding),
        "holding_days": int(holding_days),
        "entry_price": None,
        "risk_preference": risk_preference,
        "strategy_style": strategy_style,
        "position_size_hint": None,
    }
    if context:
        resolved.update(dict(context))
    resolved["is_holding"] = bool(resolved.get("is_holding"))
    resolved["holding_days"] = max(0, int(_safe_float(resolved.get("holding_days"))))
    resolved["risk_preference"] = _normalize_preference(str(resolved.get("risk_preference") or risk_preference))
    resolved["strategy_style"] = _normalize_style(str(resolved.get("strategy_style") or strategy_style))
    return resolved


def _resolve_model_version(record: Mapping[str, Any], metadata: Mapping[str, Any] | None) -> str:
    if metadata and metadata.get("model_version"):
        return str(metadata["model_version"])
    checkpoint_path = str(record.get("checkpoint_path") or "")
    if checkpoint_path:
        parts = checkpoint_path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            return parts[-2]
    return "unknown_model"


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _detect_market_regime(record: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    model_prob = _safe_float(record.get("market_regime_prob"), 0.5)
    feature_regime_score = _safe_float(record.get("market_regime_score"))
    risk_on_flag = _safe_float(record.get("risk_on_flag"))
    risk_off_flag = _safe_float(record.get("risk_off_flag"))
    limit_ratio = _safe_float(record.get("limit_count_ratio"))
    hotness_spread = _safe_float(record.get("sector_hotness_spread"))
    market_vol = max(0.0, _safe_float(record.get("market_volatility_5")))
    composite = (
        0.40 * ((model_prob - 0.5) * 2.0)
        + 0.22 * feature_regime_score
        + 0.15 * (risk_on_flag - risk_off_flag)
        + 0.13 * limit_ratio
        + 0.10 * math.tanh(hotness_spread / 0.03 if hotness_spread else 0.0)
        - 0.12 * math.tanh(market_vol / 0.04 if market_vol else 0.0)
    )
    if risk_off_flag >= 0.5 or model_prob <= 0.42 or feature_regime_score <= -0.25 or composite <= -0.18:
        regime = "risk_off"
    elif risk_on_flag >= 0.5 or model_prob >= 0.58 or feature_regime_score >= 0.25 or composite >= 0.18:
        regime = "risk_on"
    else:
        regime = "neutral"
    return regime, {
        "market_regime_prob": model_prob,
        "market_regime_score_model": (model_prob - 0.5) * 2.0,
        "market_regime_score_feature": feature_regime_score,
        "market_regime_score_composite": composite,
    }


def _bigloss_prob(record: Mapping[str, Any], horizon: int) -> float:
    if record.get(f"bigloss_prob_{horizon}") is not None:
        return _normalize_probability(record.get(f"bigloss_prob_{horizon}"))
    if horizon in (3, 5, 10):
        return _normalize_probability(record.get("bigloss_prob_5"))
    return _normalize_probability(record.get("bigloss_prob_20"))


def _compute_horizon_utilities(
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    preference = str(context["risk_preference"])
    weights = config["utility_weights"][preference]
    rank_score = _normalize_probability(record.get("rank_score_pred"))
    utility_map: dict[int, float] = {}
    component_map: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        p_win = _normalize_probability(record.get(f"p_win_prob_{horizon}"))
        ret_mu = _safe_float(record.get(f"ret_mu_pred_{horizon}"))
        risk_dd = _drawdown_magnitude(record.get(f"risk_dd_pred_{horizon}"))
        bigloss = _bigloss_prob(record, horizon)
        upside = _safe_float(record.get(f"upside_pred_{horizon}"), max(ret_mu, 0.0))
        sigma = max(0.0, _safe_float(record.get(f"ret_sigma_pred_{horizon}")))

        ret_utility = weights["ret"] * math.tanh(ret_mu / max(float(config["return_scales"][horizon]), EPS))
        win_utility = weights["p_win"] * ((p_win - 0.5) * 2.0)
        rank_utility = weights["rank"] * ((rank_score - 0.5) * 2.0)
        upside_bonus = weights["upside"] * math.tanh(upside / max(float(config["upside_scales"][horizon]), EPS))
        drawdown_penalty = weights["drawdown_penalty"] * (
            risk_dd / max(float(config["drawdown_scales"][horizon]), EPS)
        ) ** float(config["utility_shape"]["drawdown_alpha"])
        bigloss_penalty = weights["bigloss_penalty"] * _softplus(
            float(config["utility_shape"]["bigloss_beta"]) * (bigloss - float(config["utility_shape"]["bigloss_tau"]))
        )
        uncertainty_penalty = weights["uncertainty_penalty"] * math.tanh(
            sigma / max(float(config["sigma_scales"][horizon]), EPS)
        )
        utility = (
            ret_utility
            + win_utility
            + rank_utility
            + upside_bonus
            - drawdown_penalty
            - bigloss_penalty
            - uncertainty_penalty
        )
        utility_map[horizon] = utility
        component_map[horizon] = {
            "p_win": p_win,
            "ret_mu": ret_mu,
            "risk_dd": risk_dd,
            "bigloss_prob": bigloss,
            "rank_score": rank_score,
            "upside": upside,
            "sigma": sigma,
            "R": ret_utility,
            "W": win_utility,
            "Q": rank_utility,
            "X": upside_bonus,
            "D": drawdown_penalty,
            "B": bigloss_penalty,
            "V": uncertainty_penalty,
            "U": utility,
            "horizon_confidence": _clip(0.40 + 0.35 * abs(p_win - 0.5) * 2.0 + 0.25 * max(0.0, 1.0 - bigloss)),
        }
    return utility_map, component_map


def _time_weights(
    context: Mapping[str, Any],
    market_regime: str,
    config: Mapping[str, Any],
) -> dict[int, float]:
    base = dict(config["time_weights"][str(context["strategy_style"])])
    if str(context["risk_preference"]) == "aggressive" or market_regime == "risk_on":
        multipliers = {3: 1.18, 5: 1.10, 10: 1.02, 20: 0.88, 40: 0.78}
    elif str(context["risk_preference"]) == "conservative" or market_regime == "risk_off":
        multipliers = {3: 0.80, 5: 0.90, 10: 1.08, 20: 1.18, 40: 1.18}
    else:
        multipliers = {horizon: 1.0 for horizon in HORIZONS}
    if bool(context.get("is_holding")):
        hold_days = int(context.get("holding_days") or 0)
        if hold_days >= 10:
            multipliers = {
                3: multipliers[3] * 0.90,
                5: multipliers[5] * 0.95,
                10: multipliers[10],
                20: multipliers[20] * 1.08,
                40: multipliers[40] * 1.10,
            }
    adjusted = {horizon: float(base[horizon]) * multipliers[horizon] for horizon in HORIZONS}
    total = sum(adjusted.values())
    return {horizon: adjusted[horizon] / max(total, EPS) for horizon in HORIZONS}


def _compute_consistency(utility_map: Mapping[int, float]) -> float:
    values = np.asarray([utility_map[horizon] for horizon in HORIZONS], dtype=np.float64)
    positive = int(np.sum(values > 0.03))
    negative = int(np.sum(values < -0.03))
    direction_consistency = max(positive, negative, len(HORIZONS) - positive - negative) / len(HORIZONS)
    magnitude_consistency = 1.0 - min(1.0, float(values.std(ddof=0)) / 0.55)
    return _clip(0.6 * direction_consistency + 0.4 * magnitude_consistency)


def _classify_conflict(
    utility_map: Mapping[int, float],
    record: Mapping[str, Any],
    component_map: Mapping[int, Mapping[str, float]],
) -> tuple[str | None, str | None, float]:
    short_utility = _mean([utility_map[3], utility_map[5]])
    mid_utility = _mean([utility_map[10], utility_map[20]])
    long_utility = _mean([utility_map[20], utility_map[40]])
    recent_5d_return = _safe_float(record.get("ret_5"))
    risk_dd_5 = _drawdown_magnitude(record.get("risk_dd_pred_5"))
    bigloss_5 = component_map[5]["bigloss_prob"]

    if short_utility >= 0.18 and long_utility <= -0.12:
        if recent_5d_return > 0.18:
            return "短多长空", "overheated", 0.45
        if risk_dd_5 > 0.08 or bigloss_5 > 0.45:
            return "短多长空", "malignant", 0.60
        return "短多长空", "benign_mismatch", 0.22
    if short_utility <= -0.12 and long_utility >= 0.18:
        return "短空长多", "reversal", 0.30
    if short_utility <= -0.10 and mid_utility >= 0.20:
        return "短空中多", "reversal", 0.24
    if short_utility >= 0.10 and mid_utility >= 0.12:
        return None, "aligned_positive", 0.0
    return None, None, 0.0


def _aggregate_horizon_utilities(
    record: Mapping[str, Any],
    utility_map: Mapping[int, float],
    component_map: Mapping[int, Mapping[str, float]],
    market_regime: str,
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    weights = _time_weights(context, market_regime, config)
    weighted_utility = sum(weights[horizon] * utility_map[horizon] for horizon in HORIZONS)
    consistency = _compute_consistency(utility_map)
    conflict_state, conflict_type, conflict_penalty = _classify_conflict(utility_map, record, component_map)
    u_final = weighted_utility - float(config["utility_shape"]["conflict_lambda"]) * conflict_penalty
    best_horizon = max(HORIZONS, key=lambda horizon: utility_map[horizon])
    ranking = sorted(
        [{"horizon": f"k={horizon}", "days": horizon, "score": utility_map[horizon]} for horizon in HORIZONS],
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "U_final": u_final,
        "weighted_utility": weighted_utility,
        "consistency_score": consistency,
        "conflict_state": conflict_state,
        "conflict_type": conflict_type,
        "conflict_penalty": conflict_penalty,
        "best_horizon": best_horizon,
        "best_horizon_score": utility_map[best_horizon],
        "horizon_ranking": ranking,
        "time_weights": weights,
        "short_term_view": _utility_view(_mean([utility_map[3], utility_map[5]])),
        "mid_term_view": _utility_view(_mean([utility_map[10], utility_map[20], utility_map[40]])),
    }


def _utility_view(value: float) -> str:
    if value >= 0.35:
        return "strong_bullish"
    if value >= 0.15:
        return "bullish"
    if value > -0.10:
        return "neutral"
    if value > -0.30:
        return "bearish"
    return "strong_bearish"


def _holding_stage(holding_days: int, best_horizon: int) -> tuple[float, str]:
    progress = holding_days / max(float(best_horizon), 1.0)
    if progress < 0.4:
        return progress, "early_hold"
    if progress < 0.9:
        return progress, "mid_hold"
    if progress <= 1.2:
        return progress, "late_hold"
    return progress, "over_hold"


def _risk_score(
    record: Mapping[str, Any],
    component_map: Mapping[int, Mapping[str, float]],
    market_regime: str,
    config: Mapping[str, Any],
) -> tuple[float, str, dict[str, float]]:
    drawdown_risk = _clip(max(component_map[5]["risk_dd"] / 0.10, component_map[10]["risk_dd"] / 0.14))
    bigloss_risk = _clip(max(component_map[5]["bigloss_prob"], component_map[10]["bigloss_prob"]))
    market_vol = _safe_float(record.get("market_volatility_5"))
    market_risk = _clip((market_vol / 0.05) + (0.20 if market_regime == "risk_off" else 0.0))
    amount = _safe_float(record.get("amount_ma5"), _safe_float(record.get("avg_amount_5")))
    liquidity_risk = _clip((20_000_000 - amount) / 20_000_000) if amount > 0 else 0.35
    missing = _safe_float(record.get("feature_missing_rate"))
    coverage = _safe_float(record.get("sample_coverage_score"), 1.0)
    drift = _safe_float(record.get("model_drift_score"), 1.0 if record.get("model_calibration_drift") else 0.0)
    data_quality_risk = _clip(0.5 * missing / 0.35 + 0.3 * (1.0 - coverage) + 0.2 * drift)
    recent_5d_return = _safe_float(record.get("ret_5"), _safe_float(record.get("pct_chg_5d")))
    pct_chg_1d = _safe_float(record.get("pct_chg_1d"), _safe_float(record.get("pct_chg")) / 100.0)
    overheat_risk = _clip(max(recent_5d_return / 0.25, pct_chg_1d / 0.095))
    weights = config["risk_score_weights"]
    score = _clip(
        float(weights["drawdown"]) * drawdown_risk
        + float(weights["bigloss"]) * bigloss_risk
        + float(weights["market"]) * market_risk
        + float(weights["liquidity"]) * liquidity_risk
        + float(weights["data_quality"]) * data_quality_risk
        + float(weights["overheat"]) * overheat_risk
    )
    if score < float(config["risk_thresholds"]["low"]):
        level = "low"
    elif score < float(config["risk_thresholds"]["medium"]):
        level = "medium"
    elif score < float(config["risk_thresholds"]["high"]):
        level = "high"
    else:
        level = "severe"
    return score, level, {
        "drawdown_risk": drawdown_risk,
        "bigloss_risk": bigloss_risk,
        "market_risk": market_risk,
        "liquidity_risk": liquidity_risk,
        "data_quality_risk": data_quality_risk,
        "overheat_risk": overheat_risk,
    }


def _compute_confidence(summary: Mapping[str, Any], risk_score: float, risk_review: Mapping[str, Any] | None = None) -> float:
    direction_strength = min(1.0, abs(float(summary["U_final"])) / 0.55)
    confidence = 0.38 + 0.30 * float(summary["consistency_score"]) + 0.22 * direction_strength - 0.18 * risk_score
    if float(summary["conflict_penalty"]) > 0:
        confidence -= 0.08 * float(summary["conflict_penalty"]) / 0.60
    if risk_review and risk_review.get("downgraded"):
        confidence -= 0.08
    return _clip(confidence, 0.05, 0.99)


def _base_not_holding_action(
    summary: Mapping[str, Any],
    confidence: float,
    risk_level: str,
    market_regime: str,
    component_map: Mapping[int, Mapping[str, float]],
    config: Mapping[str, Any],
) -> str:
    u_final = float(summary["U_final"])
    consistency = float(summary["consistency_score"])
    conflict_penalty = float(summary["conflict_penalty"])
    thresholds = config["action_thresholds"][market_regime]
    max_bigloss = max(component_map[5]["bigloss_prob"], component_map[10]["bigloss_prob"])
    if risk_level == "severe" or u_final <= thresholds["avoid"] - 0.18:
        return "STRONG_AVOID"
    if risk_level == "high" or u_final <= thresholds["avoid"]:
        return "AVOID"
    if (
        u_final >= thresholds["strong_buy"]
        and confidence >= 0.75
        and consistency >= 0.65
        and risk_level == "low"
        and max_bigloss < 0.35
        and conflict_penalty < 0.20
    ):
        return "STRONG_BUY"
    if u_final >= thresholds["buy"] and confidence >= 0.55 and risk_level in {"low", "medium"}:
        return "BUY"
    return "WATCH"


def _base_holding_action(
    summary: Mapping[str, Any],
    confidence: float,
    risk_level: str,
    market_regime: str,
    hold_days: int,
    component_map: Mapping[int, Mapping[str, float]],
    config: Mapping[str, Any],
) -> str:
    del market_regime, config
    u_final = float(summary["U_final"])
    best_horizon = int(summary["best_horizon"])
    hold_progress, stage = _holding_stage(hold_days, best_horizon)
    risk_dd_5 = component_map[5]["risk_dd"]
    bigloss_5 = component_map[5]["bigloss_prob"]
    if (
        risk_level == "severe"
        or (risk_dd_5 >= 0.08 and (bigloss_5 >= 0.50 or u_final <= -0.20))
        or (u_final <= -0.35 and risk_level == "high")
    ):
        return "STOP_LOSS"
    if u_final <= -0.18 or stage == "over_hold":
        return "SELL"
    if risk_level == "high" or float(summary["conflict_penalty"]) >= 0.40 or stage == "late_hold":
        return "REDUCE"
    if (
        u_final >= 0.50
        and confidence >= 0.72
        and risk_level == "low"
        and market_regime != "risk_off"
        and hold_progress < 0.65
        and bigloss_5 < 0.30
        and summary.get("conflict_type") != "overheated"
    ):
        return "ADD_POSITION"
    return "HOLD"


def _downgrade_one_step(action: str) -> str:
    if action in BUY_PATH_ORDER:
        return BUY_PATH_ORDER[min(BUY_PATH_ORDER.index(action) + 1, len(BUY_PATH_ORDER) - 1)]
    if action in HOLD_PATH_ORDER:
        return HOLD_PATH_ORDER[min(HOLD_PATH_ORDER.index(action) + 1, len(HOLD_PATH_ORDER) - 1)]
    return action


def _downgrade_to_watch(action: str, is_holding: bool) -> str:
    if is_holding:
        return "HOLD" if action == "ADD_POSITION" else action
    if action in {"STRONG_BUY", "BUY", "RECOMMEND_BUY"}:
        return "WATCH"
    return action


def _force_sell_side(is_holding: bool) -> str:
    return "STOP_LOSS" if is_holding else "AVOID"


def _apply_risk_controls(
    record: Mapping[str, Any],
    *,
    original_action: str,
    market_regime: str,
    is_holding: bool,
    summary: Mapping[str, Any],
    component_map: Mapping[int, Mapping[str, float]],
    risk_score: float,
    risk_level: str,
    risk_components: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    final_action = original_action
    risk_flags: list[str] = []
    warnings: list[str] = []
    hard_blocks: list[str] = []
    hard_downgrades: list[str] = []
    soft_penalties: list[str] = []
    thresholds = config["risk_thresholds"]

    def add(kind: str, rule: str, message: str) -> None:
        if rule not in risk_flags:
            risk_flags.append(rule)
        warnings.append(message)
        if kind == "hard_block" and rule not in hard_blocks:
            hard_blocks.append(rule)
        elif kind == "hard_downgrade" and rule not in hard_downgrades:
            hard_downgrades.append(rule)
        elif kind == "soft_penalty" and rule not in soft_penalties:
            soft_penalties.append(rule)

    list_days = int(_safe_float(record.get("list_days")))
    is_st = bool(record.get("is_st"))
    is_suspended = bool(record.get("is_suspended"))
    amount = _safe_float(record.get("amount_ma5"), _safe_float(record.get("avg_amount_5")))
    feature_missing_rate = _safe_float(record.get("feature_missing_rate"))
    sample_coverage_score = _safe_float(record.get("sample_coverage_score"), 1.0)
    train_symbol_samples = int(_safe_float(record.get("train_symbol_samples")))
    train_industry_pctile = _safe_float(record.get("train_industry_pctile"))
    industry_risk_pctile = _safe_float(record.get("industry_risk_dd_5_pctile"))
    model_drift_score = _safe_float(record.get("model_drift_score"), 1.0 if record.get("model_calibration_drift") else 0.0)
    down_limit_ratio = _safe_float(record.get("down_limit_ratio"))
    major_index_drop = abs(min(_safe_float(record.get("major_index_ret_1")), 0.0))
    market_volatility_5 = _safe_float(record.get("market_volatility_5"))
    ranker_quality_score = _safe_float(record.get("ranker_quality_score"), 0.0 if record.get("model_ranking_weak") else 1.0)
    risk_dd_5 = component_map[5]["risk_dd"]
    risk_dd_10 = component_map[10]["risk_dd"]
    bigloss_5 = component_map[5]["bigloss_prob"]
    p_win_5 = component_map[5]["p_win"]
    recent_5d_return = _safe_float(record.get("ret_5"), _safe_float(record.get("pct_chg_5d")))
    if is_st:
        add("hard_block", "R0-01", "ST stock hard block.")
        final_action = _force_sell_side(is_holding)
    if is_suspended:
        add("hard_block", "R0-02", "Suspended stock hard block.")
        final_action = _force_sell_side(is_holding)
    if list_days and list_days < int(thresholds["list_days_min"]):
        add("hard_block", "R0-03", f"List days {list_days} below minimum {int(thresholds['list_days_min'])}.")
        final_action = _force_sell_side(is_holding)
    if amount and amount < float(thresholds["liquidity_hard"]):
        add("hard_block", "R1-06", f"Average amount is too low: {amount / 1e6:.1f} million.")
        final_action = _downgrade_to_watch(final_action, is_holding)
    elif amount and amount < float(thresholds["liquidity_warn"]):
        add("soft_penalty", "R1-05", f"Average amount is weak: {amount / 1e6:.0f} million.")
    if feature_missing_rate > float(thresholds["feature_missing_hard"]):
        add("hard_block", "R4-04", f"Feature missing rate is too high: {feature_missing_rate:.1%}.")
        final_action = _force_sell_side(is_holding)
    elif feature_missing_rate > float(thresholds["feature_missing_warn"]):
        add("hard_downgrade", "R4-04", f"Feature missing rate is elevated: {feature_missing_rate:.1%}.")
        final_action = _downgrade_to_watch(final_action, is_holding)
    if model_drift_score >= float(thresholds["model_drift_hard"]):
        add("hard_block", "R4-01", "Model drift score is too high.")
        final_action = _force_sell_side(is_holding)
    elif model_drift_score > 0:
        add("soft_penalty", "R4-01", "Model calibration may be drifting; use as reference only.")
    if (down_limit_ratio > 0.10 or major_index_drop > 0.05) and final_action in BUY_ACTIONS:
        add("hard_block", "R3-02", "Market is in a high-risk limit-down regime.")
        final_action = _downgrade_to_watch(final_action, is_holding)
    if bigloss_5 >= float(thresholds["hard_bigloss"]):
        add("hard_block", "R1-02", f"5-day big loss probability is too high: {bigloss_5:.2%}.")
        final_action = "STOP_LOSS" if is_holding else "STRONG_AVOID"
    elif bigloss_5 >= float(thresholds["buy_bigloss"]) and final_action in BUY_ACTIONS:
        add("hard_downgrade", "R1-02", f"5-day big loss probability is elevated: {bigloss_5:.2%}.")
        final_action = _downgrade_to_watch(final_action, is_holding)
    if (risk_dd_5 > 0.08 or risk_dd_10 > 0.12) and final_action in BUY_ACTIONS:
        add("hard_downgrade", "R1-01", f"Predicted drawdown is high: risk_dd_5={risk_dd_5:.2%}, risk_dd_10={risk_dd_10:.2%}.")
        final_action = _downgrade_to_watch(final_action, is_holding)
    if risk_dd_5 > 0.15:
        add("hard_block", "R1-03", f"5-day predicted drawdown exceeds extreme threshold: {risk_dd_5:.2%}.")
        final_action = _force_sell_side(is_holding)
    if p_win_5 < 0.45 and final_action in BUY_ACTIONS:
        add("hard_downgrade", "R1-04", f"5-day win probability is below buy floor: {p_win_5:.2%}.")
        final_action = _downgrade_to_watch(final_action, is_holding)
    elif p_win_5 < 0.35:
        add("soft_penalty", "R1-04", f"5-day win probability is very low: {p_win_5:.2%}.")
    if recent_5d_return > 0.25 and final_action in BUY_ACTIONS:
        add("hard_downgrade", "R1-07", f"Recent 5-day return is overheated: {recent_5d_return:.2%}.")
        final_action = _downgrade_one_step(final_action)
    if market_regime == "risk_off" and market_volatility_5 >= 0.035 and final_action in BUY_ACTIONS:
        add("hard_downgrade", "R3-01", f"Market regime is risk_off with volatility {market_volatility_5:.2%}.")
        final_action = _downgrade_one_step(final_action)
    if ranker_quality_score < 0.35:
        add("soft_penalty", "R4-02", "Recent ranking quality is weak.")
    if sample_coverage_score < 0.45 or (train_symbol_samples and train_symbol_samples < 100):
        add("hard_downgrade", "R4-03", "Training sample coverage is insufficient.")
        final_action = _downgrade_one_step(final_action) if final_action in BUY_ACTIONS else final_action
    elif 0.0 < train_industry_pctile < 0.10:
        add("soft_penalty", "R4-03", f"Industry training coverage percentile is low: {train_industry_pctile:.0%}.")
    if industry_risk_pctile > 0.80:
        add("soft_penalty", "R5-01", f"Industry drawdown percentile is high: {industry_risk_pctile:.0%}.")
    if float(summary["conflict_penalty"]) > 0:
        add("soft_penalty", "R2-01", f"Cross-horizon conflict detected: {summary.get('conflict_type') or 'unknown'}.")

    return {
        "passed": not hard_blocks,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_components": dict(risk_components),
        "risk_flags": risk_flags,
        "warnings": warnings,
        "risk_warnings": warnings,
        "hard_blocks": hard_blocks,
        "hard_downgrades": hard_downgrades,
        "soft_penalties": soft_penalties,
        "blocked_rules": hard_blocks + hard_downgrades,
        "original_action": original_action,
        "final_action": final_action,
        "downgraded": final_action != original_action,
    }


def _position_hint(action: str, risk_level: str, confidence: float) -> str:
    if action == "STRONG_BUY" and risk_level == "low" and confidence >= 0.78:
        return "full"
    if action in {"STRONG_BUY", "BUY", "ADD_POSITION"}:
        return "half" if risk_level in {"low", "medium"} else "light"
    if action == "REDUCE":
        return "reduce"
    if action in {"SELL", "STOP_LOSS", "AVOID", "STRONG_AVOID"}:
        return "none"
    return "light"


def _build_reasons(
    record: Mapping[str, Any],
    utility_map: Mapping[int, float],
    component_map: Mapping[int, Mapping[str, float]],
    summary: Mapping[str, Any],
    market_regime: str,
    action: str,
    risk_review: Mapping[str, Any],
) -> list[str]:
    best_horizon = int(summary["best_horizon"])
    p_win_5 = component_map[5]["p_win"]
    ret_mu_5 = component_map[5]["ret_mu"]
    risk_dd_5 = component_map[5]["risk_dd"]
    bigloss_5 = component_map[5]["bigloss_prob"]
    rank_score = component_map[5]["rank_score"]
    reasons = [
        f"5d p_win={p_win_5:.2%}, ret_mu={ret_mu_5:.2%}, risk_dd={risk_dd_5:.2%}, bigloss={bigloss_5:.2%}.",
        f"Utility U_final={float(summary['U_final']):.3f}, U_3={utility_map[3]:.3f}, U_5={utility_map[5]:.3f}, U_10={utility_map[10]:.3f}, U_20={utility_map[20]:.3f}, U_40={utility_map[40]:.3f}.",
        f"Consistency={float(summary['consistency_score']):.3f}, conflict_penalty={float(summary['conflict_penalty']):.3f}, market_regime={market_regime}, best_horizon=k={best_horizon}.",
        f"Rank score={rank_score:.3f}; base action={ACTION_META[action]['action_cn']}.",
    ]
    if summary.get("conflict_state"):
        reasons.append(f"Cross-horizon conflict: {summary['conflict_state']}, type={summary.get('conflict_type')}.")
    else:
        reasons.append("No obvious cross-horizon conflict.")
    for message in risk_review["warnings"]:
        reasons.append(f"Risk note: {message}")
    return reasons


def _build_model_output(record: Mapping[str, Any]) -> dict[str, float]:
    output = {"rank_score": _normalize_probability(record.get("rank_score_pred"))}
    for horizon in HORIZONS:
        output[f"p_win_{horizon}"] = _normalize_probability(record.get(f"p_win_prob_{horizon}"))
        output[f"ret_mu_{horizon}"] = _safe_float(record.get(f"ret_mu_pred_{horizon}"))
        output[f"risk_dd_{horizon}"] = _drawdown_magnitude(record.get(f"risk_dd_pred_{horizon}"))
        output[f"bigloss_{horizon}"] = _bigloss_prob(record, horizon)
        if record.get(f"upside_pred_{horizon}") is not None:
            output[f"upside_{horizon}"] = _safe_float(record.get(f"upside_pred_{horizon}"))
        if record.get(f"ret_sigma_pred_{horizon}") is not None:
            output[f"ret_sigma_{horizon}"] = _safe_float(record.get(f"ret_sigma_pred_{horizon}"))
    return output


def evaluate_stock_decision(
    record: Mapping[str, Any],
    *,
    is_holding: bool = False,
    holding_days: int = 0,
    risk_preference: str = "balanced",
    strategy_style: str = "short_mid",
    context: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    horizon_probs = {
        horizon: _normalize_probability(record.get(f"p_win_prob_{horizon}"))
        for horizon in HORIZONS
    }
    best_horizon = max(HORIZONS, key=lambda horizon: horizon_probs[horizon])
    best_p_win = float(horizon_probs[best_horizon])
    final_action = "BUY" if best_p_win > 0.50 else "AVOID"
    action_meta = ACTION_META[final_action]
    model_output = _build_model_output(record)
    model_output.update({f"p_win_{horizon}": float(horizon_probs[horizon]) for horizon in HORIZONS})
    scores = {
        "U_final": best_p_win,
        "U_3": float(horizon_probs[3]),
        "U_5": float(horizon_probs[5]),
        "U_10": float(horizon_probs[10]),
        "U_20": float(horizon_probs[20]),
        "U_40": float(horizon_probs[40]),
        "S_final": best_p_win,
        "S_3": float(horizon_probs[3]),
        "S_5": float(horizon_probs[5]),
        "S_10": float(horizon_probs[10]),
        "S_20": float(horizon_probs[20]),
        "S_40": float(horizon_probs[40]),
        "consistency": 1.0,
        "consistency_score": 1.0,
        "conflict_state": "simple_p_win_rule",
        "conflict_type": "none",
        "conflict_penalty": 0.0,
        "confidence_score": best_p_win,
        "risk_score": 0.0,
    }
    risk_review = {
        "passed": final_action == "BUY",
        "original_action": final_action,
        "final_action": final_action,
        "downgraded": False,
        "risk_level": "low",
        "risk_score": 0.0,
        "risk_flags": [],
        "risk_warnings": [],
        "blocked_rules": [],
        "hard_blocks": [],
        "hard_downgrades": [],
        "soft_penalties": [],
        "risk_components": {},
    }
    reasons = [
        (
            f"Simple rule: highest win probability is {best_horizon}d at {best_p_win:.2%}; "
            f"{'buy because it is above 50%' if final_action == 'BUY' else 'avoid because no horizon is above 50%'}."
        )
    ]
    return {
        "symbol": str(record.get("symbol") or ""),
        "symbol_name": str(record.get("name") or record.get("symbol_name") or ""),
        "trade_date": str(record.get("trade_date") or ""),
        "decision": {
            "action": final_action,
            "action_cn": action_meta["action_cn"],
            "confidence": best_p_win,
            "priority": action_meta["priority"],
            "path": action_meta["path"],
            "best_horizon": f"k={best_horizon}",
            "suggested_hold_days": best_horizon,
            "position_hint": "full_or_add_position" if final_action == "BUY" else "avoid_new_position",
            "holding_stage": None,
            "hold_progress": 0.0,
        },
        "utility": {
            "U_3": float(horizon_probs[3]),
            "U_5": float(horizon_probs[5]),
            "U_10": float(horizon_probs[10]),
            "U_20": float(horizon_probs[20]),
            "U_40": float(horizon_probs[40]),
            "U_final": best_p_win,
            "consistency_score": 1.0,
            "conflict_penalty": 0.0,
            "confidence_score": best_p_win,
        },
        "horizon_analysis": {
            "best_horizon": f"k={best_horizon}",
            "best_horizon_score": best_p_win,
            "horizon_ranking": [
                {"horizon": horizon, "utility": float(prob)}
                for horizon, prob in sorted(horizon_probs.items(), key=lambda item: item[1], reverse=True)
            ],
            "short_term_view": "positive" if max(horizon_probs[horizon] for horizon in (3, 5, 10)) > 0.50 else "negative",
            "mid_term_view": "positive" if max(horizon_probs[horizon] for horizon in (20, 40)) > 0.50 else "negative",
            "conflict_type": "none",
        },
        "scores": scores,
        "model_output": model_output,
        "market_regime": "neutral",
        "market_regime_detail": {},
        "risk_flags": [],
        "risk_review": risk_review,
        "reasons": reasons,
        "metadata": {
            "model_version": _resolve_model_version(record, metadata),
            "feature_version": str(
                (metadata or {}).get("feature_version")
                or record.get("feature_version")
                or record.get("data_version")
                or "unknown_feature"
            ),
            "engine_version": ENGINE_VERSION,
            "risk_preference": risk_preference,
            "strategy_style": strategy_style,
            "inference_ts": str((metadata or {}).get("inference_ts") or _now_iso()),
        },
    }

    cfg = _load_config(config)
    resolved_context = _resolve_context(
        context,
        is_holding=is_holding,
        holding_days=holding_days,
        risk_preference=risk_preference,
        strategy_style=strategy_style,
    )
    market_regime, market_details = _detect_market_regime(record)
    utility_map, component_map = _compute_horizon_utilities(record, resolved_context, cfg)
    summary = _aggregate_horizon_utilities(record, utility_map, component_map, market_regime, resolved_context, cfg)
    risk_score, risk_level, risk_components = _risk_score(record, component_map, market_regime, cfg)
    provisional_confidence = _compute_confidence(summary, risk_score)
    if bool(resolved_context["is_holding"]):
        original_action = _base_holding_action(
            summary,
            provisional_confidence,
            risk_level,
            market_regime,
            int(resolved_context["holding_days"]),
            component_map,
            cfg,
        )
    else:
        original_action = _base_not_holding_action(
            summary,
            provisional_confidence,
            risk_level,
            market_regime,
            component_map,
            cfg,
        )
    risk_review = _apply_risk_controls(
        record,
        original_action=original_action,
        market_regime=market_regime,
        is_holding=bool(resolved_context["is_holding"]),
        summary=summary,
        component_map=component_map,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_components=risk_components,
        config=cfg,
    )
    confidence = _compute_confidence(summary, risk_score, risk_review)
    final_action = str(risk_review["final_action"])
    best_horizon = int(summary["best_horizon"])
    hold_progress, holding_stage = _holding_stage(int(resolved_context["holding_days"]), best_horizon)
    action_meta = ACTION_META[final_action]
    reasons = _build_reasons(record, utility_map, component_map, summary, market_regime, original_action, risk_review)

    scores = {
        "U_final": float(summary["U_final"]),
        "U_3": utility_map[3],
        "U_5": utility_map[5],
        "U_10": utility_map[10],
        "U_20": utility_map[20],
        "U_40": utility_map[40],
        "S_final": _legacy_score_from_utility(float(summary["U_final"])),
        "S_3": _legacy_score_from_utility(utility_map[3]),
        "S_5": _legacy_score_from_utility(utility_map[5]),
        "S_10": _legacy_score_from_utility(utility_map[10]),
        "S_20": _legacy_score_from_utility(utility_map[20]),
        "S_40": _legacy_score_from_utility(utility_map[40]),
        "consistency": float(summary["consistency_score"]),
        "consistency_score": float(summary["consistency_score"]),
        "conflict_state": summary["conflict_state"],
        "conflict_type": summary["conflict_type"],
        "conflict_penalty": float(summary["conflict_penalty"]),
        "confidence_score": confidence,
        "risk_score": risk_score,
    }

    return {
        "symbol": str(record.get("symbol") or ""),
        "symbol_name": str(record.get("name") or record.get("symbol_name") or ""),
        "trade_date": str(record.get("trade_date") or ""),
        "decision": {
            "action": final_action,
            "action_cn": action_meta["action_cn"],
            "confidence": confidence,
            "priority": action_meta["priority"],
            "path": action_meta["path"],
            "best_horizon": f"k={best_horizon}",
            "suggested_hold_days": best_horizon,
            "position_hint": _position_hint(final_action, risk_level, confidence),
            "holding_stage": holding_stage if bool(resolved_context["is_holding"]) else None,
            "hold_progress": hold_progress if bool(resolved_context["is_holding"]) else 0.0,
        },
        "utility": {
            "U_3": utility_map[3],
            "U_5": utility_map[5],
            "U_10": utility_map[10],
            "U_20": utility_map[20],
            "U_40": utility_map[40],
            "U_final": float(summary["U_final"]),
            "consistency_score": float(summary["consistency_score"]),
            "conflict_penalty": float(summary["conflict_penalty"]),
            "confidence_score": confidence,
        },
        "horizon_analysis": {
            "best_horizon": f"k={best_horizon}",
            "best_horizon_score": float(summary["best_horizon_score"]),
            "horizon_ranking": summary["horizon_ranking"],
            "short_term_view": summary["short_term_view"],
            "mid_term_view": summary["mid_term_view"],
            "conflict_type": summary["conflict_type"],
        },
        "scores": scores,
        "model_output": _build_model_output(record),
        "market_regime": market_regime,
        "market_regime_detail": market_details,
        "risk_flags": risk_review["risk_flags"],
        "risk_review": risk_review,
        "reasons": reasons,
        "metadata": {
            "model_version": _resolve_model_version(record, metadata),
            "feature_version": str(
                (metadata or {}).get("feature_version")
                or record.get("feature_version")
                or record.get("data_version")
                or "unknown_feature"
            ),
            "engine_version": ENGINE_VERSION,
            "risk_preference": resolved_context["risk_preference"],
            "strategy_style": resolved_context["strategy_style"],
            "inference_ts": str((metadata or {}).get("inference_ts") or _now_iso()),
        },
    }


def _is_candidate_eligible(
    record: Mapping[str, Any],
    decision_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    del record, config
    return str(decision_result["decision"].get("action") or "") in {"STRONG_BUY", "BUY", "RECOMMEND_BUY"}


def _upside_bonus(record: Mapping[str, Any]) -> float:
    values = [_safe_float(record.get(f"upside_pred_{horizon}"), _safe_float(record.get(f"ret_mu_pred_{horizon}"))) for horizon in HORIZONS]
    return math.tanh(max(values) / 0.08) if values else 0.0


def _selection_utility(record: Mapping[str, Any], decision_result: Mapping[str, Any]) -> float:
    risk_components = decision_result["risk_review"]["risk_components"]
    return (
        0.38 * _safe_float(decision_result["scores"]["U_final"])
        + 0.18 * ((_normalize_probability(record.get("rank_score_pred")) - 0.5) * 2.0)
        + 0.16 * _safe_float(decision_result["scores"]["consistency_score"])
        + 0.14 * _upside_bonus(record)
        - 0.08 * _safe_float(risk_components.get("drawdown_risk"))
        - 0.06 * _safe_float(risk_components.get("bigloss_risk"))
    )


def _top_similar_symbols(record: Mapping[str, Any], symbol_id_to_symbol: Mapping[int, str]) -> set[str]:
    neighbor_ids = record.get("neighbor_symbol_ids") or []
    output: set[str] = set()
    for raw_id in list(neighbor_ids)[:5]:
        symbol = symbol_id_to_symbol.get(int(_safe_float(raw_id)))
        if symbol:
            output.add(symbol)
    return output


def _max_similarity(record: Mapping[str, Any], selected_symbols: set[str], symbol_id_to_symbol: Mapping[int, str]) -> float:
    neighbor_ids = list(record.get("neighbor_symbol_ids") or [])
    neighbor_scores = list(record.get("neighbor_scores") or [])
    best = 0.0
    for idx, raw_id in enumerate(neighbor_ids):
        symbol = symbol_id_to_symbol.get(int(_safe_float(raw_id)))
        if not symbol or symbol not in selected_symbols:
            continue
        score = _safe_float(neighbor_scores[idx], 1.0) if idx < len(neighbor_scores) else 1.0
        best = max(best, score)
    return _clip(best)


def _risk_percentile(sorted_values: Sequence[float], value: float) -> float:
    if not sorted_values:
        return 1.0
    if len(sorted_values) == 1:
        return 0.0
    less_than = sum(1 for candidate in sorted_values if candidate < value)
    return less_than / max(1, len(sorted_values) - 1)


def rank_market_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    top_n: int = 10,
    risk_preference: str = "balanced",
    strategy_style: str = "short_mid",
    context: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    force_top_n: bool = False,
) -> dict[str, Any]:
    if not records:
        return {
            "effective_trade_date": None,
            "top_n": int(top_n),
            "pool_size": 0,
            "selected_count": 0,
            "market_regime_counts": {},
            "candidates": [],
        }

    cfg = _load_config(config)
    top_n = max(1, int(top_n))
    symbol_id_to_symbol = {
        int(_safe_float(record.get("symbol_id"))): str(record.get("symbol"))
        for record in records
        if int(_safe_float(record.get("symbol_id"))) > 0 and record.get("symbol")
    }
    grouped_risk: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped_risk[str(record.get("industry_sw") or "Unknown")].append(_drawdown_magnitude(record.get("risk_dd_pred_5")))

    enriched: list[dict[str, Any]] = []
    market_regime_counter: Counter[str] = Counter()
    active_symbol_count = len(records)
    for record in records:
        industry = str(record.get("industry_sw") or "Unknown")
        risk_values = sorted(grouped_risk[industry])
        augmented_record = dict(record)
        augmented_record["industry_risk_dd_5_pctile"] = _risk_percentile(
            risk_values,
            _drawdown_magnitude(record.get("risk_dd_pred_5")),
        )
        market_denominator = int(_safe_float(augmented_record.get("market_symbol_count"))) or active_symbol_count
        augmented_record.setdefault("market_symbol_count", market_denominator)
        if market_denominator > 0:
            augmented_record.setdefault("down_limit_ratio", _safe_float(record.get("down_limit_count")) / market_denominator)
            augmented_record.setdefault("up_limit_ratio", _safe_float(record.get("up_limit_count")) / market_denominator)
        decision_result = evaluate_stock_decision(
            augmented_record,
            is_holding=False,
            holding_days=0,
            risk_preference=risk_preference,
            strategy_style=strategy_style,
            context=context,
            config=cfg,
            metadata=metadata,
        )
        regime = str(decision_result["market_regime"])
        market_regime_counter[regime] += 1
        selection_utility = _selection_utility(augmented_record, decision_result)
        enriched.append(
            {
                "record": augmented_record,
                "decision": decision_result,
                "selection_utility": selection_utility,
                "ranking_score": _legacy_score_from_utility(selection_utility),
                "similar_symbols": _top_similar_symbols(augmented_record, symbol_id_to_symbol),
            }
        )

    pool = [item for item in enriched if _is_candidate_eligible(item["record"], item["decision"], cfg)]
    pool.sort(key=lambda item: item["selection_utility"], reverse=True)
    dominant_regime = market_regime_counter.most_common(1)[0][0] if market_regime_counter else "neutral"
    industry_fraction = float(cfg["selection"][f"{dominant_regime}_industry_fraction"])
    board_fraction = float(cfg["selection"][f"{dominant_regime}_board_fraction"])
    industry_limit = max(1, int(math.ceil(top_n * industry_fraction)))
    board_limit = max(1, int(math.ceil(top_n * board_fraction)))
    similarity_penalty = float(cfg["selection"]["similarity_penalty"])

    selected: list[dict[str, Any]] = []
    industry_counter: Counter[str] = Counter()
    board_counter: Counter[str] = Counter()
    selected_symbols: set[str] = set()
    pool_remaining = list(pool)
    while pool_remaining and len(selected) < top_n:
        rescored: list[tuple[float, dict[str, Any], float]] = []
        for item in pool_remaining:
            max_similarity = _max_similarity(item["record"], selected_symbols, symbol_id_to_symbol)
            adjusted = item["selection_utility"] - similarity_penalty * max_similarity
            rescored.append((adjusted, item, max_similarity))
        rescored.sort(key=lambda entry: entry[0], reverse=True)
        accepted = False
        for adjusted, item, max_similarity in rescored:
            record = item["record"]
            industry = str(record.get("industry_sw") or "Unknown")
            board = str(record.get("board") or "Unknown")
            if industry_counter[industry] >= industry_limit:
                continue
            if board_counter[board] >= board_limit:
                continue
            selected_item = dict(item)
            selected_item["adjusted_selection_utility"] = adjusted
            selected_item["similarity_penalty"] = similarity_penalty * max_similarity
            selected.append(selected_item)
            selected_symbols.add(str(record.get("symbol") or ""))
            industry_counter[industry] += 1
            board_counter[board] += 1
            pool_remaining.remove(item)
            accepted = True
            break
        if not accepted:
            break

    if force_top_n and len(selected) < top_n:
        selected_symbol_keys = {str(item["record"].get("symbol") or "") for item in selected}
        fallback_ranked = sorted(
            enriched,
            key=lambda item: (
                item["selection_utility"],
                _normalize_probability(item["record"].get("p_win_prob_5")),
                _safe_float(item["record"].get("rank_score_pred")),
            ),
            reverse=True,
        )
        for item in fallback_ranked:
            symbol = str(item["record"].get("symbol") or "")
            if not symbol or symbol in selected_symbol_keys:
                continue
            fallback_item = dict(item)
            fallback_item.setdefault("adjusted_selection_utility", item["selection_utility"])
            fallback_item.setdefault("similarity_penalty", 0.0)
            selected.append(fallback_item)
            selected_symbol_keys.add(symbol)
            if len(selected) >= top_n:
                break

    candidates = []
    strict_pool_symbols = {str(item["record"].get("symbol") or "") for item in pool}
    for rank_index, item in enumerate(selected, start=1):
        record = item["record"]
        decision_result = dict(item["decision"])
        decision_result["scores"] = dict(decision_result["scores"])
        decision_result["scores"]["R_score"] = item["ranking_score"]
        decision_result["scores"]["selection_utility"] = item["selection_utility"]
        decision_result["scores"]["adjusted_selection_utility"] = item.get("adjusted_selection_utility", item["selection_utility"])
        symbol = str(record.get("symbol") or "")
        reasons = list(decision_result.get("reasons", []))
        if force_top_n and symbol not in strict_pool_symbols:
            reasons = ["兜底返回：严格过滤未凑满请求数量，已按选择效用补足。"] + reasons[:2]
        elif item.get("similarity_penalty", 0.0) > 0:
            reasons.append(f"Portfolio note: similarity penalty applied: {item['similarity_penalty']:.3f}.")
        else:
            reasons.append("Portfolio note: passed industry, board, and similarity diversification constraints.")
        candidates.append(
            {
                "rank": rank_index,
                "symbol": symbol,
                "name": str(record.get("name") or ""),
                "industry_sw": str(record.get("industry_sw") or ""),
                "board": str(record.get("board") or ""),
                "close": _safe_float(record.get("close")),
                "pct_chg": _safe_float(record.get("pct_chg")),
                "avg_amount_5": _safe_float(record.get("avg_amount_5")),
                "selection_utility": item["selection_utility"],
                "adjusted_selection_utility": item.get("adjusted_selection_utility", item["selection_utility"]),
                "decision_result": decision_result,
                "risk_flags": list(decision_result["risk_flags"]),
                "reasons": reasons,
            }
        )

    return {
        "effective_trade_date": str(records[0].get("trade_date") or ""),
        "top_n": top_n,
        "pool_size": len(pool),
        "selected_count": len(candidates),
        "market_regime_counts": dict(market_regime_counter),
        "candidates": candidates,
    }


def rank_market_candidates_quick(
    records: Sequence[Mapping[str, Any]],
    *,
    top_n: int = 10,
    holding_days: int | None = None,
) -> dict[str, Any]:
    """
    快速推荐排序：只使用模型输出的原始胜率，不再按批次或阈值做二次校准。

    流程：
    1. 接收所有已完成模型预测的候选记录。
    2. 为每只股票选择原始胜率最高的持有时长。
    3. 过滤掉胜率低于 50% 的股票，避免把低于抛硬币水平的结果当成推荐。
    4. 按原始胜率排序。
    5. 返回至多 top_n 只股票。

    Args:
        records: 预测后的候选记录。
        top_n: 返回数量。
    
    Returns:
        包含候选股票列表的字典。
    """
    if not records:
        return {
            "effective_trade_date": None,
            "top_n": int(top_n),
            "pool_size": 0,
            "selected_count": 0,
            "market_regime_counts": {},
            "candidates": [],
        }
    
    top_n = max(1, int(top_n))
    allowed_horizons = (3, 5, 10, 20, 40)
    if holding_days is not None and int(holding_days) not in allowed_horizons:
        raise ValueError("holding_days must be one of 3, 5, 10, 20, 40")
    horizons = (int(holding_days),) if holding_days is not None else (3, 5, 10)
    
    enriched: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    
    for record in records:
        best_horizon = horizons[0]
        best_p_win = -1.0
        best_ret_mu = _safe_float(record.get(f"ret_mu_pred_{best_horizon}"))

        for horizon in horizons:
            p_win_key = f"p_win_prob_{horizon}"
            p_win = _normalize_probability(record.get(p_win_key))
            ret_mu = _safe_float(record.get(f"ret_mu_pred_{horizon}"))
            candidate_score = (p_win, ret_mu, -horizon)
            if candidate_score > (best_p_win, best_ret_mu, -best_horizon):
                best_horizon = horizon
                best_p_win = p_win
                best_ret_mu = ret_mu

        if best_p_win < 0:
            best_p_win = 0.0
        
        symbol = str(record.get("symbol") or "")
        if not symbol:
            continue
        
        rank_score = _safe_float(record.get("rank_score_pred"), 0.5)
        avg_amount = _safe_float(record.get("avg_amount_5"), 0.0)
        signal_score = _safe_float(record.get("signal_score"), 0.5)
        enriched.append({
            "record": record,
            "symbol": symbol,
            "name": str(record.get("name") or ""),
            "industry_sw": str(record.get("industry_sw") or ""),
            "board": str(record.get("board") or ""),
            "close": _safe_float(record.get("close")),
            "pct_chg": _safe_float(record.get("pct_chg")),
            "avg_amount_5": avg_amount,
            "best_horizon": best_horizon,
            "best_p_win": best_p_win,
            "sorting_score": (best_p_win, best_ret_mu, signal_score, rank_score),
            "best_ret_mu": best_ret_mu,
            "rank_score": rank_score,
            "signal_score": signal_score,
        })

        if best_p_win >= QUICK_RECOMMEND_MIN_WIN_RATE:
            eligible.append(enriched[-1])

    eligible.sort(
        key=lambda x: (
            x["best_p_win"],
            x["best_ret_mu"],
            x["signal_score"],
            x["rank_score"],
        ),
        reverse=True,
    )
    selected = eligible[:top_n]
    
    candidates = []
    for rank_idx, item in enumerate(selected, start=1):
        record = item["record"]
        best_horizon = item["best_horizon"]
        best_p_win = item["best_p_win"]
        
        # 构建决策结果，供前端兼容展示使用。
        decision_result = {
            "symbol": item["symbol"],
            "symbol_name": item["name"],
            "trade_date": str(record.get("trade_date") or ""),
            "decision": {
                "action": "buy",
                "action_cn": "快速推荐",
                "confidence": best_p_win,
                "suggested_hold_days": best_horizon,
            },
            "scores": {
                "S_final": item["signal_score"],
                "S_3": _normalize_probability(record.get("p_win_prob_3")),
                "S_5": _normalize_probability(record.get("p_win_prob_5")),
                "S_10": _normalize_probability(record.get("p_win_prob_10")),
                "S_20": _normalize_probability(record.get("p_win_prob_20")),
                "S_40": _normalize_probability(record.get("p_win_prob_40")),
            },
        }
        
        candidates.append({
            "rank": rank_idx,
            "symbol": item["symbol"],
            "name": item["name"],
            "industry_sw": item["industry_sw"],
            "board": item["board"],
            "close": item["close"],
            "pct_chg": item["pct_chg"],
            "avg_amount_5": item["avg_amount_5"],
            "recommended_hold_days": best_horizon,
            "recommended_hold_label": f"{best_horizon}d",
            "predicted_win_rate": best_p_win,
            "raw_predicted_win_rate": best_p_win,
            "win_rate_source": "raw_model_output",
            "signal_score": item["signal_score"],
            "rank_score_pred": item["rank_score"],
            "ret_mu_pred": item["best_ret_mu"],
            "risk_dd_pred": _safe_float(record.get(f"risk_dd_pred_{best_horizon}")),
            "bigloss_prob": _safe_float(record.get("bigloss_prob_5")),
            "market_regime_prob": _safe_float(record.get("market_regime_prob"), 0.5),
            "market_snapshot": {
                "pct_chg": item["pct_chg"],
                "close": item["close"],
                "avg_amount_5": item["avg_amount_5"],
            },
            "decision_result": decision_result,
        })
    
    return {
        "effective_trade_date": str(records[0].get("trade_date") or "") if records else None,
        "top_n": top_n,
        "pool_size": len(eligible),
        "selected_count": len(candidates),
        "sample_size": len(enriched),  # 快速推荐中，样本池就是候选池。
        "total_candidates": len(enriched),
        "market_total_candidates": len(enriched),
        "market_regime_counts": {},  # 快速推荐不涉及市场状态分类。
        "candidates": candidates,
    }
