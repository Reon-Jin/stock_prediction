from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from utils.normalizers import dedupe_news_like, publication_to_trade_date
from utils.torch_runtime import resolve_torch_device


EVENT_EMBEDDING_DIM = 256
EVENT_EMBEDDING_COLUMN = "event_embedding"


def resolve_hf_model_source(model_name: str, prefer_local: bool = True) -> str:
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)
    if not prefer_local or "/" not in model_name:
        return model_name

    namespace, repo = model_name.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_root = cache_root / f"models--{namespace}--{repo}"
    ref_main = repo_root / "refs" / "main"
    if not ref_main.exists():
        return model_name

    revision = ref_main.read_text(encoding="utf-8").strip()
    if not revision:
        return model_name

    snapshot_dir = repo_root / "snapshots" / revision
    required_files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    if snapshot_dir.exists() and all((snapshot_dir / file_name).exists() for file_name in required_files):
        return str(snapshot_dir)
    return model_name


@dataclass(slots=True)
class EventExtractionResult:
    event_type: str
    event_direction: str
    event_strength: float
    event_confidence: float = 0.0
    event_backend: str = "rule"
    event_model: str | None = None
    event_details: dict[str, Any] | None = None


class BaseEventExtractor:
    def extract(self, title: str, content: str | None = None) -> EventExtractionResult:
        raise NotImplementedError


class RuleBasedEventExtractor(BaseEventExtractor):
    def __init__(self, config: dict[str, Any]):
        self.positive = config.get("positive_keywords", {})
        self.negative = config.get("negative_keywords", {})
        self.neutral = config.get("neutral_keywords", {})

    def extract(self, title: str, content: str | None = None) -> EventExtractionResult:
        text = f"{title} {content or ''}"
        for event_type, keywords in self.positive.items():
            if any(keyword in text for keyword in keywords):
                return EventExtractionResult(
                    event_type=event_type,
                    event_direction="positive",
                    event_strength=1.0,
                    event_confidence=1.0,
                    event_backend="rule",
                    event_model="keyword_rules",
                    event_details={"matched_group": "positive"},
                )
        for event_type, keywords in self.negative.items():
            if any(keyword in text for keyword in keywords):
                return EventExtractionResult(
                    event_type=event_type,
                    event_direction="negative",
                    event_strength=1.0,
                    event_confidence=1.0,
                    event_backend="rule",
                    event_model="keyword_rules",
                    event_details={"matched_group": "negative"},
                )
        for event_type, keywords in self.neutral.items():
            if any(keyword in text for keyword in keywords):
                return EventExtractionResult(
                    event_type=event_type,
                    event_direction="neutral",
                    event_strength=0.2,
                    event_confidence=0.6,
                    event_backend="rule",
                    event_model="keyword_rules",
                    event_details={"matched_group": "neutral"},
                )
        return EventExtractionResult(
            event_type="general_info",
            event_direction="neutral",
            event_strength=0.1,
            event_confidence=0.2,
            event_backend="rule",
            event_model="keyword_rules",
            event_details={"matched_group": "fallback"},
        )


DEFAULT_EVENT_TYPE_LABELS = {
    "earnings_up": "better earnings or profit outlook",
    "buyback": "share buyback or cancellation",
    "increase_holding": "major shareholder increases holding",
    "major_contract": "major contract or order win",
    "new_product": "new product, technology, or business progress",
    "policy_support": "policy support or industry tailwind",
    "reduction": "shareholder reduction or selling plan",
    "regulation": "regulatory investigation or penalty",
    "earnings_down": "earnings miss or business deterioration",
    "lawsuit": "lawsuit or major dispute",
    "pledge_risk": "equity pledge or liquidity risk",
    "black_swan": "black swan or sudden major risk",
    "routine": "routine disclosure or neutral progress update",
    "general_info": "general information without a clear event trigger",
}

DEFAULT_DIRECTION_LABELS = {
    "positive": "positive event or positive catalyst",
    "negative": "negative event or downside risk",
    "neutral": "neutral event or normal update",
}


class ZeroShotEventExtractor(BaseEventExtractor):
    def __init__(self, config: dict[str, Any]):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError("transformers is required for zero-shot event extraction") from exc

        self.model_name = str(config.get("classifier_model_name", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"))
        self.model_source = resolve_hf_model_source(
            self.model_name,
            prefer_local=bool(config.get("prefer_local_hf_cache", True)),
        )
        self.type_hypothesis_template = str(
            config.get("type_hypothesis_template", "This A-share news item is best described as {}.")
        )
        self.direction_hypothesis_template = str(
            config.get("direction_hypothesis_template", "The overall direction of this A-share news item is {}.")
        )
        self.type_threshold = float(config.get("type_confidence_threshold", 0.35))
        self.direction_threshold = float(config.get("direction_confidence_threshold", 0.40))
        self.max_text_length = int(config.get("max_text_length", 256))
        self.event_type_labels = config.get("event_type_labels") or DEFAULT_EVENT_TYPE_LABELS
        self.direction_labels = config.get("direction_labels") or DEFAULT_DIRECTION_LABELS
        hf_device = 0 if resolve_torch_device(config.get("hf_device", "auto")).type == "cuda" else -1
        self._classifier = pipeline(
            task="zero-shot-classification",
            model=self.model_source,
            device=hf_device,
        )

    def _truncate_text(self, title: str, content: str | None = None) -> str:
        text = f"{title.strip()} {(content or '').strip()}".strip()
        if len(text) <= self.max_text_length:
            return text
        return text[: self.max_text_length]

    def _classify(self, text: str, labels: dict[str, str], hypothesis_template: str) -> tuple[str, float, dict[str, float]]:
        label_texts = list(labels.values())
        result = self._classifier(
            sequences=text,
            candidate_labels=label_texts,
            multi_label=False,
            hypothesis_template=hypothesis_template,
        )
        ranked = {
            key: float(score)
            for label_text, score in zip(result["labels"], result["scores"])
            for key, candidate_text in labels.items()
            if candidate_text == label_text
        }
        if not ranked:
            first_key = next(iter(labels))
            return first_key, 0.0, {}
        best_key = max(ranked, key=ranked.get)
        return best_key, float(ranked[best_key]), ranked

    def extract(self, title: str, content: str | None = None) -> EventExtractionResult:
        text = self._truncate_text(title, content)
        if not text:
            return EventExtractionResult(
                event_type="general_info",
                event_direction="neutral",
                event_strength=0.1,
                event_confidence=0.0,
                event_backend="zero_shot",
                event_model=self.model_name,
                event_details={"reason": "empty_text"},
            )

        event_type, type_score, type_scores = self._classify(
            text=text,
            labels=self.event_type_labels,
            hypothesis_template=self.type_hypothesis_template,
        )
        event_direction, direction_score, direction_scores = self._classify(
            text=text,
            labels=self.direction_labels,
            hypothesis_template=self.direction_hypothesis_template,
        )

        if type_score < self.type_threshold:
            event_type = "general_info"
        if direction_score < self.direction_threshold:
            event_direction = "neutral"

        event_strength = float(max(0.05, min(1.0, 0.5 * type_score + 0.5 * direction_score)))
        return EventExtractionResult(
            event_type=event_type,
            event_direction=event_direction,
            event_strength=event_strength,
            event_confidence=float(min(type_score, direction_score)),
            event_backend="zero_shot",
            event_model=self.model_name,
            event_details={
                "type_scores": type_scores,
                "direction_scores": direction_scores,
            },
        )


class HybridEventExtractor(BaseEventExtractor):
    def __init__(self, primary: BaseEventExtractor, fallback: BaseEventExtractor, min_confidence: float = 0.35):
        self.primary = primary
        self.fallback = fallback
        self.min_confidence = float(min_confidence)

    def extract(self, title: str, content: str | None = None) -> EventExtractionResult:
        primary_result = self.primary.extract(title=title, content=content)
        if primary_result.event_confidence >= self.min_confidence and primary_result.event_type != "general_info":
            return primary_result
        fallback_result = self.fallback.extract(title=title, content=content)
        fallback_details = dict(fallback_result.event_details or {})
        fallback_details["primary_candidate"] = {
            "event_type": primary_result.event_type,
            "event_direction": primary_result.event_direction,
            "event_confidence": primary_result.event_confidence,
            "event_model": primary_result.event_model,
        }
        return EventExtractionResult(
            event_type=fallback_result.event_type,
            event_direction=fallback_result.event_direction,
            event_strength=max(fallback_result.event_strength, primary_result.event_strength),
            event_confidence=max(fallback_result.event_confidence, primary_result.event_confidence),
            event_backend="hybrid",
            event_model=primary_result.event_model or fallback_result.event_model,
            event_details=fallback_details,
        )


class LightweightClassifierExtractor(BaseEventExtractor):
    def __init__(self, fallback: BaseEventExtractor):
        self.fallback = fallback

    def extract(self, title: str, content: str | None = None) -> EventExtractionResult:
        return self.fallback.extract(title=title, content=content)


class DailyNewsEncoder:
    def __init__(self, config: dict[str, Any]):
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("transformers is required for daily news encoding") from exc

        self.model_name = str(
            config.get("embedding_model_name")
            or config.get("classifier_model_name")
            or "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        )
        self.output_dim = int(config.get("daily_event_embedding_dim", EVENT_EMBEDDING_DIM))
        self.max_length = int(config.get("embedding_max_length", config.get("max_text_length", 256)))
        self.batch_size = int(config.get("embedding_batch_size", 8))
        self.device = resolve_torch_device(config.get("hf_device", "auto"))
        self.model_source = resolve_hf_model_source(
            self.model_name,
            prefer_local=bool(config.get("prefer_local_hf_cache", True)),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_source)
        self._model = AutoModel.from_pretrained(self.model_source)
        self._model.to(self.device)
        self._model.eval()
        self.zero_vector = np.zeros(self.output_dim, dtype=np.float32)

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _project(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape[0] == self.output_dim:
            return vector.astype(np.float32)
        if vector.shape[0] > self.output_dim:
            parts = np.array_split(vector, self.output_dim)
            return np.asarray([float(part.mean()) if len(part) else 0.0 for part in parts], dtype=np.float32)
        output = np.zeros(self.output_dim, dtype=np.float32)
        output[: vector.shape[0]] = vector.astype(np.float32)
        return output

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for left in range(0, len(texts), self.batch_size):
            batch = texts[left : left + self.batch_size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.no_grad():
                outputs = self._model(**tokens)
                pooled = self._mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
            batch_vectors = pooled.detach().cpu().numpy()
            for item in batch_vectors:
                vectors.append(self._project(np.asarray(item, dtype=np.float32)))
        return np.vstack(vectors).astype(np.float32)

    def encode_daily(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return self.zero_vector.copy()
        article_vectors = self.encode_texts(texts)
        if article_vectors.size == 0:
            return self.zero_vector.copy()
        daily_vector = article_vectors.mean(axis=0)
        norm = float(np.linalg.norm(daily_vector))
        if norm > 0:
            daily_vector = daily_vector / norm
        return daily_vector.astype(np.float32)


def build_event_extractor(config: dict[str, Any]) -> BaseEventExtractor:
    rule = RuleBasedEventExtractor(config)
    mode = config.get("extraction_mode", "rule")
    if mode == "zero_shot":
        try:
            return ZeroShotEventExtractor(config)
        except Exception:
            return rule
    if mode == "hybrid":
        try:
            primary = ZeroShotEventExtractor(config)
            min_confidence = float(config.get("hybrid_min_confidence", 0.35))
            return HybridEventExtractor(primary=primary, fallback=rule, min_confidence=min_confidence)
        except Exception:
            return rule
    if mode == "classifier":
        return LightweightClassifierExtractor(rule)
    return rule


def normalize_event_records(
    raw_df: pd.DataFrame,
    id_column: str,
    calendar: Any,
    extractor: BaseEventExtractor,
) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()
    records = dedupe_news_like(raw_df.to_dict("records"), id_key=id_column)
    df = pd.DataFrame(records)
    df["publish_time"] = pd.to_datetime(df["publish_time"])
    if "target_trade_date" in df.columns:
        target_trade_dates = pd.to_datetime(df["target_trade_date"], errors="coerce")
        df["trade_date"] = target_trade_dates.dt.date
        missing_mask = target_trade_dates.isna()
        if missing_mask.any():
            df.loc[missing_mask, "trade_date"] = (
                df.loc[missing_mask, "publish_time"].apply(lambda x: publication_to_trade_date(x, calendar).date())
            )
    else:
        df["trade_date"] = df["publish_time"].apply(lambda x: publication_to_trade_date(x, calendar).date())
    extracted = df.apply(lambda row: extractor.extract(str(row["title"]), row.get("content")), axis=1)
    df["event_type"] = [item.event_type for item in extracted]
    df["event_direction"] = [item.event_direction for item in extracted]
    df["event_strength"] = [item.event_strength for item in extracted]
    df["event_confidence"] = [item.event_confidence for item in extracted]
    df["event_backend"] = [item.event_backend for item in extracted]
    df["event_model"] = [item.event_model for item in extracted]
    df["event_details"] = [item.event_details for item in extracted]
    df["normalized_flag"] = True
    return df


def build_event_features(
    news_norm: pd.DataFrame,
    daily_bars: pd.DataFrame,
    encoder: DailyNewsEncoder,
) -> pd.DataFrame:
    base = daily_bars[["trade_date"]].copy() if not daily_bars.empty else pd.DataFrame(columns=["trade_date"])
    if base.empty:
        return pd.DataFrame(columns=["trade_date", EVENT_EMBEDDING_COLUMN])

    base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce")
    base = base.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"]).sort_values("trade_date").copy()

    if news_norm.empty:
        base[EVENT_EMBEDDING_COLUMN] = [encoder.zero_vector.tolist() for _ in range(len(base))]
        return base[["trade_date", EVENT_EMBEDDING_COLUMN]]

    news = news_norm.copy()
    news["trade_date"] = pd.to_datetime(news["trade_date"], errors="coerce")
    news["publish_time"] = pd.to_datetime(news.get("publish_time"), errors="coerce")
    if "source" in news.columns:
        news["source"] = news["source"].fillna("").astype(str)
    else:
        news["source"] = ""
    news = news[news["source"].str.startswith("daily_")].copy()
    if news.empty:
        base[EVENT_EMBEDDING_COLUMN] = [encoder.zero_vector.tolist() for _ in range(len(base))]
        return base[["trade_date", EVENT_EMBEDDING_COLUMN]]

    daily_vector_map: dict[pd.Timestamp, list[float]] = {}
    ordered_news = news.sort_values(["trade_date", "source", "publish_time", "news_id"])
    for trade_date, group in ordered_news.groupby("trade_date", sort=True):
        texts: list[str] = []
        for row in group.head(10).to_dict("records"):
            title = str(row.get("title") or "").strip()
            summary = str(row.get("summary") or "").strip()
            content = str(row.get("content") or "").strip()
            source = str(row.get("source") or "").strip()
            texts.append(f"source={source}\ntitle={title}\nsummary={summary}\ncontent={content}".strip())
        daily_vector = encoder.encode_daily(texts)
        daily_vector_map[pd.Timestamp(trade_date).normalize()] = daily_vector.astype(float).tolist()

    zero_vector = encoder.zero_vector.astype(float).tolist()
    base[EVENT_EMBEDDING_COLUMN] = [
        list(daily_vector_map.get(pd.Timestamp(item).normalize(), zero_vector)) for item in base["trade_date"]
    ]
    return base[["trade_date", EVENT_EMBEDDING_COLUMN]]
