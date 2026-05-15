"""Quantile random forest surrogate for neutral EA candidate records."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from quantile_forest import RandomForestQuantileRegressor

try:
    from .surrogate_features import feature_dicts_to_matrix
except ImportError:  
    from surrogate_features import feature_dicts_to_matrix


@dataclass
class SurrogateSample:
    candidate_id: int
    features: dict
    deterministic_makespan: float
    robust_makespan: float
    robust_makespan_stdev: float
    R: float
    n_simulations: int
    source: str = "real"


@dataclass
class SurrogatePrediction:
    candidate_id: int
    mean_R: float
    q10_R: float
    q50_R: float
    q90_R: float
    uncertainty_R: float
    predicted_robust_makespan: float
    score: float


class QRFSurrogate:
    def __init__(
        self,
        min_samples_before_fit: int = 100,
        n_estimators: int = 300,
        min_samples_leaf: int = 3,
        max_features="sqrt",
        random_state: int | None = None,
    ):
        self.min_samples_before_fit = int(min_samples_before_fit)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

        self.samples: list[SurrogateSample] = []
        self.feature_names: list[str] | None = None
        self.model: RandomForestQuantileRegressor | None = None

    def add_sample(self, sample: SurrogateSample) -> None:
        self.samples.append(sample)

    def add_samples(self, samples: list[SurrogateSample]) -> None:
        self.samples.extend(samples)

    def is_ready(self) -> bool:
        return len(self.samples) >= self.min_samples_before_fit

    def fit(self) -> bool:
        if not self.is_ready():
            return False

        X, self.feature_names = feature_dicts_to_matrix(
            [sample.features for sample in self.samples],
            self.feature_names,
        )
        y = np.asarray([float(sample.R) for sample in self.samples], dtype=float)

        self.model = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return True

    def _require_fitted(self) -> RandomForestQuantileRegressor:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("QRFSurrogate is not fitted yet.")
        return self.model

    def _predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        model = self._require_fitted()
        try:
            quantiles = model.predict(X, quantiles=[0.1, 0.5, 0.9])
        except TypeError:
            quantiles = model.predict(X, quantiles=[10, 50, 90])
        return np.asarray(quantiles, dtype=float)

    def predict_one(
        self,
        candidate_id: int,
        features: dict,
        deterministic_makespan: float,
    ) -> SurrogatePrediction:
        X, _ = feature_dicts_to_matrix([features], self.feature_names)
        quantiles = self._predict_quantiles(X).reshape(1, -1)[0]
        q10_R, q50_R, q90_R = sorted(float(value) for value in quantiles)

        model = self._require_fitted()
        try:
            mean_R = float(np.asarray(model.predict(X, quantiles="mean")).reshape(-1)[0])
        except (TypeError, ValueError):
            mean_R = float(np.asarray(model.predict(X)).reshape(-1)[0])
        deterministic_makespan = float(deterministic_makespan)

        return SurrogatePrediction(
            candidate_id=int(candidate_id),
            mean_R=mean_R,
            q10_R=q10_R,
            q50_R=q50_R,
            q90_R=q90_R,
            uncertainty_R=q90_R - q10_R,
            predicted_robust_makespan=q50_R * deterministic_makespan,
            score=q90_R * deterministic_makespan,
        )

    def predict_many(self, candidate_records: list[dict]) -> list[SurrogatePrediction]:
        self._require_fitted()
        return [
            self.predict_one(
                candidate_id=record["candidate_id"],
                features=record["features"],
                deterministic_makespan=record["deterministic_makespan"],
            )
            for record in candidate_records
        ]

    def select_for_real_evaluation(
        self,
        predictions: list[SurrogatePrediction],
        top_fraction: float = 0.02,
        uncertain_fraction: float = 0.005,
        random_fraction: float = 0.005,
        min_count: int = 5,
        rng=None,
    ) -> set[int]:
        if not predictions:
            return set()

        n_predictions = len(predictions)
        top_count = self._fraction_count(n_predictions, top_fraction)
        uncertain_count = self._fraction_count(n_predictions, uncertain_fraction)
        random_count = self._fraction_count(n_predictions, random_fraction)

        selected: set[int] = set()
        selected.update(
            prediction.candidate_id
            for prediction in sorted(predictions, key=lambda item: item.score)[:top_count]
        )
        selected.update(
            prediction.candidate_id
            for prediction in sorted(
                predictions,
                key=lambda item: item.uncertainty_R,
                reverse=True,
            )[:uncertain_count]
        )

        rng = np.random.default_rng(rng) if rng is None or isinstance(rng, int) else rng
        remaining = [p.candidate_id for p in predictions if p.candidate_id not in selected]
        draw_count = min(random_count, len(remaining))
        if draw_count > 0:
            selected.update(self._random_choice(rng, remaining, draw_count))

        if len(selected) < min_count:
            for prediction in sorted(predictions, key=lambda item: item.score):
                selected.add(prediction.candidate_id)
                if len(selected) >= min(min_count, n_predictions):
                    break

        return selected

    @staticmethod
    def _fraction_count(total: int, fraction: float) -> int:
        if total <= 0 or fraction <= 0.0:
            return 0
        return min(total, max(1, int(np.ceil(total * fraction))))

    @staticmethod
    def _random_choice(rng: Any, values: list[int], count: int) -> list[int]:
        if hasattr(rng, "choice"):
            chosen = rng.choice(values, size=count, replace=False)
            return [int(value) for value in np.asarray(chosen).reshape(-1)]
        return list(np.random.default_rng().choice(values, size=count, replace=False))

    def to_training_frame(self):
        import pandas as pd

        rows = []
        for sample in self.samples:
            row = {
                "candidate_id": sample.candidate_id,
                "deterministic_makespan": sample.deterministic_makespan,
                "robust_makespan": sample.robust_makespan,
                "robust_makespan_stdev": sample.robust_makespan_stdev,
                "R": sample.R,
                "n_simulations": sample.n_simulations,
                "source": sample.source,
            }
            row.update({f"feature__{key}": value for key, value in sample.features.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "QRFSurrogate":
        path = Path(path)
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected QRFSurrogate in {path}, got {type(loaded)!r}.")
        return loaded
