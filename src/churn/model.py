"""Logistic-regression churn model in pure NumPy.

No scikit-learn dependency: the model is gradient descent on standardised
numeric features plus one-hot categoricals, with L2 regularisation. That
keeps ``requirements.txt`` light, makes the coefficients directly readable
as churn drivers, and keeps the Streamlit Cloud install fast.

AUC is computed by the Mann-Whitney rank statistic, so evaluation also
needs nothing beyond NumPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC; returns 0.5 when only one class is present."""
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / counts
    ranks = avg[inv]
    return float(
        (ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2)
        / (len(pos) * len(neg))
    )


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(_auc(y_true, prob)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


class ChurnModel:
    """Standardised logistic regression over engineered features."""

    def __init__(self, lr: float = 0.10, epochs: int = 800, l2: float = 0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.categories_: dict[str, list] = {}
        self.feature_names_: list[str] = []
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def _fit_encoder(self, df: pd.DataFrame) -> None:
        feats = add_features(df)
        self.mean_ = feats[NUMERIC_FEATURES].to_numpy(dtype=float).mean(axis=0)
        self.std_ = feats[NUMERIC_FEATURES].to_numpy(dtype=float).std(axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.categories_ = {
            c: sorted(feats[c].astype(str).unique().tolist())
            for c in CATEGORICAL_FEATURES
        }
        names = list(NUMERIC_FEATURES)
        for col in CATEGORICAL_FEATURES:
            names += [f"{col}={v}" for v in self.categories_[col]]
        self.feature_names_ = names

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        feats = add_features(df)
        num = feats[NUMERIC_FEATURES].to_numpy(dtype=float)
        num = (num - self.mean_) / self.std_
        blocks = [num]
        for col in CATEGORICAL_FEATURES:
            cats = self.categories_[col]
            onehot = np.zeros((len(feats), len(cats)))
            for j, cat in enumerate(cats):
                onehot[:, j] = (feats[col].astype(str) == cat).astype(float)
            blocks.append(onehot)
        return np.hstack(blocks)

    def fit(self, df: pd.DataFrame) -> "ChurnModel":
        self._fit_encoder(df)
        X = self._encode(df)
        y = add_features(df)["churn_flag"].to_numpy(dtype=float)
        w = np.zeros(X.shape[1])
        b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(X @ w + b)
            err = p - y
            grad_w = X.T @ err / len(y) + self.l2 * w
            grad_b = err.mean()
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._encode(df)
        return _sigmoid(X @ self.coef_ + self.intercept_)

    def top_drivers(self, n: int = 10) -> pd.DataFrame:
        order = np.argsort(-np.abs(self.coef_))[:n]
        return pd.DataFrame(
            {
                "feature": [self.feature_names_[i] for i in order],
                "coefficient": [float(self.coef_[i]) for i in order],
            }
        )


def train_churn_model(df: pd.DataFrame, test_size: float = 0.20,
                       seed: int = 42) -> dict:
    """Deterministic shuffle split; returns model + train/test metrics."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    model = ChurnModel().fit(train)
    return {
        "model": model,
        "train": _metrics(
            add_features(train)["churn_flag"].to_numpy(),
            model.predict_proba(train),
        ),
        "test": _metrics(
            add_features(test)["churn_flag"].to_numpy(),
            model.predict_proba(test),
        ),
        "n_train": len(train),
        "n_test": len(test),
    }
