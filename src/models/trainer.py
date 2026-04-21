from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.config_loader import ModelConfig


class ModelTrainer:
    """
    Trains multiple ML models with time series validation
    and builds an ensemble.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.ensemble: VotingClassifier | None = None
        self.feature_names: list[str] = []

    def prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Extract features and target variable."""
        # Exclude bookmaker odds features (norm_prob_*, odds_spread,
        # odds_prob_*) so the model learns to beat the bookmaker
        # rather than copy them. Also exclude raw xG proxy columns
        # that used same-match shot data (now replaced by rolling xG).
        excluded = {
            "norm_prob_H",
            "norm_prob_D",
            "norm_prob_A",
            "odds_prob_H",
            "odds_prob_D",
            "odds_prob_A",
            "odds_spread",
            "home_xG_proxy",
            "away_xG_proxy",
            "home_xG_overperf",
            "away_xG_overperf",
        }
        feature_cols = [
            c
            for c in df.columns
            if c.startswith(
                (
                    "home_",
                    "away_",
                    "diff_",
                    "elo_",
                    "h2h_",
                    "rest_",
                    "is_midweek",
                    "xG_diff",
                    "avg_draw_pct",
                    "form_gap",
                    "attack_similarity",
                    "defense_similarity",
                    "combined_defensive",
                )
            )
            and c not in excluded
        ]

        X = df[feature_cols].copy()
        y = df["Result"].copy()

        X = X.fillna(X.median())

        self.feature_names = feature_cols
        return X, y, feature_cols

    def _build_models(self) -> dict[str, object]:
        """Build individual model instances from config."""
        lr_cfg = self.config.logistic_regression
        rf_cfg = self.config.random_forest
        xgb_cfg = self.config.xgboost

        return {
            "Logistic Regression": LogisticRegression(
                max_iter=lr_cfg.max_iter,
                C=lr_cfg.C,
                class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=rf_cfg.n_estimators,
                max_depth=rf_cfg.max_depth,
                min_samples_leaf=rf_cfg.min_samples_leaf,
                random_state=self.config.random_state,
                class_weight="balanced",
            ),
            "XGBoost": XGBClassifier(
                n_estimators=xgb_cfg.n_estimators,
                max_depth=xgb_cfg.max_depth,
                learning_rate=xgb_cfg.learning_rate,
                subsample=xgb_cfg.subsample,
                colsample_bytree=xgb_cfg.colsample_bytree,
                random_state=self.config.random_state,
                eval_metric="mlogloss",
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                random_state=self.config.random_state,
            ),
        }

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> dict[str, dict[str, float]]:
        """Train and evaluate multiple models with TimeSeriesSplit."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        models = self._build_models()
        results: dict[str, dict[str, float]] = {}

        for name, model in models.items():
            fold_accuracies: list[float] = []
            fold_log_losses: list[float] = []

            for train_idx, test_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)  # type: ignore[union-attr]

                preds = model.predict(X_test_scaled)  # type: ignore[union-attr]
                proba = model.predict_proba(X_test_scaled)  # type: ignore[union-attr]

                fold_accuracies.append(float(accuracy_score(y_test, preds)))
                fold_log_losses.append(float(log_loss(y_test, proba)))

            results[name] = {
                "accuracy_mean": float(np.mean(fold_accuracies)),
                "accuracy_std": float(np.std(fold_accuracies)),
                "log_loss_mean": float(np.mean(fold_log_losses)),
                "log_loss_std": float(np.std(fold_log_losses)),
            }

            print(f"\n{'=' * 50}")
            print(f"  {name}")
            print(
                f"  Accuracy:  {results[name]['accuracy_mean']:.4f} "
                f"+/- {results[name]['accuracy_std']:.4f}"
            )
            print(
                f"  Log Loss:  {results[name]['log_loss_mean']:.4f} "
                f"+/- {results[name]['log_loss_std']:.4f}"
            )

        return results

    def build_ensemble(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Build and train an ensemble model with soft voting."""
        lr_cfg = self.config.logistic_regression
        rf_cfg = self.config.random_forest
        xgb_cfg = self.config.xgboost
        ens_cfg = self.config.ensemble

        self.ensemble = VotingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        max_iter=lr_cfg.max_iter,
                        C=lr_cfg.C,
                        class_weight="balanced",
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=rf_cfg.n_estimators,
                        max_depth=rf_cfg.max_depth,
                        min_samples_leaf=rf_cfg.min_samples_leaf,
                        random_state=self.config.random_state,
                        class_weight="balanced",
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=xgb_cfg.n_estimators,
                        max_depth=xgb_cfg.max_depth,
                        learning_rate=xgb_cfg.learning_rate,
                        subsample=xgb_cfg.subsample,
                        colsample_bytree=xgb_cfg.colsample_bytree,
                        random_state=self.config.random_state,
                        eval_metric="mlogloss",
                    ),
                ),
            ],
            voting=ens_cfg.voting,
            weights=ens_cfg.weights,
        )

        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.ensemble.fit(X_train_scaled, y_train)

        preds = self.ensemble.predict(X_test_scaled)
        proba = self.ensemble.predict_proba(X_test_scaled)

        accuracy = float(accuracy_score(y_test, preds))
        logloss = float(log_loss(y_test, proba))

        print(f"\n{'=' * 60}")
        print("  ENSEMBLE (Soft Voting)")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Log Loss:  {logloss:.4f}")
        print(
            f"\n{classification_report(y_test, preds, target_names=['Away Win', 'Draw', 'Home Win'])}"
        )

        return {"accuracy": accuracy, "log_loss": logloss}

    def save_model(self, path: str | Path, last_match_date: str | None = None) -> None:
        """Save the trained ensemble and scaler."""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.ensemble is not None:
            joblib.dump(self.ensemble, output_path / "ensemble_model.joblib")
        joblib.dump(self.scaler, output_path / "scaler.joblib")
        joblib.dump(self.feature_names, output_path / "feature_names.joblib")
        self._last_match_date = last_match_date
        print(f"Model saved to {output_path}")

    def load_model(self, path: str | Path) -> None:
        """Load a trained ensemble and scaler."""
        model_path = Path(path)
        self.ensemble = joblib.load(model_path / "ensemble_model.joblib")
        self.scaler = joblib.load(model_path / "scaler.joblib")
        self.feature_names = joblib.load(model_path / "feature_names.joblib")
        print(f"Model loaded from {model_path}")
