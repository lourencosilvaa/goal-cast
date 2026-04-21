import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["font.size"] = 11
sns.set_style("whitegrid")


class PredictionPlots:
    """Visualization tools for football prediction analysis."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def plot_model_comparison(self, results: dict[str, dict[str, float]]) -> None:
        """Visualization of model accuracy comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        names = list(results.keys())
        accuracies = [results[n]["accuracy_mean"] for n in names]
        acc_stds = [results[n]["accuracy_std"] for n in names]
        log_losses = [results[n]["log_loss_mean"] for n in names]
        ll_stds = [results[n]["log_loss_std"] for n in names]

        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

        bars = axes[0].barh(
            names,
            accuracies,
            xerr=acc_stds,
            color=colors[: len(names)],
            edgecolor="white",
            linewidth=1.5,
        )
        axes[0].set_xlabel("Accuracy")
        axes[0].set_title("Model Accuracy (TimeSeriesSplit CV)")
        axes[0].set_xlim(0.3, 0.65)
        for bar, val in zip(bars, accuracies):
            axes[0].text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontweight="bold",
            )

        bars = axes[1].barh(
            names,
            log_losses,
            xerr=ll_stds,
            color=colors[: len(names)],
            edgecolor="white",
            linewidth=1.5,
        )
        axes[1].set_xlabel("Log Loss")
        axes[1].set_title("Model Log Loss (lower = better)")
        for bar, val in zip(bars, log_losses):
            axes[1].text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison.png", bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        labels = ["Away Win", "Draw", "Home Win"]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_xlabel("Predicted Result", fontsize=12)
        ax.set_ylabel("Actual Result", fontsize=12)
        ax.set_title("Confusion Matrix - Ensemble Model", fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrix.png", bbox_inches="tight")
        plt.close()

    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: list[str],
        top_n: int = 15,
    ) -> None:
        """Feature importance visualization."""
        indices = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))  # type: ignore[attr-defined]
        ax.barh(
            range(top_n),
            importances[indices],
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Important Features", fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png", bbox_inches="tight")
        plt.close()

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_idx: int = 2,
        class_name: str = "Home Win",
    ) -> None:
        """Calibration plot showing predicted vs actual probabilities."""
        prob_true, prob_pred = calibration_curve(
            (y_true == class_idx).astype(int),
            y_proba[:, class_idx],
            n_bins=10,
            strategy="uniform",
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(
            prob_pred,
            prob_true,
            "s-",
            color="#e74c3c",
            label=f"Model ({class_name})",
            linewidth=2,
            markersize=8,
        )
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color="#e74c3c")

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Actual fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/calibration_curve.png", bbox_inches="tight")
        plt.close()

    def plot_divergence_scatter(self, matches: list[dict[str, float]]) -> None:
        """Scatter plot: ML probabilities vs bookmaker probabilities."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        outcomes = [
            ("home", "H", "Home Win"),
            ("draw", "D", "Draw"),
            ("away", "A", "Away Win"),
        ]
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

        for ax, (key, label, title), color in zip(axes, outcomes, colors):
            ml_probs = [m.get(f"ml_prob_{label}", 0) for m in matches]
            bk_probs = [m.get(f"bk_prob_{label}", 0) for m in matches]

            ax.scatter(
                bk_probs,
                ml_probs,
                alpha=0.6,
                color=color,
                edgecolors="white",
                s=60,
            )
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)

            ax.set_xlabel("Bookmaker P", fontsize=11)
            ax.set_ylabel("ML Model P", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")

        plt.suptitle(
            "Probability Divergence: ML Model vs Bookmakers\n"
            "Points far from diagonal = potential value",
            fontsize=14,
            y=1.04,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/divergence_scatter.png", bbox_inches="tight")
        plt.close()

    def plot_triple_radar(
        self,
        match_name: str,
        ml_probs: dict[str, float],
        bookmaker_probs: dict[str, float],
    ) -> None:
        """Radar chart comparing ML and bookmaker probability sources."""
        categories = ["Home Win", "Draw", "Away Win"]
        keys = ["home", "draw", "away"]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        sources = [
            ("ML Model", ml_probs, "#2ecc71"),
            ("Bookmakers", bookmaker_probs, "#3498db"),
        ]

        for label, probs, color in sources:
            values = [probs.get(k, 0) for k in keys]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 0.8)
        ax.set_title(f"ML vs Bookmakers: {match_name}", fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/radar_{match_name.replace(' ', '_')}.png",
            bbox_inches="tight",
        )
        plt.close()
