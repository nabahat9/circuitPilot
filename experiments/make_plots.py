
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml

from src.circuits import describe_architecture, make_classifier, count_parameters
from src.io_utils import load_json, results_dir, run_name_argument

parser = argparse.ArgumentParser()
run_name_argument(parser)
args = parser.parse_args()
out_dir = results_dir(args.run_name)

summary = load_json(out_dir / "search_summary.json")
history = pd.read_csv(out_dir / "search_history.csv")
candidates = pd.read_csv(out_dir / "search_candidates.csv")
final = pd.read_csv(out_dir / "final_results.csv")
search_noise = summary["config"]["search_noise"]

LABELS = {
    "logistic_regression": "Logistic\nregression",
    "manual_qnn": "Manual\nQNN",
    "random_search_best": "Random-search\nbest QNN",
    "evolved_best": "Evolved\nbest QNN",
}

# 1. Baseline vs evolved, ideal and noisy, on the test set
fig, ax = plt.subplots(figsize=(7, 4))
models = list(LABELS)
x = np.arange(len(models))
for offset, noise, label in [(-0.2, 0.0, "ideal"), (0.2, search_noise, f"noisy (p={search_noise})")]:
    subset = final[final["noise_prob"] == noise].groupby("model")["test_acc"]
    means = subset.mean().reindex(models)
    stds = subset.std().reindex(models).fillna(0)
    ax.bar(x + offset, means, width=0.4, yerr=stds, capsize=4, label=label)
ax.set_xticks(x, [LABELS[m] for m in models])
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.5, 1.0)
ax.set_title("Test accuracy (error bars: std over training seeds)")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "baseline_vs_evolved.png", dpi=150)
plt.close(fig)

# 2. Search progress
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history["generation"], history["best_fitness"], "o-", label="best in generation")
ax.plot(history["generation"], history["mean_fitness"], "s--", label="population mean")
ax.axhline(summary["manual_baseline"]["fitness"], color="gray", linestyle=":", label="manual QNN")
ax.axhline(summary["best_random"]["fitness"], color="tab:red", linestyle=":", label="random-search best")
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness (validation-based)")
ax.set_title("Evolutionary search progress")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "search_progress.png", dpi=150)
plt.close(fig)

# 3. Accuracy vs complexity for every evaluated candidate
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, column, xlabel in [(axes[0], "gate_count", "Gate count"), (axes[1], "depth", "Circuit depth")]:
    for source, marker in [("random", "x"), ("evolution", "o")]:
        subset = candidates[candidates["source"] == source]
        ax.scatter(subset[column], subset["val_acc"], marker=marker, alpha=0.6, label=f"{source} candidates")
    ax.scatter(summary["manual_baseline"][column], summary["manual_baseline"]["val_acc"],
               s=150, marker="*", color="black", label="manual QNN")
    ax.scatter(summary["best_evolved"][column], summary["best_evolved"]["val_acc"],
               s=150, marker="*", color="tab:green", label="evolved best")
    ax.set_xlabel(xlabel)
axes[0].set_ylabel("Validation accuracy (ideal)")
axes[0].legend(fontsize=8)
fig.suptitle("Validation accuracy vs circuit complexity")
fig.tight_layout()
fig.savefig(out_dir / "accuracy_vs_complexity.png", dpi=150)
plt.close(fig)

# 4. Ideal vs noisy: accuracy as noise increases
fig, ax = plt.subplots(figsize=(7, 4))
for model in ["manual_qnn", "random_search_best", "evolved_best"]:
    grouped = final[final["model"] == model].groupby("noise_prob")["test_acc"]
    means, stds = grouped.mean(), grouped.std().fillna(0)
    ax.plot(means.index, means.values, "o-", label=LABELS[model].replace("\n", " "))
    ax.fill_between(means.index, means - stds, means + stds, alpha=0.2)
logreg = final[final["model"] == "logistic_regression"]["test_acc"].iloc[0]
ax.axhline(logreg, color="gray", linestyle=":", label="logistic regression (no noise)")
ax.set_xlabel("Depolarizing probability per gate")
ax.set_ylabel("Test accuracy")
ax.set_title("Robustness to simulated depolarizing noise")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "noise_robustness.png", dpi=150)
plt.close(fig)

# 5. Structure of the best evolved circuit
best_arch = summary["best_evolved"]["arch"]
circuit = make_classifier(best_arch)
dummy_weights = np.zeros(count_parameters(best_arch))
dummy_x = np.zeros(4)
fig, ax = qml.draw_mpl(circuit, decimals=None)(dummy_weights, dummy_x)
fig.suptitle("Best evolved circuit (first column of RY gates = fixed data encoding)")
fig.savefig(out_dir / "best_circuit.png", dpi=150, bbox_inches="tight")
plt.close(fig)
with open(out_dir / "best_circuit.txt", "w", encoding="utf-8") as f:
    f.write(describe_architecture(best_arch) + "\n\n")
    f.write(qml.draw(circuit, decimals=None)(dummy_weights, dummy_x) + "\n")

print(f"Saved 5 figures and best_circuit.txt to {out_dir}/")
