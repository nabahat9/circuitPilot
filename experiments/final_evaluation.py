import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from src.circuits import MANUAL_BASELINE
from src.data import load_dataset
from src.evaluation import accuracy, train_classifier
from src.io_utils import environment_info, load_json, results_dir, run_name_argument, save_json

parser = argparse.ArgumentParser()
run_name_argument(parser)
args = parser.parse_args()
out_dir = results_dir(args.run_name)

summary = load_json(out_dir / "search_summary.json")
DATA_SEED = summary["config"]["data_seed"]
TRAIN_SEEDS = list(range(2 if summary["quick"] else 10))
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]

data = load_dataset(seed=DATA_SEED)
models = {
    "manual_qnn": MANUAL_BASELINE,
    "random_search_best": summary["best_random"]["arch"],
    "evolved_best": summary["best_evolved"]["arch"],
}

rows = []
logreg = LogisticRegression(random_state=DATA_SEED).fit(data["X_train"], data["y_train"])
logreg_test_acc = logreg.score(data["X_test"], data["y_test"])
for noise in NOISE_LEVELS:
    rows.append({"model": "logistic_regression", "seed": 0, "noise_prob": noise, "test_acc": logreg_test_acc})

for name, arch in models.items():
    for seed in TRAIN_SEEDS:
        weights, bias, _ = train_classifier(arch, data["X_train"], data["y_train"], seed=seed)
        for noise in NOISE_LEVELS:
            acc = accuracy(arch, weights, bias, data["X_test"], data["y_test"], noise)
            rows.append({"model": name, "seed": seed, "noise_prob": noise, "test_acc": acc})
    print(f"finished {name}")

results = pd.DataFrame(rows)
results.to_csv(out_dir / "final_results.csv", index=False)

table = results.groupby(["model", "noise_prob"])["test_acc"].agg(["mean", "std"]).reset_index()
print("\nTest accuracy (mean and std over training seeds):")
print(table.pivot(index="model", columns="noise_prob", values="mean").round(3))


def paired_comparison(model_a, model_b, noise):
    """Compare two QNNs seed by seed (same seed = same initialisation rule and data)."""
    subset = results[results["noise_prob"] == noise]
    a = subset[subset["model"] == model_a].sort_values("seed")["test_acc"].to_numpy()
    b = subset[subset["model"] == model_b].sort_values("seed")["test_acc"].to_numpy()
    diff = a - b
    if len(diff) > 1 and np.std(diff) > 0:
        p_value = float(stats.ttest_rel(a, b).pvalue)
    else:
        p_value = None  # undefined when every difference is identical
    return {
        "noise_prob": noise,
        "mean_difference": float(diff.mean()),
        "std_difference": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
        "paired_t_test_p": p_value,
    }


comparisons = {}
for other in ["manual_qnn", "random_search_best"]:
    comparisons[f"evolved_best_vs_{other}"] = [paired_comparison("evolved_best", other, n) for n in [0.0, 0.02, 0.05]]

print("\nEvolved best minus other model (test accuracy, paired over seeds):")
for label, comps in comparisons.items():
    for c in comps:
        p = "n/a" if c["paired_t_test_p"] is None else f"{c['paired_t_test_p']:.3f}"
        print(f"  {label:38s} p_noise={c['noise_prob']:<5} diff={c['mean_difference']:+.3f} "
              f"(sd {c['std_difference']:.3f}) W/T/L={c['wins']}/{c['ties']}/{c['losses']} p={p}")

save_json({
    "data_seed": DATA_SEED,
    "train_seeds": TRAIN_SEEDS,
    "noise_levels": NOISE_LEVELS,
    "test_size": int(len(data["y_test"])),
    "summary_table": table.to_dict(orient="records"),
    "paired_comparisons": comparisons,
    "environment": environment_info(),
}, out_dir / "final_summary.json")
print(f"\nSaved {out_dir / 'final_results.csv'} and {out_dir / 'final_summary.json'}")
