import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np
import pandas as pd

from src.circuits import (MANUAL_BASELINE, architecture_key, describe_architecture,
                          random_architecture)
from src.data import load_dataset
from src.evaluation import (COMPLEXITY_WEIGHT, LEARNING_RATE, NOISE_WEIGHT, SEARCH_NOISE,
                            TRAIN_STEPS, evaluate_architecture)
from src.evolution import run_search
from src.io_utils import environment_info, results_dir, run_name_argument, save_json

parser = argparse.ArgumentParser()
run_name_argument(parser)
parser.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
parser.add_argument("--complexity-weight", type=float, default=COMPLEXITY_WEIGHT)
parser.add_argument("--noise-weight", type=float, default=NOISE_WEIGHT)
args = parser.parse_args()

CONFIG = {
    "data_seed": 42,
    "train_seed": 0,         # same parameter initialisation rule for every candidate
    "search_seed": 7,
    "population_size": 4 if args.quick else 12,
    "generations": 2 if args.quick else 15,
    "elite_count": 2,
    "tournament_size": 3,
    "crossover_rate": 0.5,
    "patience": 6,
    "train_steps": TRAIN_STEPS,
    "learning_rate": LEARNING_RATE,
    "search_noise": SEARCH_NOISE,
    "noise_weight": args.noise_weight,
    "complexity_weight": args.complexity_weight,
}
out_dir = results_dir(args.run_name)
print(f"Run '{args.run_name}': noise weight {args.noise_weight}, complexity weight {args.complexity_weight}")

data = load_dataset(seed=CONFIG["data_seed"])


def evaluate(arch):
    return evaluate_architecture(arch, data, seed=CONFIG["train_seed"], noise_prob=CONFIG["search_noise"],
                                 noise_weight=CONFIG["noise_weight"],
                                 complexity_weight=CONFIG["complexity_weight"])


start = time.time()
print("Evolutionary search")
search = run_search(evaluate, CONFIG, np.random.default_rng(CONFIG["search_seed"]))
search_time = time.time() - start
budget = len(search["all_results"])

# Random-search control: the same number of unique evaluations, no evolution.
print(f"\nRandom search with the same budget ({budget} unique architectures)")
rng = np.random.default_rng(CONFIG["search_seed"] + 1)
random_results, seen = [], set()
while len(random_results) < budget:
    arch = random_architecture(rng)
    if architecture_key(arch) not in seen:
        seen.add(architecture_key(arch))
        random_results.append({"arch": arch, **evaluate(arch)})
random_best = max(random_results, key=lambda r: r["fitness"])

manual = {"arch": MANUAL_BASELINE, **evaluate(MANUAL_BASELINE)}
best = search["best"]

for name, result in [("Manual baseline", manual), ("Random-search best", random_best), ("Evolved best", best)]:
    print(f"\n{name}: fitness {result['fitness']:.4f}, val acc {result['val_acc']:.3f}, "
          f"noisy val acc {result['noisy_val_acc']:.3f}, gates {result['gate_count']}, depth {result['depth']}")
    print(describe_architecture(result["arch"]))

pd.DataFrame(search["history"]).to_csv(out_dir / "search_history.csv", index=False)
candidate_rows = []
for source, results in [("evolution", search["all_results"]), ("random", random_results)]:
    for r in results:
        row = {k: v for k, v in r.items() if k != "arch"}
        row["source"] = source
        row["arch"] = architecture_key(r["arch"])
        candidate_rows.append(row)
pd.DataFrame(candidate_rows).to_csv(out_dir / "search_candidates.csv", index=False)

save_json({
    "run_name": args.run_name,
    "quick": args.quick,
    "config": CONFIG,
    "unique_evaluations": budget,
    "search_time_seconds": search_time,
    "generations_run": len(search["history"]),
    "best_evolved": best,
    "best_random": random_best,
    "manual_baseline": manual,
    "environment": environment_info(),
}, out_dir / "search_summary.json")
print(f"\nSearch took {search_time:.0f} s. Saved results to {out_dir}/")
