import copy

import numpy as np

from src.circuits import MAX_LAYERS, MANUAL_BASELINE, architecture_key, random_architecture, validate_architecture
from src.evolution import crossover, mutate, run_search


def test_mutation_always_gives_a_valid_different_circuit():
    rng = np.random.default_rng(0)
    for _ in range(300):
        parent = random_architecture(rng)
        before = copy.deepcopy(parent)
        child = mutate(parent, rng)
        validate_architecture(child)
        assert parent == before  # the parent must not be modified
        assert architecture_key(child) != architecture_key(parent)


def test_crossover_always_gives_a_valid_circuit():
    rng = np.random.default_rng(1)
    for _ in range(300):
        child = crossover(random_architecture(rng), random_architecture(rng), rng)
        validate_architecture(child)
        assert 1 <= len(child) <= MAX_LAYERS


def test_search_runs_and_never_retrains_duplicates():
    calls = []

    def fake_evaluate(arch):
        # Cheap stand-in for training: prefer circuits with fewer layers.
        calls.append(architecture_key(arch))
        return {"fitness": -len(arch), "val_acc": 0.5, "noisy_val_acc": 0.5,
                "gate_count": 10, "depth": 5}

    config = {"population_size": 6, "generations": 5, "elite_count": 2,
              "tournament_size": 3, "crossover_rate": 0.5, "patience": 10}
    result = run_search(fake_evaluate, config, np.random.default_rng(0), log=lambda *_: None)

    assert len(calls) == len(set(calls))
    assert len(result["best"]["arch"]) == 1
    best_so_far = [row["best_so_far_fitness"] for row in result["history"]]
    assert best_so_far == sorted(best_so_far)  # elitism: best never gets worse


def test_search_is_reproducible():
    def fake_evaluate(arch):
        return {"fitness": float(sum(g == "RY" for layer in arch for g in layer["rotations"])),
                "val_acc": 0.5, "noisy_val_acc": 0.5, "gate_count": 10, "depth": 5}

    config = {"population_size": 5, "generations": 3, "elite_count": 1,
              "tournament_size": 2, "crossover_rate": 0.5, "patience": 10}
    a = run_search(fake_evaluate, config, np.random.default_rng(3), log=lambda *_: None)
    b = run_search(fake_evaluate, config, np.random.default_rng(3), log=lambda *_: None)
    assert a["best"]["arch"] == b["best"]["arch"]
