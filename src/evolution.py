import copy

import numpy as np

from src.circuits import (ENTANGLERS, GATE_OPTIONS, MAX_LAYERS, N_QUBITS, architecture_key,
                          random_architecture, random_layer, repair_architecture,
                          validate_architecture)

MUTATION_TYPES = ("change_gate", "change_entangler", "add_layer", "remove_layer")


def mutate(arch: list, rng: np.random.Generator, n_qubits: int = N_QUBITS,
           max_layers: int = MAX_LAYERS) -> list:
    """Apply one random structural change, retrying if repair undid the change."""
    for _ in range(10):
        child = _mutate_once(arch, rng, n_qubits, max_layers)
        if architecture_key(child) != architecture_key(arch):
            break
    return child


def _mutate_once(arch: list, rng: np.random.Generator, n_qubits: int, max_layers: int) -> list:
    child = copy.deepcopy(arch)
    mutation = str(rng.choice(MUTATION_TYPES))

    # Fall back to a gate change when adding/removing a layer is impossible.
    if (mutation == "add_layer" and len(child) >= max_layers) or \
       (mutation == "remove_layer" and len(child) <= 1):
        mutation = "change_gate"

    if mutation == "change_gate":
        layer = child[int(rng.integers(len(child)))]
        wire = int(rng.integers(n_qubits))
        options = [g for g in GATE_OPTIONS if g != layer["rotations"][wire]]
        layer["rotations"][wire] = str(rng.choice(options))
    elif mutation == "change_entangler":
        layer = child[int(rng.integers(len(child)))]
        options = [e for e in ENTANGLERS if e != layer["entangler"]]
        layer["entangler"] = str(rng.choice(options))
    elif mutation == "add_layer":
        child.insert(int(rng.integers(len(child) + 1)), random_layer(rng, n_qubits))
    else:
        child.pop(int(rng.integers(len(child))))

    child = repair_architecture(child, rng)
    validate_architecture(child, n_qubits, max_layers)
    return child


def crossover(parent_a: list, parent_b: list, rng: np.random.Generator,
              max_layers: int = MAX_LAYERS) -> list:
    """One-point crossover on whole layers: first part of A followed by last part of B.

    Swapping whole layers (not single gates) keeps each layer's rotations and
    entangler together, and the result always has at least one layer.
    """
    cut_a = int(rng.integers(1, len(parent_a) + 1))
    cut_b = int(rng.integers(0, len(parent_b)))
    child = copy.deepcopy(parent_a[:cut_a] + parent_b[cut_b:])[:max_layers]
    child = repair_architecture(child, rng)
    validate_architecture(child, max_layers=max_layers)
    return child


def tournament_select(population: list, fitnesses: list, rng: np.random.Generator, k: int = 3) -> list:
    contestants = rng.choice(len(population), size=min(k, len(population)), replace=False)
    winner = max(contestants, key=lambda i: fitnesses[i])
    return population[winner]


def run_search(evaluate, config: dict, rng: np.random.Generator, log=print) -> dict:
    """Evolve architectures.

    evaluate(arch) -> dict with at least a "fitness" key. Results are cached by
    architecture_key, so a duplicate architecture is never trained twice.
    """
    cache = {}

    def get_result(arch):
        key = architecture_key(arch)
        if key not in cache:
            cache[key] = {"arch": arch, **evaluate(arch)}
            cache[key]["first_generation"] = generation
        return cache[key]

    def unique_child(new_population, parents):
        """Make a child that is not already in the new population (a few attempts)."""
        taken = {architecture_key(a) for a in new_population}
        for _ in range(10):
            parent_a = tournament_select(*parents, rng, config["tournament_size"])
            if rng.random() < config["crossover_rate"]:
                parent_b = tournament_select(*parents, rng, config["tournament_size"])
                child = crossover(parent_a, parent_b, rng)
            else:
                child = copy.deepcopy(parent_a)
            child = mutate(child, rng)
            if architecture_key(child) not in taken:
                return child
        return child

    generation = 0
    population, seen = [], set()
    while len(population) < config["population_size"]:
        arch = random_architecture(rng)
        if architecture_key(arch) not in seen:
            seen.add(architecture_key(arch))
            population.append(arch)

    history = []
    best_overall = None
    generations_without_improvement = 0

    for generation in range(config["generations"]):
        results = [get_result(arch) for arch in population]
        fitnesses = [r["fitness"] for r in results]
        best = max(results, key=lambda r: r["fitness"])

        if best_overall is None or best["fitness"] > best_overall["fitness"] + 1e-9:
            best_overall = best
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        history.append({
            "generation": generation,
            "best_fitness": best["fitness"],
            "mean_fitness": float(np.mean(fitnesses)),
            "best_val_acc": best["val_acc"],
            "best_noisy_val_acc": best["noisy_val_acc"],
            "best_gate_count": best["gate_count"],
            "best_depth": best["depth"],
            "best_so_far_fitness": best_overall["fitness"],
            "unique_evaluated": len(cache),
        })
        log(f"gen {generation:2d} | best {best['fitness']:.4f} | mean {np.mean(fitnesses):.4f} | "
            f"best val acc {best['val_acc']:.3f} | gates {best['gate_count']} | "
            f"unique evaluated {len(cache)}")

        if generations_without_improvement >= config["patience"]:
            log(f"Stopping: no improvement for {config['patience']} generations.")
            break
        if generation == config["generations"] - 1:
            break

        # Elitism: the best architectures survive unchanged.
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        new_population = [population[i] for i in ranked[:config["elite_count"]]]
        while len(new_population) < config["population_size"]:
            new_population.append(unique_child(new_population, (population, fitnesses)))
        population = new_population

    return {"history": history, "best": best_overall, "all_results": list(cache.values())}
