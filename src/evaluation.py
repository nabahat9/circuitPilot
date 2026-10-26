import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.circuits import N_QUBITS, circuit_metrics, count_parameters, make_classifier, MAX_LAYERS

TRAIN_STEPS = 60
LEARNING_RATE = 0.1
INIT_SCALE = 0.1
SEARCH_NOISE = 0.02

# Weights of the scalar fitness function (see compute_fitness).
NOISE_WEIGHT = 0.5
COMPLEXITY_WEIGHT = 0.1


def train_classifier(arch: list, X_train, y_train, seed: int = 0,
                     steps: int = TRAIN_STEPS, learning_rate: float = LEARNING_RATE):
    """Full-batch Adam on the squared loss between circuit output + bias and labels in {-1, +1}."""
    circuit = make_classifier(arch)
    rng = np.random.default_rng(seed)
    # Small initial angles keep the untrained circuit close to the identity,
    # which gives a stable starting point for every candidate.
    weights = pnp.array(rng.normal(0, INIT_SCALE, count_parameters(arch)), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    X = pnp.array(X_train, requires_grad=False)
    y = pnp.array(y_train, requires_grad=False)

    def loss(weights, bias):
        predictions = circuit(weights, X) + bias
        return pnp.mean((predictions - y) ** 2)

    optimizer = qml.AdamOptimizer(learning_rate)
    loss_history = []
    for _ in range(steps):
        (weights, bias), current_loss = optimizer.step_and_cost(loss, weights, bias)
        loss_history.append(float(current_loss))
    return np.array(weights), float(bias), loss_history


def predict(arch: list, weights, bias: float, X, noise_prob: float = 0.0) -> np.ndarray:
    circuit = make_classifier(arch, noise_prob=noise_prob)
    outputs = np.asarray(circuit(weights, X)) + bias
    return np.where(outputs >= 0, 1, -1)


def accuracy(arch: list, weights, bias: float, X, y, noise_prob: float = 0.0) -> float:
    return float(np.mean(predict(arch, weights, bias, X, noise_prob) == y))


def max_complexity(n_qubits: int = N_QUBITS, max_layers: int = MAX_LAYERS) -> dict:
    """Metrics of the largest circuit in the search space, used to normalise complexity to [0, 1]."""
    largest = [{"rotations": ["RX"] * n_qubits, "entangler": "ring"}] * max_layers
    return circuit_metrics(largest, n_qubits)


def compute_fitness(val_acc: float, noisy_val_acc: float, metrics: dict,
                    noise_weight: float = NOISE_WEIGHT,
                    complexity_weight: float = COMPLEXITY_WEIGHT) -> float:
    """fitness = val_acc - noise_weight * accuracy_drop - complexity_weight * complexity

    - accuracy_drop = how much validation accuracy is lost under simulated noise
      (0 if noise happens to help, which can occur by chance).
    - complexity = average of normalised gate count and normalised depth, in [0, 1].
    With the default weights, losing 2 points of accuracy to noise costs as much
    as 1 point of clean accuracy, and going from the smallest to the largest
    circuit costs at most ~0.1 (about 10 points of accuracy).
    """
    largest = max_complexity()
    complexity = 0.5 * metrics["gate_count"] / largest["gate_count"] + 0.5 * metrics["depth"] / largest["depth"]
    accuracy_drop = max(0.0, val_acc - noisy_val_acc)
    return val_acc - noise_weight * accuracy_drop - complexity_weight * complexity


def evaluate_architecture(arch: list, data: dict, seed: int = 0, noise_prob: float = SEARCH_NOISE,
                          noise_weight: float = NOISE_WEIGHT,
                          complexity_weight: float = COMPLEXITY_WEIGHT) -> dict:
    """Train on the training split; score on the VALIDATION split only.

    The test split is never touched here, so it stays unseen until the final
    evaluation of the selected architecture.
    """
    weights, bias, loss_history = train_classifier(arch, data["X_train"], data["y_train"], seed=seed)
    metrics = circuit_metrics(arch)
    val_acc = accuracy(arch, weights, bias, data["X_val"], data["y_val"])
    noisy_val_acc = accuracy(arch, weights, bias, data["X_val"], data["y_val"], noise_prob)
    return {
        **metrics,
        "train_acc": accuracy(arch, weights, bias, data["X_train"], data["y_train"]),
        "val_acc": val_acc,
        "noisy_val_acc": noisy_val_acc,
        "final_loss": loss_history[-1],
        "fitness": compute_fitness(val_acc, noisy_val_acc, metrics, noise_weight, complexity_weight),
    }
