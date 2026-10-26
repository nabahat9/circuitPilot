import numpy as np

from src.circuits import MANUAL_BASELINE, circuit_metrics
from src.data import load_dataset
from src.evaluation import accuracy, compute_fitness, train_classifier


def test_training_reduces_loss_and_beats_chance():
    data = load_dataset(seed=42)
    weights, bias, losses = train_classifier(MANUAL_BASELINE, data["X_train"], data["y_train"], seed=0, steps=30)
    assert losses[-1] < losses[0]
    assert accuracy(MANUAL_BASELINE, weights, bias, data["X_val"], data["y_val"]) > 0.7


def test_training_is_reproducible():
    data = load_dataset(seed=42)
    w1, b1, _ = train_classifier(MANUAL_BASELINE, data["X_train"], data["y_train"], seed=5, steps=5)
    w2, b2, _ = train_classifier(MANUAL_BASELINE, data["X_train"], data["y_train"], seed=5, steps=5)
    assert np.allclose(w1, w2) and b1 == b2


def test_fitness_penalises_noise_drop_and_complexity():
    small = circuit_metrics([{"rotations": ["RY", "none", "none", "none"], "entangler": "none"}])
    large = circuit_metrics(MANUAL_BASELINE * 2)
    assert compute_fitness(0.9, 0.9, small) > compute_fitness(0.9, 0.8, small)
    assert compute_fitness(0.9, 0.9, small) > compute_fitness(0.9, 0.9, large)
    # noise that happens to help is not rewarded
    assert compute_fitness(0.9, 0.95, small) == compute_fitness(0.9, 0.9, small)
