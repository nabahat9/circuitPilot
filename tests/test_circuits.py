import numpy as np
import pennylane as qml
import pytest

from src.circuits import (MANUAL_BASELINE, architecture_key, circuit_metrics, count_parameters,
                          make_classifier, random_architecture, validate_architecture)


def test_manual_baseline_metrics():
    metrics = circuit_metrics(MANUAL_BASELINE)
    # 4 encoding RY + 8 trainable RY + 8 CNOT
    assert metrics["gate_count"] == 20
    assert metrics["two_qubit_gates"] == 8
    assert metrics["n_params"] == 8


@pytest.mark.parametrize("seed", range(5))
def test_metrics_agree_with_pennylane_specs(seed):
    arch = random_architecture(np.random.default_rng(seed))
    circuit = make_classifier(arch)
    specs = qml.specs(circuit)(np.zeros(count_parameters(arch)), np.zeros(4))
    metrics = circuit_metrics(arch)
    assert metrics["depth"] == specs["resources"].depth
    assert metrics["gate_count"] == specs["resources"].num_gates


def test_random_architectures_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(200):
        validate_architecture(random_architecture(rng))


@pytest.mark.parametrize("bad_arch", [
    [],
    [{"rotations": ["RY", "RY", "RY"], "entangler": "ring"}],
    [{"rotations": ["RY", "RY", "RY", "XX"], "entangler": "ring"}],
    [{"rotations": ["RY"] * 4, "entangler": "star"}],
    [{"rotations": ["none"] * 4, "entangler": "ring"}],
])
def test_invalid_architectures_are_rejected(bad_arch):
    with pytest.raises(ValueError):
        validate_architecture(bad_arch)


def test_classifier_output_is_an_expectation_value():
    circuit = make_classifier(MANUAL_BASELINE)
    X = np.random.default_rng(0).uniform(0, np.pi, size=(10, 4))
    outputs = np.asarray(circuit(np.ones(8), X))
    assert outputs.shape == (10,)
    assert np.all(np.abs(outputs) <= 1 + 1e-9)


def test_noisy_circuit_with_tiny_noise_matches_ideal():
    x = np.array([0.3, 1.2, 2.0, 0.7])
    ideal = make_classifier(MANUAL_BASELINE)(np.ones(8), x)
    noisy = make_classifier(MANUAL_BASELINE, noise_prob=1e-9)(np.ones(8), x)
    assert np.isclose(ideal, noisy, atol=1e-6)


def test_duplicate_architectures_have_same_key():
    copy = [dict(layer, rotations=list(layer["rotations"])) for layer in MANUAL_BASELINE]
    assert architecture_key(copy) == architecture_key(MANUAL_BASELINE)
