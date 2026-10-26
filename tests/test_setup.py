import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def test_bell_state_probabilities():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    assert np.allclose(circuit(), [0.5, 0.0, 0.0, 0.5])


def test_rotation_gradient_matches_analytic_value():
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = pnp.array(0.5, requires_grad=True)
    assert np.isclose(qml.grad(circuit)(theta), -np.sin(0.5))


def test_depolarizing_noise_reduces_expectation():
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circuit(p):
        qml.DepolarizingChannel(p, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert np.isclose(circuit(0.1), 1 - 4 * 0.1 / 3)
    assert np.isclose(circuit(0.0), 1.0)
