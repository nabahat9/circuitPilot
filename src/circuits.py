

import json

import numpy as np
import pennylane as qml

N_QUBITS = 4
MAX_LAYERS = 4
ROTATIONS = ("RX", "RY", "RZ")
GATE_OPTIONS = ROTATIONS + ("none",)
ENTANGLERS = ("none", "linear", "ring")

ROTATION_GATES = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}

# A conventional hand-designed ansatz ("hardware-efficient" style):
# two layers of RY rotations, each followed by a ring of CNOTs.
MANUAL_BASELINE = [
    {"rotations": ["RY"] * N_QUBITS, "entangler": "ring"},
    {"rotations": ["RY"] * N_QUBITS, "entangler": "ring"},
]


def random_architecture(rng: np.random.Generator, n_qubits: int = N_QUBITS,
                        max_layers: int = MAX_LAYERS) -> list:
    n_layers = int(rng.integers(1, max_layers + 1))
    arch = [random_layer(rng, n_qubits) for _ in range(n_layers)]
    return repair_architecture(arch, rng)


def random_layer(rng: np.random.Generator, n_qubits: int = N_QUBITS) -> dict:
    return {
        "rotations": [str(rng.choice(GATE_OPTIONS)) for _ in range(n_qubits)],
        "entangler": str(rng.choice(ENTANGLERS)),
    }


def repair_architecture(arch: list, rng: np.random.Generator) -> list:
    """Make sure the circuit has at least one trainable gate."""
    if count_parameters(arch) == 0:
        layer = arch[int(rng.integers(len(arch)))]
        layer["rotations"][int(rng.integers(len(layer["rotations"])))] = str(rng.choice(ROTATIONS))
    return arch


def validate_architecture(arch: list, n_qubits: int = N_QUBITS, max_layers: int = MAX_LAYERS) -> None:
    if not isinstance(arch, list) or not 1 <= len(arch) <= max_layers:
        raise ValueError(f"Architecture must have between 1 and {max_layers} layers.")
    for i, layer in enumerate(arch):
        if len(layer["rotations"]) != n_qubits:
            raise ValueError(f"Layer {i} must have {n_qubits} rotation entries.")
        for gate in layer["rotations"]:
            if gate not in GATE_OPTIONS:
                raise ValueError(f"Layer {i} has unknown gate {gate!r}.")
        if layer["entangler"] not in ENTANGLERS:
            raise ValueError(f"Layer {i} has unknown entangler {layer['entangler']!r}.")
    if count_parameters(arch) == 0:
        raise ValueError("Architecture has no trainable gates.")


def architecture_key(arch: list) -> str:
    """A string that is identical for identical architectures (used to detect duplicates)."""
    return json.dumps(arch, sort_keys=True)


def count_parameters(arch: list) -> int:
    return sum(gate != "none" for layer in arch for gate in layer["rotations"])


def entangling_pairs(pattern: str, n_qubits: int = N_QUBITS) -> list:
    if pattern == "none":
        return []
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    if pattern == "ring" and n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    return pairs


def gate_sequence(arch: list, n_qubits: int = N_QUBITS) -> list:
    """Every gate in the full circuit (encoding + ansatz) as (name, wires, weight_index).

    weight_index is None for encoding gates and CNOTs. Both the PennyLane circuit
    and the complexity metrics are generated from this one list, so they cannot
    disagree with each other.
    """
    gates = [("ENCODE_RY", (w,), None) for w in range(n_qubits)]
    weight_index = 0
    for layer in arch:
        for wire, gate in enumerate(layer["rotations"]):
            if gate != "none":
                gates.append((gate, (wire,), weight_index))
                weight_index += 1
        for control, target in entangling_pairs(layer["entangler"], n_qubits):
            gates.append(("CNOT", (control, target), None))
    return gates


def circuit_metrics(arch: list, n_qubits: int = N_QUBITS) -> dict:
    """Depth, gate counts and parameter count of the full circuit.

    These are different quantities: two circuits with the same gate count can
    have very different depths (gates in parallel vs in sequence), and CNOTs
    add gates and depth but no parameters.
    """
    wire_depth = [0] * n_qubits
    two_qubit = 0
    gates = gate_sequence(arch, n_qubits)
    for name, wires, _ in gates:
        layer_index = max(wire_depth[w] for w in wires) + 1
        for w in wires:
            wire_depth[w] = layer_index
        two_qubit += len(wires) == 2
    return {
        "depth": max(wire_depth),
        "gate_count": len(gates),
        "two_qubit_gates": two_qubit,
        "n_params": count_parameters(arch),
        "n_layers": len(arch),
    }


def make_classifier(arch: list, n_qubits: int = N_QUBITS, noise_prob: float = 0.0):
    """Return a QNode f(weights, x) -> <Z on qubit 0>, a value in [-1, 1].

    With noise_prob > 0 the circuit runs on a density-matrix simulator and a
    depolarizing channel is applied to every wire a gate acts on, right after
    that gate. This is a simple, uniform noise model, not a model of any real device.
    """
    validate_architecture(arch, n_qubits)
    gates = gate_sequence(arch, n_qubits)
    device_name = "default.mixed" if noise_prob > 0 else "default.qubit"
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, x):
        for name, wires, weight_index in gates:
            if name == "ENCODE_RY":
                qml.RY(x[..., wires[0]], wires=wires[0])
            elif name == "CNOT":
                qml.CNOT(wires=wires)
            else:
                ROTATION_GATES[name](weights[weight_index], wires=wires[0])
            if noise_prob > 0:
                for w in wires:
                    qml.DepolarizingChannel(noise_prob, wires=w)
        return qml.expval(qml.PauliZ(0))

    return circuit


def describe_architecture(arch: list) -> str:
    lines = []
    for i, layer in enumerate(arch):
        gates = " ".join(f"{g:>4}" for g in layer["rotations"])
        lines.append(f"layer {i + 1}: [{gates}]  entangler={layer['entangler']}")
    return "\n".join(lines)
