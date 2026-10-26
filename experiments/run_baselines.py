import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lets "import src" work

from sklearn.linear_model import LogisticRegression

from src.circuits import MANUAL_BASELINE, describe_architecture, make_classifier
from src.data import load_dataset
from src.evaluation import SEARCH_NOISE, evaluate_architecture, train_classifier
from src.io_utils import environment_info, results_dir, save_json

import pennylane as qml

DATA_SEED = 42
TRAIN_SEED = 0

out_dir = results_dir()

data = load_dataset(seed=DATA_SEED)
print(f"Train/val/test sizes: {len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])}")
print(f"Variance kept by 4 PCA components: {data['explained_variance']:.1%}")

logreg = LogisticRegression(random_state=DATA_SEED).fit(data["X_train"], data["y_train"])
logreg_val_acc = logreg.score(data["X_val"], data["y_val"])
print(f"\nLogistic regression  val acc: {logreg_val_acc:.3f}")

print("\nManual QNN architecture:")
print(describe_architecture(MANUAL_BASELINE))
weights, bias, losses = train_classifier(MANUAL_BASELINE, data["X_train"], data["y_train"], seed=TRAIN_SEED)
print(f"Training loss: start {losses[0]:.3f} -> end {losses[-1]:.3f}")
print(qml.draw(make_classifier(MANUAL_BASELINE), decimals=2)(weights, data["X_val"][0]))

qnn = evaluate_architecture(MANUAL_BASELINE, data, seed=TRAIN_SEED)
print(f"\nManual QNN  train acc: {qnn['train_acc']:.3f}  val acc: {qnn['val_acc']:.3f}  "
      f"noisy val acc (p={SEARCH_NOISE}): {qnn['noisy_val_acc']:.3f}")
print(f"Depth {qnn['depth']}, gates {qnn['gate_count']} ({qnn['two_qubit_gates']} CNOTs), "
      f"trainable params {qnn['n_params']} + 1 bias")

save_json({
    "data_seed": DATA_SEED,
    "train_seed": TRAIN_SEED,
    "logistic_regression_val_acc": logreg_val_acc,
    "manual_qnn": qnn,
    "manual_qnn_loss_history": losses,
    "environment": environment_info(),
}, out_dir / "baselines.json")
print(f"\nSaved {out_dir / 'baselines.json'}")
