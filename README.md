# CircuitPilot

A small project using a genetic algorithm to search for quantum neural network (QNN) architectures for binary classification.

## Goal

Find circuits that balance accuracy, circuit size, and robustness to simulated noise. Compare genetic search with random search and a manually designed circuit.

## How the Project Works

1. **Data preparation:** Uses the breast cancer dataset from scikit-learn. Standardization and PCA reduce the input to four features, encoded into a four-qubit circuit.

2. **Circuit generation:** PennyLane builds quantum neural networks using rotation gates (RX, RY, RZ) and CNOT connections. Each candidate has a different architecture.

3. **Architecture search:** A genetic algorithm evolves the circuits through selection, crossover, and mutation, searching for better designs.

4. **Training and evaluation:** Each circuit is trained using Adam. Its fitness depends on validation accuracy, circuit complexity, and accuracy loss under simulated depolarizing noise.

5. **Comparison:** The best evolved circuit is compared with random search, a manually designed QNN, and logistic regression.

6. **Final testing:** Selected models are evaluated on unseen data at different noise levels to measure their accuracy and robustness.

## Technologies

Python, PennyLane, NumPy, scikit-learn, Matplotlib, and pytest.

## Installation and Running

Activate the virtual environment, then install the dependencies:

.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Run the setup check and tests:

python experiments/check_setup.py
pytest -q

Run the experiments in order:

python experiments/run_baselines.py
python experiments/run_search.py --run-name main
python experiments/final_evaluation.py --run-name main
python experiments/make_plots.py --run-name main

For a quick test run:

python experiments/run_search.py --quick --run-name quick

Results and plots are saved in `results/main/` or `results/quick/`, depending on the run name.
