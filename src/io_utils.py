import json
import platform
from pathlib import Path

import numpy as np
import pennylane as qml
import sklearn


def results_dir(run_name: str = "") -> Path:
    """results/ for shared outputs, results/<run_name>/ for one search run."""
    path = Path("results") / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_name_argument(parser) -> None:
    parser.add_argument("--run-name", default="main",
                        help="results are read from / written to results/<run-name>/")


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "pennylane": qml.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier experiment scripts first.")
    with open(path) as f:
        return json.load(f)
