import numpy as np

from src.data import load_dataset


def test_split_sizes_and_shapes():
    data = load_dataset(n_features=4, seed=42)
    assert data["X_train"].shape == (341, 4)
    assert data["X_val"].shape == (114, 4)
    assert data["X_test"].shape == (114, 4)


def test_features_are_valid_angles_and_labels_are_plus_minus_one():
    data = load_dataset(seed=42)
    for split in ["train", "val", "test"]:
        X, y = data[f"X_{split}"], data[f"y_{split}"]
        assert X.min() >= 0 and X.max() <= np.pi
        assert set(np.unique(y)) == {-1, 1}


def test_splits_are_stratified():
    data = load_dataset(seed=42)
    train_share = np.mean(data["y_train"] == 1)
    for split in ["val", "test"]:
        assert abs(np.mean(data[f"y_{split}"] == 1) - train_share) < 0.03


def test_same_seed_gives_same_data():
    a, b = load_dataset(seed=1), load_dataset(seed=1)
    assert np.array_equal(a["X_test"], b["X_test"])
