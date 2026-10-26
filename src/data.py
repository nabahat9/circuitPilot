import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_dataset(n_features: int = 4, seed: int = 42) -> dict:
    """The breast cancer dataset is reduced to n_features using PCA, then split into training, validation, and test sets (60/20/20).

     Features are scaled to \([0,\pi]\) to use as rotation angles in the quantum circuit. Labels are encoded as -1 for malignant and +1 for benign, matching the output range of a Pauli-Z measurement.
    """
    X, y = load_breast_cancer(return_X_y=True)
    y = 2 * y - 1

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=seed
    )

    # All preprocessing is fitted on the training split only, so no information
    # from validation or test data leaks into the features.
    standardizer = StandardScaler().fit(X_train)
    pca = PCA(n_components=n_features, random_state=seed).fit(standardizer.transform(X_train))
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(
        pca.transform(standardizer.transform(X_train))
    )

    def transform(X_raw):
        X_reduced = angle_scaler.transform(pca.transform(standardizer.transform(X_raw)))
        # Val/test points can fall slightly outside the training range.
        return np.clip(X_reduced, 0, np.pi)

    return {
        "X_train": transform(X_train),
        "y_train": y_train,
        "X_val": transform(X_val),
        "y_val": y_val,
        "X_test": transform(X_test),
        "y_test": y_test,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
