"""Training script for the Growth Accelerator recommender system.

This script reads a CSV file of implicit user–item interactions and
learns latent factors using truncated singular value decomposition
(SVD).  The resulting model (user and item factors along with the
corresponding user and item identifiers) is saved to a joblib file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def build_matrix(df: pd.DataFrame) -> Dict[str, any]:
    """Construct a sparse user–item matrix from interactions.

    Parameters
    ----------
    df:
        DataFrame with columns ``user_id`` and ``item_id``.

    Returns
    -------
    dict
        A dictionary containing ``X`` (2D NumPy array), ``user_ids`` and
        ``item_ids``.
    """
    # Get unique users and items
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    user_index = {u: idx for idx, u in enumerate(user_ids)}
    item_index = {i: idx for idx, i in enumerate(item_ids)}
    # Initialise matrix
    X = np.zeros((len(user_ids), len(item_ids)), dtype=float)
    for _user_id, _item_id in zip(df["user_id"], df["item_id"]):
        X[user_index[_user_id], item_index[_item_id]] += 1.0
    return {"X": X, "user_ids": user_ids, "item_ids": item_ids}


def train_svd(X: np.ndarray, n_factors: int = 50, random_state: int = 42) -> Dict[str, np.ndarray]:
    """Train a TruncatedSVD model on the user–item matrix.

    Parameters
    ----------
    X:
        User–item interaction matrix of shape (n_users, n_items).
    n_factors:
        Number of latent factors to compute.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary with keys ``user_factors`` and ``item_factors``.
    """
    # Decompose X using SVD.  We centre the matrix by subtracting the
    # mean of each row to mitigate user bias.
    user_means = X.mean(axis=1, keepdims=True)
    X_centered = X - user_means
    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    user_factors = svd.fit_transform(X_centered)
    item_factors = svd.components_.T  # shape (n_items, n_factors)
    # Normalise factors to unit length for cosine similarity
    user_norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
    item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    user_factors = user_factors / np.maximum(user_norms, 1e-8)
    item_factors = item_factors / np.maximum(item_norms, 1e-8)
    return {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_means": user_means.flatten(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a truncated SVD recommender.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "transactions.csv"),
        help="Path to the CSV file of interactions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="svd",
        help="Identifier for the model to train (currently only 'svd').",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=50,
        help="Number of latent factors to compute.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory to write the trained model to.",
    )
    args = parser.parse_args()
    if args.model != "svd":
        raise ValueError("Only 'svd' model is currently supported.")
    df = pd.read_csv(args.data)
    matrix_data = build_matrix(df)
    factors = train_svd(matrix_data["X"], n_factors=args.factors)
    model_obj = {
        "user_ids": matrix_data["user_ids"],
        "item_ids": matrix_data["item_ids"],
        "user_factors": factors["user_factors"],
        "item_factors": factors["item_factors"],
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model}.joblib"
    joblib.dump(model_obj, model_path)
    print(f"Trained {args.model} model with {args.factors} factors and saved to {model_path}")


if __name__ == "__main__":
    main()