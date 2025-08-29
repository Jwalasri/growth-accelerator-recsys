"""Inference script for the Growth Accelerator recommender system.

Given a trained SVD recommender model and a user identifier, this
script outputs the top‑N item recommendations along with their
similarity scores.  It is intended for demonstration purposes
rather than production use and does not handle unseen users.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np


def load_model(path: Path | str) -> dict:
    """Load a recommender model object from disk."""
    return joblib.load(path)


def recommend(model: dict, user_id: int, n: int = 5) -> List[Tuple[int, float]]:
    """Generate top‑N recommendations for a given user.

    Parameters
    ----------
    model:
        Dictionary with keys ``user_ids``, ``item_ids``, ``user_factors`` and ``item_factors``.
    user_id:
        The ID of the user to generate recommendations for.
    n:
        Number of recommendations to return.

    Returns
    -------
    list of tuple(int, float)
        A list of (item_id, score) pairs sorted by descending score.
    """
    user_ids = model["user_ids"]
    item_ids = model["item_ids"]
    user_factors = model["user_factors"]
    item_factors = model["item_factors"]
    try:
        user_index = user_ids.index(user_id)
    except ValueError:
        raise ValueError(f"Unknown user_id {user_id}. Available user_ids: {user_ids[:5]}...")
    # Compute cosine similarity via dot product (since factors are normalised)
    scores = item_factors @ user_factors[user_index]
    # Get top N indices
    top_indices = np.argsort(scores)[::-1][:n]
    recommendations: List[Tuple[int, float]] = []
    for idx in top_indices:
        recommendations.append((item_ids[idx], float(scores[idx])))
    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate recommendations for a given user.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (.joblib).")
    parser.add_argument("--user_id", type=int, required=True, help="User ID to generate recommendations for.")
    parser.add_argument("--n", type=int, default=5, help="Number of recommendations to output.")
    args = parser.parse_args()
    model = load_model(args.model)
    recs = recommend(model, args.user_id, n=args.n)
    for item_id, score in recs:
        print(f"Item {item_id}: score={score:.4f}")


if __name__ == "__main__":
    main()