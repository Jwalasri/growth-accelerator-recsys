"""Unit tests for the Growth Accelerator recommender project."""

from pathlib import Path

import pandas as pd

from data.generate_transactions import generate_interactions
from src.train import build_matrix, train_svd
from src.infer import recommend


def test_generate_interactions() -> None:
    df = generate_interactions(num_users=10, num_items=5, density=0.2, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"user_id", "item_id"}
    # There should be at most users*items*0.2 interactions
    assert len(df) <= 10 * 5


def test_train_and_recommend() -> None:
    # Build small interaction dataset
    df = generate_interactions(num_users=20, num_items=10, density=0.1, seed=1)
    mat = build_matrix(df)
    factors = train_svd(mat["X"], n_factors=5, random_state=0)
    model = {
        "user_ids": mat["user_ids"],
        "item_ids": mat["item_ids"],
        "user_factors": factors["user_factors"],
        "item_factors": factors["item_factors"],
    }
    # Pick a valid user ID
    user_id = model["user_ids"][0]
    recs = recommend(model, user_id, n=3)
    assert isinstance(recs, list)
    assert len(recs) == 3
    for item_id, score in recs:
        assert item_id in model["item_ids"]