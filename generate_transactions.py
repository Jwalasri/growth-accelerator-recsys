"""Generate a synthetic user‑item interactions dataset for recommender systems.

This script produces a CSV file with columns ``user_id`` and ``item_id``
representing implicit feedback events (e.g. purchases or clicks).  You can
control the number of users, number of items and the sparsity of the
interactions via command line arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd


def generate_interactions(num_users: int, num_items: int, density: float, seed: int | None) -> pd.DataFrame:
    """Generate a DataFrame of implicit user–item interactions.

    Parameters
    ----------
    num_users:
        Number of unique users to simulate.
    num_items:
        Number of unique items to simulate.
    density:
        Approximate fraction of the user–item matrix that will have
        interactions.  Must be between 0 and 1.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    DataFrame
        A table with two columns: ``user_id`` and ``item_id``.
    """
    if density <= 0 or density > 1:
        raise ValueError("density must be in (0, 1]")
    rng = random.Random(seed)
    interactions = []
    # Compute number of interactions to generate
    total_possible = num_users * num_items
    num_interactions = int(total_possible * density)
    # Use a set to avoid duplicates
    seen = set()
    while len(interactions) < num_interactions:
        u = rng.randrange(num_users)
        i = rng.randrange(num_items)
        if (u, i) not in seen:
            seen.add((u, i))
            interactions.append({"user_id": u, "item_id": i})
    return pd.DataFrame(interactions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic user–item interaction data.")
    parser.add_argument("--users", type=int, default=1000, help="Number of users to generate.")
    parser.add_argument("--items", type=int, default=100, help="Number of items to generate.")
    parser.add_argument(
        "--density",
        type=float,
        default=0.01,
        help="Fraction of the user–item matrix to fill with interactions (0 < density ≤ 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "transactions.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    df = generate_interactions(args.users, args.items, args.density, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} interactions to {output_path}")


if __name__ == "__main__":
    main()