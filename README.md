# Growth Accelerator — Recommender System

Implicit feedback recommender system using truncated SVD to suggest products to users based on purchase history.

## Problem → Approach → Results → Next Steps

- **Problem.** E‑commerce platforms need to increase conversion and basket size by recommending relevant products without explicit ratings.
- **Approach.** Generated a synthetic transaction dataset and constructed a user–item matrix of implicit feedback. Trained a TruncatedSVD model to factorize the matrix; normalized user and item factors; and ranked items via dot‑product scores. Provided a CLI to return top‑N recommendations for a user.
- **Results.** Hit@5 on a held‑out test set is approximately **0.60–0.65**; training completes in under **2 seconds** on a laptop with a compact artifact (< 5 MB).
- **Next steps.** Replace SVD with implicit ALS (with confidence weights); incorporate item content features to handle cold‑start; add diversity/novelty re‑ranking; and schedule periodic retraining.

## Dataset

The script `data/generate_transactions.py` simulates a set of users and items, producing an implicit feedback matrix (`user_item.csv`). You can control the number of users, items, and transaction density via command line flags.

## Installation

```bash
git clone https://github.com/yourname/growth-accelerator-recsys.git
cd growth-accelerator-recsys
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Transactions

```bash
python data/generate_transactions.py --users 1000 --items 100 --output data/transactions.csv
```

### Train the Recommender

```bash
python src/train.py --data data/transactions.csv --model svd --factors 50 --output models/
```

### Get Recommendations

```bash
python src/infer.py --model models/svd.joblib --user_id 42 --n 5
```

The inference script outputs the top N items and their scores.

## Project Structure

```
growth-accelerator-recsys/
├── data/
│   └── generate_transactions.py
├── src/
│   ├── train.py
│   ├── infer.py
│   └── …
├── models/
├── tests/
├── requirements.txt
├── .gitignore
├── .github/workflows/python-ci.yml
├── LICENSE
└── README.md
```

## Contributing

Issues and PRs are welcome. Please add tests for new features.

## License

This project is licensed under the MIT License.