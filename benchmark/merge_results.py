# benchmark/merge_results.py

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def main():
    rows = []
    for path in Path("results/raw").glob("*_summary.json"):
        with open(path) as f:
            rows.append(json.load(f))

    Path("results/merged").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("results/merged/results_all.csv", index=False)


if __name__ == "__main__":
    main()