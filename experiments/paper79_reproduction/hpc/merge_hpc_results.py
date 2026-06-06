"""Merge HPC benchmark CSVs into one summary table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def fieldnames_union(rows: Iterable[Dict[str, str]]) -> List[str]:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Merge paper79 reproduction HPC CSV results")
    parser.add_argument("--indir", type=Path, default=Path("results") / "hpc_qubit_mixed")
    parser.add_argument("--pattern", default="**/*.csv")
    parser.add_argument("--out", type=Path, default=Path("results") / "hpc_qubit_mixed_summary.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = sorted(p for p in args.indir.glob(args.pattern) if not p.name.endswith(".partial.jsonl"))
    rows: List[Dict[str, str]] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["source_csv"] = str(path)
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames_union(rows)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(files)} files and {len(rows)} rows into {args.out}")


if __name__ == "__main__":
    main()
