"""Small utilities shared by experiment runners.

These helpers are intentionally kept out of the numerical solver modules.
They handle CSV argument parsing, first-hit reporting, checkpoint files, and
optional final-state persistence for post-processing tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def parse_csv_ints(spec: str, *, all_values: Optional[Sequence[int]] = None) -> List[int]:
    """Parse ``"1,2,5"`` or ranges such as ``"3-7"``.

    If ``spec == "all"``, ``all_values`` must be supplied.
    """

    spec = spec.strip().lower()
    if spec == "all":
        if all_values is None:
            raise ValueError("'all' requires all_values.")
        return [int(x) for x in all_values]

    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, stop = int(a), int(b)
            step = 1 if start <= stop else -1
            out.extend(range(start, stop + step, step))
        else:
            out.append(int(part))
    return out


def parse_csv_floats(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def parse_csv_strings(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def tol_label(tol: float) -> str:
    """Return a compact tolerance label, e.g. ``1e-4 -> 1em04``."""

    return f"{float(tol):.0e}".replace("-", "m").replace("+", "p")


def safe_filename(text: Any) -> str:
    """Turn a method label or instance label into a filesystem-safe token."""

    raw = str(text)
    raw = raw.replace("eta=eps/N", "eta_eps_over_N")
    raw = raw.replace("eta=eps", "eta_eps")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "value"


def first_hit_index(values: Sequence[float], tol: float) -> int:
    """Return the first index where ``values <= tol``, or ``-1`` if absent."""

    arr = np.asarray(values, dtype=float)
    hit = np.where(arr <= float(tol))[0]
    return int(hit[0]) if hit.size else -1


def final_state_dir(args: argparse.Namespace) -> Path:
    """Default directory for final-state ``.npz`` files."""

    state_dir = getattr(args, "state_dir", None)
    if state_dir is not None:
        return Path(state_dir)
    out = Path(getattr(args, "out", Path("results") / "experiment.csv"))
    return out.parent / f"{out.stem}_states"


def save_final_state(
    *,
    args: argparse.Namespace,
    row: Dict[str, Any],
    res: Any,
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    dims: Sequence[int],
    eps: float,
) -> str:
    """Persist final coupling and potentials for post-processing.

    The function returns the path as a string, or ``""`` if saving is disabled
    or the result object does not contain both ``pi`` and ``U_list``.
    """

    if not getattr(args, "save_final_state", False):
        return ""
    if res is None:
        return ""
    pi = getattr(res, "pi", None)
    U_list = getattr(res, "U_list", None)
    if pi is None or U_list is None:
        return ""

    out_dir = final_state_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        safe_filename(row.get("experiment", "exp")),
        safe_filename(row.get("paper79_label", row.get("paper79_index", "instance"))),
        f"eps{safe_filename(f'{float(eps):.0e}')}",
        safe_filename(row.get("method", "method")),
    ]
    if row.get("M_inner") not in (None, ""):
        parts.append(f"M{safe_filename(row['M_inner'])}")
    path = out_dir / ("__".join(parts) + ".npz")

    payload: Dict[str, Any] = {
        "pi": np.asarray(pi),
        "H": np.asarray(H),
        "dims": np.asarray(dims, dtype=int),
        "eps": np.asarray(float(eps)),
        "metadata_json": np.asarray(json.dumps(row, sort_keys=True, default=str)),
    }
    for i, gamma in enumerate(gammas):
        payload[f"gamma_{i}"] = np.asarray(gamma)
    for i, Ui in enumerate(U_list):
        payload[f"U_{i}"] = np.asarray(Ui)

    np.savez_compressed(path, **payload)
    return str(path)


def fieldnames_union(rows: Iterable[Dict[str, Any]], *, preferred: Optional[Sequence[str]] = None) -> List[str]:
    """Stable CSV field order: preferred columns first, then first-seen extras."""

    names: List[str] = []
    seen = set()
    for name in preferred or ():
        if name not in seen:
            seen.add(name)
            names.append(name)
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def write_csv(path: Path, rows: List[Dict[str, Any]], *, preferred: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames_union(rows, preferred=preferred)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def default_checkpoint_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".partial.jsonl")


def reset_checkpoint(path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_checkpoint(path: Optional[Path], rows: List[Dict[str, Any]]) -> None:
    if path is None or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        f.flush()


__all__ = [
    "append_checkpoint",
    "default_checkpoint_path",
    "fieldnames_union",
    "final_state_dir",
    "first_hit_index",
    "parse_csv_floats",
    "parse_csv_ints",
    "parse_csv_strings",
    "reset_checkpoint",
    "safe_filename",
    "save_final_state",
    "tol_label",
    "write_csv",
]
