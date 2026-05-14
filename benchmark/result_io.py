# benchmark/result_io.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _json_safe(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save_summary(summary: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)


def save_history(
    path: str | Path,
    *,
    gibbs_calls,
    F_list,
    e_tr_list,
    times,
    dual_obj: Optional[list] = None,
    grad_norm: Optional[list] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(F_list)

    data = {
        "idx": list(range(n)),
        "gibbs_calls": list(gibbs_calls),
        "F_marg": list(F_list),
        "e_tr": list(e_tr_list),
        "time_sec": list(times),
    }

    if dual_obj is not None:
        data["dual_obj"] = list(dual_obj[:n])
    if grad_norm is not None:
        data["grad_norm"] = list(grad_norm[:n])

    pd.DataFrame(data).to_csv(path, index=False)