# benchmark/make_jobs.py

from __future__ import annotations

from pathlib import Path
import pandas as pd


def main():
    samples = pd.read_csv("manifests/samples.csv")
    jobs = []

    methods = [
        {"method": "KL", "M_inner": None, "optimizer": None, "lr": None},
        {"method": "MD", "M_inner": 1, "optimizer": None, "lr": None},
        {"method": "MD", "M_inner": 2, "optimizer": None, "lr": None},
        {"method": "MD", "M_inner": 5, "optimizer": None, "lr": None},
        {"method": "SDPLab-Adam", "M_inner": None, "optimizer": "adam", "lr": 1e-2},
    ]

    job_id = 0
    for _, s in samples.iterrows():
        for m in methods:
            jid = f"j{job_id:06d}"
            jobs.append({
                "job_id": jid,
                "sample_id": s["sample_id"],
                "instance_file": s["instance_file"],
                "method": m["method"],
                "M_inner": m["M_inner"],
                "optimizer": m["optimizer"],
                "lr": m["lr"],
                "tol_F": 1e-8,
                "tol_tr": "",
                "max_gibbs_calls": 25000,
                "checkpoint_every": 10,
                "output_prefix": f"results/raw/{jid}",
            })
            job_id += 1

    Path("manifests").mkdir(exist_ok=True)
    pd.DataFrame(jobs).to_csv("manifests/jobs.csv", index=False)

    print(f"Wrote {len(jobs)} jobs to manifests/jobs.csv")
    print(pd.DataFrame(jobs)["method"].value_counts())


if __name__ == "__main__":
    main()