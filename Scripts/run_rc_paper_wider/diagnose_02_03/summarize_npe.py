"""Collect the posterior summaries of the production fits and the refits of run_npe_variants.py."""
import json

import pandas as pd

from common import NPE, RESULTS

RUNS = {"production": "results_npe_unweighted_{cond}", "vgwin": None, "noise": None, "both": None}


def load_all():
    rows = []
    for cond in ["02AElatentAM_wider", "03EnvLatentAM_wider"]:
        for variant in RUNS:
            base = NPE / f"results_npe_unweighted_{cond}" if variant == "production" else RESULTS / "npe" / f"{cond}_{variant}"
            summary = base / "predictions" / "posterior_summary.csv"
            if not summary.exists():
                continue
            sampling = json.loads((base / "predictions" / "posterior_sampling.json").read_text())
            for _, r in pd.read_csv(summary).iterrows():
                rows.append(dict(condition=cond.split("_")[0], variant=variant, parameter=r["parameter"], mean=r["mean"],
                                 sd=r["std"], inside=sampling.get("share_of_unconstrained_flow_inside_prior"),
                                 method=sampling.get("sampling_method")))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    d = load_all()
    print(d.groupby(["condition", "variant"])[["inside"]].first().round(4).join(d.groupby(["condition", "variant"])["method"].first()))
    d["est"] = d["mean"].map("{:.3f}".format) + " ± " + d["sd"].map("{:.3f}".format)
    print(d.pivot_table(index=["condition", "parameter"], columns="variant", values="est", aggfunc="first").to_string())
