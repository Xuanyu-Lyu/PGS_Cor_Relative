"""
make_observed_ea4new.py
=======================
Write observed_correlations_PGS_ea4new.csv: the observed EA PGS correlations computed with
the updated EA4 PGS (the ctype == "new" rows of ea4_new_old.csv), with each relationship
named in the path notation of the simulations and the training data.

ea4_new_old.csv names a relationship by its steps from a person to the relative, joined by
"_": par (parent), chi (child), sib (full sibling), spo (spouse). relationship_finder.py
uses one letter per step, read in the same order: P, C, S, M. For example spo_sib_chi
(spouse's sibling's child) is MSC. Pairs are unordered, so a path and its reverse (steps in
reverse order, P <-> C) are the same relationship; where the project already uses the
reverse, that name is kept (spo_sib_spo_sib = MSMS -> SMSM).

Usage:
    python make_observed_ea4new.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ea4_new_old.csv"
OUT = HERE / "observed_correlations_PGS_ea4new.csv"

STEP = {"par": "P", "chi": "C", "sib": "S", "spo": "M"}
INVERSE = {"P": "C", "C": "P", "S": "S", "M": "M"}
# relationship types of the training data (and of observed_correlations_PGS.csv)
TRAINING_TYPES = ["S", "PSC", "PPSCC", "M", "MS", "SMS", "MSC", "MSM", "SMSC", "SMSM",
                  "SMSMS", "MSMSM", "MSMSC", "PSMSC", "SMSMSC", "MSMSMS"]


def to_path(name):
    path = "".join(STEP[s] for s in name.split("_"))
    reverse = "".join(INVERSE[s] for s in reversed(path))
    return reverse if path not in TRAINING_TYPES and reverse in TRAINING_TYPES else path


def main():
    src = pd.read_csv(SOURCE, encoding="utf-8-sig")
    new = src[src["ctype"] == "new"].copy()
    new["RelType"] = new["type"].map(to_path)

    # the file's own step counts must agree with the translation
    bio = new["RelType"].str.count("[PCS]")
    spo = new["RelType"].str.count("M")
    bad = new[(bio != new["bio"]) | (spo != new["spo"])]
    if len(bad):
        raise ValueError(f"step counts do not match bio/spo:\n{bad}")
    missing = sorted(set(TRAINING_TYPES) - set(new["RelType"]))
    if missing:
        raise ValueError(f"training relationship types missing from {SOURCE.name}: {missing}")
    if new["RelType"].duplicated().any():
        raise ValueError(f"two rows map to the same path: {new[new['RelType'].duplicated(keep=False)]}")

    out = new.rename(columns={"cor": "Correlation", "npairs": "N_Pairs", "type": "ea4_type"})
    out = out[["RelType", "Correlation", "SE", "N_Pairs", "ea4_type"]]
    out.to_csv(OUT, index=False)
    extra = [r for r in out["RelType"] if r not in TRAINING_TYPES]
    print(f"wrote {OUT.name}: {len(out)} relationship types "
          f"({len(out) - len(extra)} in the training data, {len(extra)} new: {', '.join(extra)})")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()