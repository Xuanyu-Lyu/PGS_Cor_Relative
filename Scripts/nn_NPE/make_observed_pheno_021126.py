"""
make_observed_pheno_021126.py
=============================
Write observed_correlations_pheno_021126.csv: the updated observed EA phenotypic correlations
(eapheno_021126.csv), with each relationship named in the path notation of the simulations and
the training data.

eapheno_021126.csv names relationships as ea4_new_old.csv does (steps joined by "_": par, chi,
sib, spo), so the translation is the one of make_observed_ea4new.py (see there); the file must
hold the same 26 relationship types as observed_correlations_PGS_ea4new.csv.

Usage:
    python make_observed_pheno_021126.py
"""

from pathlib import Path

import pandas as pd

from make_observed_ea4new import OUT as PGS_FILE, TRAINING_TYPES, to_path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "eapheno_021126.csv"
OUT = HERE / "observed_correlations_pheno_021126.csv"


def main():
    src = pd.read_csv(SOURCE, encoding="utf-8-sig")
    src["RelType"] = src["type"].map(to_path)

    missing = sorted(set(TRAINING_TYPES) - set(src["RelType"]))
    if missing:
        raise ValueError(f"training relationship types missing from {SOURCE.name}: {missing}")
    if src["RelType"].duplicated().any():
        raise ValueError(f"two rows map to the same path: {src[src['RelType'].duplicated(keep=False)]}")
    pgs_types = set(pd.read_csv(PGS_FILE)["RelType"])
    if set(src["RelType"]) != pgs_types:
        raise ValueError(f"relationship types differ from {PGS_FILE.name}: "
                         f"{sorted(set(src['RelType']) ^ pgs_types)}")

    out = src.rename(columns={"cor": "Correlation", "type": "source_type"})
    out = out[["RelType", "Correlation", "SE", "source_type"]]
    out.to_csv(OUT, index=False)
    extra = [r for r in out["RelType"] if r not in TRAINING_TYPES]
    print(f"wrote {OUT.name}: {len(out)} relationship types "
          f"({len(out) - len(extra)} in the training data, {len(extra)} new: {', '.join(extra)})")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
