"""
test_simulation.py
==================
Tests for AssortativeMatingSimulation (base class) and
IslandMigrationSimulation, covering:

  1. Baseline AM — no islands, no mtDNA (sanity check)
  2. mtDNA component — verify maternal transmission, no paternal
     signal, diagonal-only validation
  3. Island migration — verify stratification (Y2 gradient across
     islands), within-island AM, offspring island assignment
  4. Island + mtDNA combined — both features active simultaneously
  5. Burn-in — confirm recorded history length is unaffected by
     burn-in, population has had time to evolve
  6. Edge cases — mt_mat off-diagonal raises ValueError; pop_size
     not divisible by n_islands*2 raises ValueError

Run with:
    python test_simulation.py
or:
    python -m pytest test_simulation.py -v
"""

import sys
import os
import traceback

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from any directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from SimulationFunctions import AssortativeMatingSimulation
from IslandSimulation import IslandMigrationSimulation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_matrices(h2_1=0.45, h2_2=0.40, re=0.0):
    """Return a standard set of 2×2 model matrices."""
    ve_1 = 1.0 - h2_1
    ve_2 = 1.0 - h2_2
    cove_off = re * np.sqrt(ve_1 * ve_2)
    return dict(
        cove_mat=np.array([[ve_1, cove_off], [cove_off, ve_2]]),
        f_mat=np.zeros((2, 2)),
        s_mat=np.zeros((2, 2)),
        a_mat=np.zeros((2, 2)),
        d_mat=np.diag([np.sqrt(h2_1), np.sqrt(h2_2)]),
        covy_mat=np.eye(2),
        k2_matrix=np.zeros((2, 2)),
    )


def _base_matrices_with_mt(h2_1=0.45, h2_2=0.40, mt1=0.30, mt2=0.20):
    """Return matrices with cove_mat shrunk to accommodate mtDNA variance."""
    mt_var_1 = mt1 ** 2  # approx Var(MT1) ≈ mt1² × Var(Y1=1)
    mt_var_2 = mt2 ** 2
    ve_1 = max(1.0 - h2_1 - mt_var_1, 0.05)
    ve_2 = max(1.0 - h2_2 - mt_var_2, 0.05)
    return dict(
        cove_mat=np.diag([ve_1, ve_2]),
        f_mat=np.zeros((2, 2)),
        s_mat=np.zeros((2, 2)),
        a_mat=np.zeros((2, 2)),
        d_mat=np.diag([np.sqrt(h2_1), np.sqrt(h2_2)]),
        covy_mat=np.eye(2),
        k2_matrix=np.zeros((2, 2)),
        mt_mat=np.diag([mt1, mt2]),
    )


# Simple pass/fail tracker
_pass = 0
_fail = 0
_results = []

def _check(label, condition, detail=""):
    global _pass, _fail
    status = "PASS" if condition else "FAIL"
    if condition:
        _pass += 1
    else:
        _fail += 1
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"\n         {detail}"
    _results.append(msg)
    print(msg)


# ===========================================================================
# Test 1 — Baseline AM (no islands, no mtDNA)
# ===========================================================================

def test_baseline_am():
    print("\n=== Test 1: Baseline AM (no islands, no mtDNA) ===")
    mats = _base_matrices()
    N_GEN = 3
    N = 2000
    am_list = [0.4] * N_GEN

    sim = AssortativeMatingSimulation(
        n_CV=500, rg_effects=0.4,
        maf_min=0.01, maf_max=0.50,
        num_generations=N_GEN, pop_size=N,
        mating_type="phenotypic",
        mate_on_trait=1,
        am_list=am_list,
        seed=1,
        **mats,
    )
    results = sim.run_simulation()

    # Basic structure
    _check("run_simulation returns dict",
           isinstance(results, dict))
    _check("SUMMARY.RES has correct length (gen0 + N_GEN)",
           len(results['SUMMARY.RES']) == N_GEN + 1,
           f"got {len(results['SUMMARY.RES'])}, expected {N_GEN + 1}")
    _check("PHEN is a DataFrame",
           isinstance(results['PHEN'], pd.DataFrame))
    _check("PHEN has expected columns Y1, Y2, AO1, AO2",
           all(c in results['PHEN'].columns for c in ['Y1','Y2','AO1','AO2']))
    _check("pop size approximately correct",
           abs(len(results['PHEN']) - N) <= N * 0.10,
           f"got {len(results['PHEN'])}")

    # MT columns should exist and be all zero (no mt_mat supplied)
    phen = results['PHEN']
    _check("MT1 and MT2 columns present",
           'MT1' in phen.columns and 'MT2' in phen.columns)
    _check("MT1 and MT2 are all zero when mt_mat not supplied",
           np.allclose(phen['MT1'].astype(float).fillna(0), 0) and
           np.allclose(phen['MT2'].astype(float).fillna(0), 0))

    # h2.mt should be (near-)zero
    last = results['SUMMARY.RES'][-1]
    _check("h2.mt near zero when mt_mat not supplied",
           all(abs(v) < 0.01 for v in last.get('h2.mt', [1, 1])),
           f"h2.mt = {last.get('h2.mt')}")


# ===========================================================================
# Test 2 — mtDNA component
# ===========================================================================

def test_mtdna():
    print("\n=== Test 2: mtDNA maternal transmission ===")
    mt1_path, mt2_path = 0.35, 0.25
    mats = _base_matrices_with_mt(mt1=mt1_path, mt2=mt2_path)
    N_GEN = 5
    N = 4000
    am_list = [0.4] * N_GEN

    sim = AssortativeMatingSimulation(
        n_CV=500, rg_effects=0.3,
        maf_min=0.01, maf_max=0.50,
        num_generations=N_GEN, pop_size=N,
        mating_type="phenotypic",
        mate_on_trait=1,
        am_list=am_list,
        seed=2,
        **mats,
    )
    results = sim.run_simulation()

    phen   = results['PHEN'].copy()
    last   = results['SUMMARY.RES'][-1]

    # VMT and h2.mt should be non-trivial
    _check("VMT present in SUMMARY.RES",
           'VMT' in last)
    vmt = np.array(last['VMT'])
    _check("VMT diagonal is positive (mtDNA has variance)",
           vmt[0, 0] > 0 and vmt[1, 1] > 0,
           f"VMT = {vmt}")
    h2mt = last.get('h2.mt', [0, 0])
    _check("h2.mt trait1 > 0.01",
           h2mt[0] > 0.01,
           f"h2.mt = {h2mt}")
    _check("h2.mt trait2 > 0.01",
           h2mt[1] > 0.01,
           f"h2.mt = {h2mt}")

    # Verify maternal inheritance in final generation
    # offspring MT1 should correlate with mother's Y1
    mother_map = (phen[phen['Mother.ID'].notna()]
                  .copy()
                  .assign(Mother_ID_int=lambda d: d['Mother.ID'].astype(float).astype('Int64')))

    # Build a lookup from ID → Y1, Y2
    id_lookup = phen.set_index('ID')[['Y1', 'Y2']].astype(float)

    history = results['HISTORY']
    if history is not None:
        # history PHEN[0] is founders, last entry is the final gen
        # Use final history generation for parent look-up
        prev_phen = history['PHEN'][-2].set_index('ID')[['Y1', 'Y2']].astype(float)
        children = history['PHEN'][-1].copy()
        children_with_mom = children[children['Mother.ID'].notna()].copy()
        children_with_mom['mY1'] = children_with_mom['Mother.ID'].astype(float).map(prev_phen['Y1'])
        children_with_mom['mY2'] = children_with_mom['Mother.ID'].astype(float).map(prev_phen['Y2'])
        children_with_mom['fY1'] = children_with_mom['Father.ID'].astype(float).map(prev_phen['Y1'])
        children_with_mom['fY2'] = children_with_mom['Father.ID'].astype(float).map(prev_phen['Y2'])
        ok_rows = children_with_mom.dropna(subset=['MT1','MT2','mY1','mY2','fY1','fY2'])

        if len(ok_rows) >= 30:
            mt1_vals = ok_rows['MT1'].astype(float).values
            mt2_vals = ok_rows['MT2'].astype(float).values
            mY1      = ok_rows['mY1'].values
            mY2      = ok_rows['mY2'].values
            fY1      = ok_rows['fY1'].values
            fY2      = ok_rows['fY2'].values

            cor_mt1_mY1 = np.corrcoef(mt1_vals, mY1)[0, 1]
            cor_mt2_mY2 = np.corrcoef(mt2_vals, mY2)[0, 1]
            cor_mt1_fY1 = np.corrcoef(mt1_vals, fY1)[0, 1]
            cor_mt2_fY2 = np.corrcoef(mt2_vals, fY2)[0, 1]

            _check("MT1 ~ mother Y1 correlation is strongly positive (≥0.15)",
                   cor_mt1_mY1 >= 0.15,
                   f"cor(MT1, mother_Y1) = {cor_mt1_mY1:.4f}")
            _check("MT2 ~ mother Y2 correlation is strongly positive (≥0.15)",
                   cor_mt2_mY2 >= 0.15,
                   f"cor(MT2, mother_Y2) = {cor_mt2_mY2:.4f}")
            _check("MT1 ~ father Y1 correlation weaker than maternal (no paternal mtDNA)",
                   abs(cor_mt1_fY1) < abs(cor_mt1_mY1),
                   f"cor(MT1, father_Y1)={cor_mt1_fY1:.4f}  cor(MT1, mother_Y1)={cor_mt1_mY1:.4f}")
            _check("MT2 ~ father Y2 correlation weaker than maternal",
                   abs(cor_mt2_fY2) < abs(cor_mt2_mY2),
                   f"cor(MT2, father_Y2)={cor_mt2_fY2:.4f}  cor(MT2, mother_Y2)={cor_mt2_mY2:.4f}")
        else:
            _check("Enough rows to check parent-offspring mtDNA correlation", False,
                   f"only {len(ok_rows)} valid rows")

    # Diagonal-only validation
    try:
        bad_mt = np.array([[0.3, 0.1], [0.0, 0.2]])  # off-diagonal non-zero
        AssortativeMatingSimulation(
            n_CV=100, rg_effects=0.3,
            maf_min=0.01, maf_max=0.50,
            num_generations=1, pop_size=200,
            mating_type="phenotypic",
            mate_on_trait=1,
            am_list=[0.4],
            seed=0,
            mt_mat=bad_mt,
            **_base_matrices(),
        )
        _check("Non-diagonal mt_mat raises ValueError", False,
               "No error raised — should have raised ValueError")
    except ValueError as e:
        _check("Non-diagonal mt_mat raises ValueError", True, str(e))


# ===========================================================================
# Test 3 — Island migration (no mtDNA)
# ===========================================================================

def test_island_migration():
    print("\n=== Test 3: Island migration ===")
    N_ISLANDS = 5
    N_PER_ISLAND = 400   # total = 2000, divisible by n_islands*2=10
    N = N_PER_ISLAND * N_ISLANDS
    N_BURN = 10
    N_MAIN = 3
    mats = _base_matrices()

    sim = IslandMigrationSimulation(
        n_islands=N_ISLANDS,
        move_p=0.15,
        within_island_am=0.4,
        migration_trait=2,
        mating_trait=1,
        n_jobs=1,
        # base-class params
        n_CV=500, rg_effects=0.5,
        maf_min=0.01, maf_max=0.50,
        num_generations=N_MAIN,
        n_burn_in=N_BURN,
        pop_size=N,
        mating_type="phenotypic",
        am_list=[0.4] * N_MAIN,
        seed=42,
        **mats,
    )
    results = sim.run_simulation()

    phen = results['PHEN']

    # island_id column present
    _check("island_id column present in PHEN",
           'island_id' in phen.columns)

    # island IDs span 1..n_islands
    unique_ids = set(phen['island_id'].dropna().astype(int).unique())
    _check(f"island_id takes values 1..{N_ISLANDS}",
           unique_ids == set(range(1, N_ISLANDS + 1)),
           f"unique island_ids = {sorted(unique_ids)}")

    # After burn-in + main phase, Y2 means should increase with island_id
    # (migration_trait=2 → high-Y2 individuals accumulate on high islands)
    island_means = (phen.assign(island_id=phen['island_id'].astype(int))
                    .groupby('island_id')['Y2']
                    .mean()
                    .astype(float)
                    .sort_index())
    is_monotone = all(island_means.iloc[i] <= island_means.iloc[i+1]
                      for i in range(len(island_means) - 1))
    _check("Y2 island means increase with island_id (stratification)",
           is_monotone,
           f"Y2 means by island: {island_means.values.round(3).tolist()}")

    # island_summary() output
    summary = sim.island_summary()
    _check("island_summary returns DataFrame",
           isinstance(summary, pd.DataFrame))
    _check("island_summary has Y2_mean column",
           'Y2_mean' in summary.columns)

    # SUMMARY.RES length unchanged by burn-in
    # gen-0 entry + N_MAIN recorded generations
    _check("SUMMARY.RES length = 1 + N_MAIN (burn-in not recorded)",
           len(results['SUMMARY.RES']) == 1 + N_MAIN,
           f"got {len(results['SUMMARY.RES'])}, expected {1 + N_MAIN}")

    # Pop-size divisibility guard
    try:
        IslandMigrationSimulation(
            n_islands=N_ISLANDS,
            move_p=0.10,
            within_island_am=0.4,
            migration_trait=2,
            mating_trait=1,
            n_CV=100, rg_effects=0.3,
            maf_min=0.01, maf_max=0.50,
            num_generations=1,
            pop_size=N + 3,   # not divisible by n_islands*2
            mating_type="phenotypic",
            am_list=[0.4],
            seed=0,
            **mats,
        )
        _check("Indivisible pop_size raises ValueError", False,
               "No error raised — should have raised ValueError")
    except ValueError as e:
        _check("Indivisible pop_size raises ValueError", True, str(e))


# ===========================================================================
# Test 4 — Island + mtDNA combined
# ===========================================================================

def test_island_plus_mtdna():
    print("\n=== Test 4: Island migration + mtDNA ===")
    N_ISLANDS = 4
    N_PER_ISLAND = 500
    N = N_PER_ISLAND * N_ISLANDS   # 2000, divisible by 4*2=8
    N_BURN = 5
    N_MAIN = 3
    mt1_path, mt2_path = 0.25, 0.15
    mats = _base_matrices_with_mt(mt1=mt1_path, mt2=mt2_path)

    sim = IslandMigrationSimulation(
        n_islands=N_ISLANDS,
        move_p=0.10,
        within_island_am=0.4,
        migration_trait=2,
        mating_trait=1,
        n_jobs=1,
        n_CV=500, rg_effects=0.4,
        maf_min=0.01, maf_max=0.50,
        num_generations=N_MAIN,
        n_burn_in=N_BURN,
        pop_size=N,
        mating_type="phenotypic",
        am_list=[0.4] * N_MAIN,
        seed=7,
        **mats,
    )
    results = sim.run_simulation()
    phen = results['PHEN']
    last = results['SUMMARY.RES'][-1]

    _check("island_id column present (island+mtDNA)",
           'island_id' in phen.columns)
    _check("MT1 and MT2 columns present (island+mtDNA)",
           'MT1' in phen.columns and 'MT2' in phen.columns)

    # VMT positive
    vmt = np.array(last.get('VMT', [[0, 0], [0, 0]]))
    _check("VMT Trait1 positive in combined run",
           vmt[0, 0] > 0,
           f"VMT[0,0] = {vmt[0,0]:.4f}")

    # Stratification still present
    island_means = (phen.assign(island_id=phen['island_id'].astype(int))
                    .groupby('island_id')['Y2']
                    .mean()
                    .astype(float)
                    .sort_index())
    is_monotone = all(island_means.iloc[i] <= island_means.iloc[i+1]
                      for i in range(len(island_means) - 1))
    _check("Y2 stratification maintained with mtDNA active",
           is_monotone,
           f"Y2 means: {island_means.values.round(3).tolist()}")

    # MT1 values should not all be zero
    mt1_vals = phen['MT1'].astype(float)
    _check("MT1 values are non-zero (mtDNA transmitted)",
           mt1_vals.abs().max() > 0,
           f"MT1 range: [{mt1_vals.min():.4f}, {mt1_vals.max():.4f}]")


# ===========================================================================
# Test 5 — Burn-in without islands
# ===========================================================================

def test_burn_in_base():
    print("\n=== Test 5: Burn-in (base class, no islands) ===")
    mats = _base_matrices()
    N_BURN = 5
    N_MAIN = 3
    N = 1000
    am_list = [0.4] * N_MAIN

    # Run with burn-in
    sim_burn = AssortativeMatingSimulation(
        n_CV=300, rg_effects=0.3,
        maf_min=0.01, maf_max=0.50,
        num_generations=N_MAIN, n_burn_in=N_BURN,
        pop_size=N,
        mating_type="phenotypic",
        mate_on_trait=1,
        am_list=am_list,
        seed=10,
        **mats,
    )
    res_burn = sim_burn.run_simulation()

    # Run without burn-in (same seed)
    sim_no_burn = AssortativeMatingSimulation(
        n_CV=300, rg_effects=0.3,
        maf_min=0.01, maf_max=0.50,
        num_generations=N_MAIN, n_burn_in=0,
        pop_size=N,
        mating_type="phenotypic",
        mate_on_trait=1,
        am_list=am_list,
        seed=10,
        **mats,
    )
    res_no_burn = sim_no_burn.run_simulation()

    # Both should have same SUMMARY.RES length
    _check("Burn-in does not increase SUMMARY.RES length",
           len(res_burn['SUMMARY.RES']) == len(res_no_burn['SUMMARY.RES']),
           f"with burn-in: {len(res_burn['SUMMARY.RES'])}, "
           f"without: {len(res_no_burn['SUMMARY.RES'])}")

    # With burn-in, h2 in final gen should be higher (AM has had more time)
    h2_burn    = res_burn['SUMMARY.RES'][-1].get('h2', [0, 0])[0]
    h2_no_burn = res_no_burn['SUMMARY.RES'][-1].get('h2', [0, 0])[0]
    _check("Burn-in allows AM to inflate h2 further before recording starts",
           h2_burn >= h2_no_burn - 0.05,   # burn-in h2 ≥ no-burn-in h2 (approximate)
           f"h2_burn={h2_burn:.4f}, h2_no_burn={h2_no_burn:.4f}")

    # Last recorded main gen has gen_abs_idx = n_burn_in + num_generations - 1
    expected_last_abs = N_BURN + N_MAIN - 1
    _check("GEN_ABS in final summary reflects burn-in offset",
           res_burn['SUMMARY.RES'][-1].get('GEN_ABS', 0) == expected_last_abs,
           f"GEN_ABS = {res_burn['SUMMARY.RES'][-1].get('GEN_ABS')}, "
           f"expected {expected_last_abs}")


# ===========================================================================
# Test 6 — Reproducibility (same seed → identical results)
# ===========================================================================

def test_reproducibility():
    print("\n=== Test 6: Reproducibility ===")
    mats = _base_matrices()

    # Pre-compute cv_info once so both runs use identical CVs.
    # (Without this, prepare_CV_random_selection consumes the global RNG
    # before __init__ resets the seed, causing different CV effect sizes.)
    np.random.seed(0)
    cv_df = AssortativeMatingSimulation.prepare_CV_random_selection(
        n_CV=300, rg_effects=0.3,
        maf_min=0.01, maf_max=0.50,
        prop_h2_obs_1=0.8, prop_h2_obs_2=0.8,
    )

    kwargs = dict(
        cv_info=cv_df,
        num_generations=2, pop_size=1000,
        mating_type="phenotypic",
        mate_on_trait=1,
        am_list=[0.3, 0.3],
        seed=99,
        **mats,
    )
    res1 = AssortativeMatingSimulation(**kwargs).run_simulation()
    res2 = AssortativeMatingSimulation(**kwargs).run_simulation()

    phen1 = res1['PHEN']
    phen2 = res2['PHEN']

    _check("Same seed + same cv_info → identical final population size",
           len(phen1) == len(phen2),
           f"run1 n={len(phen1)}, run2 n={len(phen2)}")

    if len(phen1) == len(phen2):
        y1_a = phen1['Y1'].astype(float).values
        y1_b = phen2['Y1'].astype(float).values
        _check("Same seed + same cv_info → identical final Y1 values",
               np.allclose(y1_a, y1_b),
               f"max abs diff = {np.abs(y1_a - y1_b).max():.2e}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    tests = [
        test_baseline_am,
        test_mtdna,
        test_island_migration,
        test_island_plus_mtdna,
        test_burn_in_base,
        test_reproducibility,
    ]

    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"\n  [ERROR] {t.__name__} raised an unexpected exception:")
            traceback.print_exc()
            _fail_global = True

    print("\n" + "=" * 60)
    print(f"RESULTS: {_pass} passed, {_fail} failed")
    print("=" * 60)
    if _fail > 0:
        print("\nFailed checks:")
        for line in _results:
            if "[FAIL]" in line:
                print(line)
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
