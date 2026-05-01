"""
IslandSimulation.py
====================
Extends AssortativeMatingSimulation with phenotype-driven island
migration.  All genetic mechanics (genotype generation, reproduction,
PGS components, summary stats) are inherited unchanged.

New class
---------
IslandMigrationSimulation(AssortativeMatingSimulation)

What it adds
------------
1. An ``island_id`` column (1..n_islands) on every individual.
2. A vectorised migration step before each generation's mating:
   a fraction ``move_p`` of each sex moves to islands sorted by their
   value on ``migration_trait`` (Y1 or Y2), mirroring the R
   ``island x.R`` script.
3. Within-island assortative mating using a per-generation
   ``within_island_am`` parameter that is independent of the base
   class ``am_list`` (which continues to govern the genetic
   initialisation structure).
4. Optional burn-in (inherited ``n_burn_in``): the population
   evolves to equilibrium before any statistics are recorded.
5. Optional parallel per-island mating via joblib.

Parameters inherited from base class (all work as before)
----------------------------------------------------------
cv_info / n_CV / rg_effects / maf_min / maf_max / maf_dist,
num_generations, pop_size  (must equal n_per_island * n_islands),
mating_type, avoid_inbreeding, save_each_gen, save_covs, seed,
output_summary_filename, summary_file_scope, n_burn_in,
cove_mat, f_mat, s_mat, a_mat, d_mat, am_list,
covy_mat, k2_matrix.

New parameters
--------------
n_islands        : int   (default 10)   — number of islands
move_p           : float (default 0.10) — fraction of each sex that
                                          migrates per generation
within_island_am : float or list        — AM correlation used for
                                          within-island mating.
                                          A float is broadcast to a
                                          list of length
                                          n_burn_in + num_generations.
migration_trait  : int   (default 2)    — 1=Y1 or 2=Y2 used to sort
                                          migrants onto islands
mating_trait     : int   (default 1)    — 1 or 2; the trait used for
                                          within-island assortment.
                                          Sets mate_on_trait internally.
n_jobs           : int   (default 1)    — parallel jobs for per-island
                                          mating; -1 = all CPUs.
                                          Requires joblib.

Notes
-----
* ``mate_on_trait`` is set automatically from ``mating_trait``; do
  not pass it separately.
* ``pop_size`` must be divisible by ``n_islands * 2``.
* During burn-in the AM value used is
  ``within_island_am_list[burn_gen_index]``.
* Island IDs are assigned to offspring via their mother's island at
  the time of mating (before the *next* generation's migration).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from SimulationFunctions import AssortativeMatingSimulation


# ---------------------------------------------------------------------------
# Module-level helper — must be outside the class for joblib pickling
# ---------------------------------------------------------------------------

def _mate_one_island(
    island_phen_df: pd.DataFrame,
    mating_type: str,
    am_val: float,
    n_offspring_target: int,
    island_seed: int,
    # sim config that _assort_mate needs from self
    mate_on_trait: int,
    avoid_inbreeding: bool,
    # references to the two PD-fixing methods (passed as callables)
    _is_positive_definite,
    _make_positive_definite_nearest,
    _find_unique_closest_indices,
) -> dict:
    """Run assortative mating for a single island.

    This is a module-level function so that joblib can pickle it.
    It re-implements the core logic of _assort_mate for a single
    scalar AM value (mate_on_trait mode).
    """
    from scipy.spatial.distance import cdist

    np.random.seed(island_seed)

    ancestor_id_cols = [
        'Father.ID', 'Mother.ID',
        'Fathers.Father.ID', 'Fathers.Mother.ID',
        'Mothers.Father.ID', 'Mothers.Mother.ID',
    ]
    cols = ["ID", "Sex", "Y1", "Y2"] + ancestor_id_cols
    existing_cols = [c for c in cols if c in island_phen_df.columns]

    males_slim   = island_phen_df.loc[island_phen_df["Sex"] == 1, existing_cols].copy()
    females_slim = island_phen_df.loc[island_phen_df["Sex"] == 0, existing_cols].copy()

    males_slim.rename(columns={"Y1": "mating1", "Y2": "mating2"}, inplace=True)
    females_slim.rename(columns={"Y1": "mating1", "Y2": "mating2"}, inplace=True)

    nm, nf = len(males_slim), len(females_slim)
    if nm > nf:
        males_slim = males_slim.sample(n=nf, replace=False)
    elif nf > nm:
        females_slim = females_slim.sample(n=nm, replace=False)
    n_pairs = len(males_slim)

    if n_pairs == 0:
        return {
            'males.PHENDATA': pd.DataFrame(),
            'females.PHENDATA': pd.DataFrame(),
            'achieved_spousal_corr': np.eye(2),
        }

    # Build bivariate template for rank-matching (single-trait mode)
    target_mu = float(am_val)
    matcor = np.array([[1.0, target_mu], [target_mu, 1.0]])
    if not _is_positive_definite(matcor):
        matcor = _make_positive_definite_nearest(matcor)

    x_sim = np.random.multivariate_normal(np.zeros(2), matcor, size=n_pairs)
    x_sim_m = x_sim[:, 0:1]
    x_sim_f = x_sim[:, 1:2]

    trait_col = f"mating{mate_on_trait}"

    def get_scaled(df):
        vals = df[[trait_col]].values
        return (vals - vals.mean(0)) / (vals.std(0, ddof=1) + 1e-9)

    m_idx = _find_unique_closest_indices(cdist(get_scaled(males_slim),   x_sim_m))
    f_idx = _find_unique_closest_indices(cdist(get_scaled(females_slim), x_sim_f))

    valid = (m_idx != -1) & (f_idx != -1)
    paired_m = males_slim.iloc[m_idx[valid]].reset_index(drop=True)
    paired_f = females_slim.iloc[f_idx[valid]].reset_index(drop=True)

    m_full = island_phen_df.loc[island_phen_df['ID'].isin(paired_m['ID'])]\
        .set_index('ID').loc[paired_m['ID']].reset_index()
    f_full = island_phen_df.loc[island_phen_df['ID'].isin(paired_f['ID'])]\
        .set_index('ID').loc[paired_f['ID']].reset_index()

    n_act = len(m_full)
    if n_act > 0:
        off_counts = np.random.poisson(n_offspring_target / n_act, n_act)
        diff = n_offspring_target - off_counts.sum()
        if diff > 0:
            off_counts[np.random.choice(n_act, diff, replace=True)] += 1
        elif diff < 0:
            eligible = np.where(off_counts > 0)[0]
            for _ in range(abs(diff)):
                if len(eligible) == 0:
                    break
                i = np.random.choice(eligible)
                off_counts[i] = max(0, off_counts[i] - 1)
        m_full = m_full.copy(); f_full = f_full.copy()
        m_full['num.offspring'] = off_counts
        f_full['num.offspring'] = off_counts
        m_full['Spouse.ID'] = f_full['ID'].values
        f_full['Spouse.ID'] = m_full['ID'].values

    return {
        'males.PHENDATA': m_full,
        'females.PHENDATA': f_full,
        'achieved_spousal_corr': np.eye(2),
    }


# ---------------------------------------------------------------------------
# Main subclass
# ---------------------------------------------------------------------------

class IslandMigrationSimulation(AssortativeMatingSimulation):
    """Assortative mating simulation with island population structure
    and phenotype-driven migration.

    See module docstring for full parameter documentation.
    """

    def __init__(
        self,
        # --- island-specific parameters ---
        n_islands: int = 10,
        move_p: float = 0.10,
        within_island_am=0.4,
        migration_trait: int = 2,
        mating_trait: int = 1,
        n_jobs: int = 1,
        # --- all base-class parameters (passed through) ---
        **base_kwargs,
    ):
        # ---- validation ----
        pop_size = base_kwargs.get('pop_size')
        if pop_size is None:
            raise ValueError("pop_size must be provided.")
        _ps = int(pop_size) if isinstance(pop_size, (int, float)) else int(pop_size[0])
        if _ps % (n_islands * 2) != 0:
            raise ValueError(
                f"pop_size ({_ps}) must be divisible by n_islands*2 "
                f"({n_islands}*2={n_islands*2})."
            )

        # ---- island config ----
        self.n_islands = int(n_islands)
        self.n_per_island = _ps // self.n_islands
        self.move_p = float(move_p)
        self.migration_trait = int(migration_trait)
        self.mating_trait = int(mating_trait)
        self.n_jobs = int(n_jobs)

        # ---- build within_island_am_list ----
        n_burn_in = int(base_kwargs.get('n_burn_in', 0))
        num_generations = int(base_kwargs.get('num_generations', 0))
        total_gens = n_burn_in + num_generations
        if isinstance(within_island_am, (int, float)):
            self.within_island_am_list = [float(within_island_am)] * total_gens
        else:
            wia = list(within_island_am)
            if len(wia) < total_gens:
                # pad with last value
                wia = wia + [wia[-1]] * (total_gens - len(wia))
            self.within_island_am_list = [float(v) for v in wia]

        # force mate_on_trait to mating_trait — we handle AM ourselves
        base_kwargs['mate_on_trait'] = mating_trait

        # track current absolute generation for _assort_mate override
        self._current_gen_abs = 0

        # ---- call base class __init__ ----
        super().__init__(**base_kwargs)

    # ------------------------------------------------------------------
    # Hook 1: extend phen_column_names and assign initial island IDs
    # ------------------------------------------------------------------

    def _extend_phen_column_names(self):
        """Append 'island_id' to column list and assign initial island IDs."""
        if 'island_id' not in self.phen_column_names:
            self.phen_column_names.append('island_id')

        n_pop = len(self.phen_df)
        # Balanced assignment: each island gets exactly n_per_island individuals
        ids = np.repeat(np.arange(1, self.n_islands + 1), self.n_per_island)
        if len(ids) < n_pop:
            ids = np.concatenate([ids, np.full(n_pop - len(ids), self.n_islands)])
        self.phen_df['island_id'] = ids[:n_pop]

    # ------------------------------------------------------------------
    # Hook 2: run migration before mating, record current gen
    # ------------------------------------------------------------------

    def _pre_mating_hook(self, gen_abs_idx: int):
        self._current_gen_abs = gen_abs_idx
        self._migrate()

    # ------------------------------------------------------------------
    # Hook 3: propagate island_id from mother to offspring
    # ------------------------------------------------------------------

    def _post_offspring_hook(self, offspring_data: dict, gen_abs_idx: int) -> dict:
        if 'island_id' not in self.phen_df.columns:
            return offspring_data
        # Snapshot mother -> island_id AFTER migration (which already happened
        # in _pre_mating_hook), so offspring inherit their mother's post-
        # migration island.
        mother_island = (
            self.phen_df[['ID', 'island_id']]
            .dropna(subset=['island_id'])
            .set_index('ID')['island_id']
        )
        phen = offspring_data['PHEN']
        if 'Mother.ID' in phen.columns:
            phen['island_id'] = phen['Mother.ID'].map(mother_island)
        offspring_data['PHEN'] = phen
        return offspring_data

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate(self):
        """Vectorised phenotype-driven migration.

        For each sex separately:
        1. From each island, randomly select ``move_p`` fraction as movers.
        2. Sort all movers (across all islands) by ``Y{migration_trait}``.
        3. Redistribute sorted movers evenly across islands 1..n_islands,
           so the highest-scoring movers end up on the highest-numbered
           island — matching the R script's logic.
        """
        df = self.phen_df
        migration_col = f"Y{self.migration_trait}"
        n_sex_per_island = self.n_per_island // 2  # half female, half male per island
        n_movers_per_island = max(1, int(round(n_sex_per_island * self.move_p)))

        for sex_val in (0, 1):
            sex_mask = (df['Sex'] == sex_val).values
            sex_indices = np.where(sex_mask)[0]

            all_mover_indices = []
            for isl in range(1, self.n_islands + 1):
                isl_mask = (df['island_id'] == isl).values & sex_mask
                isl_indices = np.where(isl_mask)[0]
                if len(isl_indices) == 0:
                    continue
                n_move = min(n_movers_per_island, len(isl_indices))
                chosen = np.random.choice(isl_indices, size=n_move, replace=False)
                all_mover_indices.extend(chosen.tolist())

            if not all_mover_indices:
                continue

            mover_idx = np.array(all_mover_indices)
            # Sort movers by migration trait (ascending → island 1 = lowest)
            trait_vals = pd.to_numeric(df[migration_col].iloc[mover_idx], errors='coerce').values
            sort_order = np.argsort(trait_vals, kind='stable')
            sorted_mover_idx = mover_idx[sort_order]

            # Divide into n_islands equal chunks; assign island IDs 1..n_islands
            n_total_movers = len(sorted_mover_idx)
            chunk_size = n_total_movers / self.n_islands
            new_island_ids = np.empty(n_total_movers, dtype=int)
            for isl in range(self.n_islands):
                start = int(round(isl * chunk_size))
                end   = int(round((isl + 1) * chunk_size))
                new_island_ids[start:end] = isl + 1

            self.phen_df.iloc[sorted_mover_idx, self.phen_df.columns.get_loc('island_id')] = new_island_ids

    # ------------------------------------------------------------------
    # Within-island mating override
    # ------------------------------------------------------------------

    def _assort_mate(
        self,
        phendata_df_current_gen,
        mating_type,
        pheno_mate_corr_target_xsim_mu,  # unused — we use within_island_am_list
        pop_size_target_offspring,
    ) -> dict:
        """Override: run assortative mating independently per island,
        then concatenate results into a single mates dict.
        """
        am_val = self.within_island_am_list[self._current_gen_abs]
        n_offspring_per_island = pop_size_target_offspring // self.n_islands

        # Group phen_df by island
        islands = {}
        for isl in range(1, self.n_islands + 1):
            mask = self.phen_df['island_id'] == isl
            islands[isl] = self.phen_df.loc[mask].copy()

        # Callables we need to pass to the module-level helper
        _ispd  = self._is_positive_definite
        _mkpd  = self._make_positive_definite_nearest
        _funci = self._find_unique_closest_indices

        def _do_island(isl):
            island_seed = (
                (self.seed if self.seed != 0 else 42)
                + isl * 1000
                + self._current_gen_abs * 37
            )
            return _mate_one_island(
                island_phen_df=islands[isl],
                mating_type=mating_type,
                am_val=am_val,
                n_offspring_target=n_offspring_per_island,
                island_seed=island_seed,
                mate_on_trait=self.mating_trait,
                avoid_inbreeding=self.avoid_inbreeding,
                _is_positive_definite=_ispd,
                _make_positive_definite_nearest=_mkpd,
                _find_unique_closest_indices=_funci,
            )

        if self.n_jobs == 1 or self.n_islands == 1:
            results = [_do_island(isl) for isl in range(1, self.n_islands + 1)]
        else:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_mate_one_island)(
                        island_phen_df=islands[isl],
                        mating_type=mating_type,
                        am_val=am_val,
                        n_offspring_target=n_offspring_per_island,
                        island_seed=(
                            (self.seed if self.seed != 0 else 42)
                            + isl * 1000
                            + self._current_gen_abs * 37
                        ),
                        mate_on_trait=self.mating_trait,
                        avoid_inbreeding=self.avoid_inbreeding,
                        _is_positive_definite=_ispd,
                        _make_positive_definite_nearest=_mkpd,
                        _find_unique_closest_indices=_funci,
                    )
                    for isl in range(1, self.n_islands + 1)
                )
            except ImportError:
                print("Warning: joblib not available, falling back to serial execution.")
                results = [_do_island(isl) for isl in range(1, self.n_islands + 1)]

        # Concatenate across islands
        all_males   = pd.concat([r['males.PHENDATA']   for r in results if not r['males.PHENDATA'].empty],   ignore_index=True)
        all_females = pd.concat([r['females.PHENDATA'] for r in results if not r['females.PHENDATA'].empty], ignore_index=True)

        return {
            'males.PHENDATA':   all_males,
            'females.PHENDATA': all_females,
            'achieved_spousal_corr': np.eye(2),
        }

    # ------------------------------------------------------------------
    # Convenience: island-level summary statistics
    # ------------------------------------------------------------------

    def island_summary(self) -> pd.DataFrame:
        """Return a DataFrame with per-island means and SDs of Y1, Y2,
        genotypic values (AO1, AO2), and population counts.

        Useful for checking stratification after burn-in.
        """
        if self.phen_df is None or 'island_id' not in self.phen_df.columns:
            return pd.DataFrame()

        cols = [c for c in ['Y1', 'Y2', 'AO1', 'AO2'] if c in self.phen_df.columns]
        agg = {c: ['mean', 'std', 'count'] for c in cols}
        summary = self.phen_df.groupby('island_id').agg(agg)
        summary.columns = ['_'.join(c) for c in summary.columns]
        return summary.reset_index()
