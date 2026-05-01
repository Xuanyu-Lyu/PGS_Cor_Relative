# SimulationFunctions — Reference Documentation

Two files, one class hierarchy.

```
SimulationFunctions.py          ← base class + utility functions
IslandSimulation.py             ← island-migration subclass
```

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Phenotype Equation & Variance Components](#2-phenotype-equation--variance-components)
3. [Output Columns (`phen_df`)](#3-output-columns-phen_df)
4. [Utility Functions](#4-utility-functions)
5. [AssortativeMatingSimulation](#5-assortativematingsimulation)
   - [Constructor Parameters](#constructor-parameters)
   - [Key Model Matrices](#key-model-matrices)
   - [Public Methods](#public-methods)
   - [Extension Hooks](#extension-hooks)
   - [Return Value of `run_simulation()`](#return-value-of-run_simulation)
6. [IslandMigrationSimulation](#6-islandmigrationsimulation)
   - [Additional Parameters](#additional-parameters)
   - [Public Methods](#public-methods-1)
   - [Migration Logic](#migration-logic)
7. [Scenario Cookbook](#7-scenario-cookbook)
   - [Scenario A — Minimal single-trait AM, no islands](#scenario-a--minimal-single-trait-am-no-islands)
   - [Scenario B — Two-trait AM with correlated traits](#scenario-b--two-trait-am-with-correlated-traits)
   - [Scenario C — Island stratification with burn-in](#scenario-c--island-stratification-with-burn-in)
   - [Scenario D — Island simulation with changing AM over generations](#scenario-d--island-simulation-with-changing-am-over-generations)
   - [Scenario E — Parallel per-island mating](#scenario-e--parallel-per-island-mating)
   - [Scenario F — Pre-computed CV info (reproducible runs)](#scenario-f--pre-computed-cv-info-reproducible-runs)
   - [Scenario G — Burn-in without islands (base class)](#scenario-g--burn-in-without-islands-base-class)
   - [Scenario H — Accessing results and post-processing](#scenario-h--accessing-results-and-post-processing)
   - [Scenario I — mtDNA maternal effects](#scenario-i--mtdna-maternal-effects)
8. [Matrix Construction Guide](#8-matrix-construction-guide)
9. [FAQ / Gotchas](#9-faq--gotchas)

---

## 1. Model Overview

The simulation tracks two quantitative traits (Trait 1 = study trait, Trait 2 = ancillary / migration-driving trait) across multiple overlapping components:

- **Observed genetic value** (`AO`): the part of the genome captured by measured SNPs
- **Latent genetic value** (`AL`): the remainder of the genome, unobserved
- **Vertical transmission** (`F`): parental phenotype → offspring phenotype path
- **Non-shared environment** (`E`): residual, uncorrelated across family members

The split between observed and latent is governed by masks applied to the causal variant (CV) effect sizes, set implicitly via `d_mat` and `a_mat`.

---

## 2. Phenotype Equation & Variance Components

$$Y = A_O + A_L + F + MT + E$$

where $MT$ is the maternally inherited mitochondrial DNA (mtDNA) component. It is zero by default (`mt_mat = zeros`). When non-zero, each offspring's $MT$ value is computed from the mother's phenotype via the `mt_mat` path matrix.

At generation 0, variances are set by the matrices supplied to the constructor. After each generation the variance components evolve due to assortment and vertical transmission. `summary_results` tracks empirical covariance matrices each generation.

**Heritability decomposition**

| Symbol | Meaning |
|--------|----------|
| `h2`   | $(V_{AO} + V_{AL}) / V_Y$ — total narrow-sense |
| `h2.obs` | $V_{AO} / V_Y$ — from observed SNPs |
| `h2.lat` | $V_{AL} / V_Y$ — from latent (unobserved) variants |
| `h2.mt` | $V_{MT} / V_Y$ — from maternally transmitted mtDNA |

---

## 3. Output Columns (`phen_df`)

Every individual row in `phen_df` (and in `history['PHEN'][gen]`) contains:

| Column | Description |
|--------|-------------|
| `ID` | Unique integer ID |
| `Father.ID`, `Mother.ID` | Parent IDs (NaN in founder gen) |
| `Fathers.Father.ID`, `Fathers.Mother.ID`, `Mothers.Father.ID`, `Mothers.Mother.ID` | Grandparent IDs |
| `Sex` | 0 = female, 1 = male |
| `AO1`, `AO2` | Observed genetic value, traits 1 & 2 |
| `AL1`, `AL2` | Latent genetic value, traits 1 & 2 |
| `F1`, `F2` | Vertical transmission component |
| `MT1`, `MT2` | mtDNA maternal transmission component (zero when `mt_mat` is not set) |
| `E1`, `E2` | Non-shared environment |
| `Y1`, `Y2` | Total phenotype |
| `Y1P/Y2P`, `Y1M/Y2M` | Paternal / maternal phenotypes (copied from parents during reproduction) |
| `F1P/F2P`, `F1M/F2M` | Paternal / maternal F values |
| `TPO1/2`, `TMO1/2` | Transmitted paternal / maternal **observed** haplotype score |
| `NTPO1/2`, `NTMO1/2` | Non-transmitted paternal / maternal **observed** haplotype score |
| `TPL1/2`, `TML1/2` | Transmitted paternal / maternal **latent** haplotype score |
| `NTPL1/2`, `NTML1/2` | Non-transmitted paternal / maternal **latent** haplotype score |
| `island_id` | *(IslandMigrationSimulation only)* Island membership (1..n_islands) |

**PGS computation from columns**

```python
df['PGS1'] = df['TPO1'] + df['TMO1']   # observed PGS, trait 1
df['PGS2'] = df['TPO2'] + df['TMO2']   # observed PGS, trait 2
```

---

## 4. Utility Functions

These are module-level helpers in `SimulationFunctions.py`.

### `is_even(x)` / `is_odd(x)`
```python
is_even(4)   # True
is_odd(3)    # True
```

---

### `next_smallest(x_list, value_thresh, return_index=True)`
Find the largest element in `x_list` that is strictly less than `value_thresh`.

```python
next_smallest([1, 3, 5, 7], 6)        # returns 2 (0-based index of 5)
next_smallest([1, 3, 5, 7], 6, False) # returns 5 (the value)
```

Returns `None` when no element satisfies the condition.

---

### `cor2cov(X, var1, var2)`
Convert a 2×2 correlation matrix to a covariance matrix.

```python
import numpy as np
R = np.array([[1.0, 0.3], [0.3, 1.0]])
C = cor2cov(R, var1=0.5, var2=0.8)
# C = [[0.5, 0.19], [0.19, 0.8]]
```

Useful for building `cove_mat` from known variances and a residual correlation.

---

### `AssortativeMatingSimulation.prepare_CV_random_selection(...)`
Static method. Generates a causal variant (CV) info DataFrame from scratch. Usually called automatically by the constructor when `n_CV` / `rg_effects` are supplied.

```python
cv_df = AssortativeMatingSimulation.prepare_CV_random_selection(
    n_CV=2000,
    rg_effects=0.5,      # genetic correlation between traits
    maf_min=0.01,
    maf_max=0.50,
    prop_h2_obs_1=0.8,   # fraction of h2 captured by observed SNPs, trait 1
    prop_h2_obs_2=0.6,
    maf_dist="uniform",  # or "normal"
)
# Returns DataFrame with columns: maf, alpha1, alpha2, mask_obs1, mask_obs2
```

Pass the result as `cv_info=cv_df` to fix CVs across replicate runs.

---

## 5. AssortativeMatingSimulation

```python
from SimulationFunctions import AssortativeMatingSimulation
```

### Constructor Parameters

#### CV / Genotype Setup

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cv_info` | DataFrame | if not using `n_CV` | Pre-built CV table from `prepare_CV_random_selection` |
| `n_CV` | int | if not using `cv_info` | Number of causal variants |
| `rg_effects` | float | if not using `cv_info` | Genetic correlation of CV effects between traits |
| `maf_min` | float | if not using `cv_info` | Minimum minor allele frequency |
| `maf_max` | float | if not using `cv_info` | Maximum minor allele frequency |
| `maf_dist` | str | no | `"uniform"` (default) or `"normal"` |

#### Simulation Structure

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | int | required | Number of **recorded** generations to run |
| `pop_size` | int or list[int] | required | Population size each generation. Scalar = constant; list must have length `num_generations` |
| `n_burn_in` | int | `0` | Unrecorded warm-up generations run before the main loop. Population evolves but no history/stats are saved. `am_list[0]` is used throughout burn-in |
| `seed` | int | `0` | Global NumPy random seed. `0` = unseeded |

#### Mating

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mating_type` | str | `"phenotypic"` | `"phenotypic"`, `"social"`, or `"genotypic"` |
| `avoid_inbreeding` | bool | `True` | Exclude first-degree relatives from pairing |
| `am_list` | list | required | Per-generation assortative mating target. If `mate_on_trait` is set: list of floats (scalar correlation). If not set: list of 2×2 numpy arrays (cross-trait mating correlation matrix). Length must equal `num_generations` |
| `mate_on_trait` | int or None | `None` | `1` or `2` = single-trait AM mode (scalar `am_list`). `None` = two-trait AM mode (matrix `am_list`) |

#### mtDNA Effects

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mt_mat` | 2×2 array or None | `None` | Diagonal matrix of mtDNA transmission paths. `mt_mat[0,0]` = path from mother Y1 → offspring MT1; `mt_mat[1,1]` = path from mother Y2 → offspring MT2. Off-diagonal elements must be zero (raises `ValueError` if violated). Inheritance is strictly maternal — fathers never transmit mtDNA. Default `None` sets all paths to zero |

---

#### Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_each_gen` | bool | `True` | Store full `phen_df`, `xo`, `xl`, and mates dict in `history` for every generation |
| `save_covs` | bool | `True` | Store per-generation covariance matrices in `covariances_log` |
| `output_summary_filename` | str or None | `None` | If given, write a text summary file after each generation |
| `summary_file_scope` | str | `"final"` | `"final"` = only write last generation; `"all"` = write all |

---

### Key Model Matrices

All matrices are 2×2 (one row/column per trait) unless noted.

| Matrix | Parameter | Description |
|--------|-----------|-------------|
| `cove_mat` | `cove_mat` | Covariance matrix of the environment term **E** |
| `f_mat` | `f_mat` | Vertical transmission matrix. `f_mat[i,j]` = path from parent trait j to offspring F component i |
| `s_mat` | `s_mat` | Social homogamy / assortment on shared environment (set to zeros if unused) |
| `a_mat` | `a_mat` | Latent genetic path matrix. Used to derive the observed/latent SNP split |
| `d_mat` | `d_mat` | Observed genetic path matrix (same interpretation as `a_mat` for observed SNPs) |
| `covy_mat` | `covy_mat` | Target phenotypic covariance matrix (used to scale/validate the setup) |
| `k2_matrix` | `k2_matrix` | K2 matrix: genetic nurture / indirect genetic effects |
| `mt_mat` | `mt_mat` | Diagonal 2×2 mtDNA path matrix. `mt_mat[i,i]` = path from mother's $Y_i$ to offspring's $MT_i$. **Must be diagonal** (no cross-trait effects). Default: `None` (treated as zero matrix) |

See [Section 8](#8-matrix-construction-guide) for how to build these from heritabilities.

---

### Public Methods

#### `run_simulation() → dict`

Run burn-in (if `n_burn_in > 0`) then the main loop. Returns a results dict (see [Return Value](#return-value-of-run_simulation)).

```python
results = sim.run_simulation()
```

---

### Extension Hooks

Override these in subclasses to extend behaviour without duplicating the loop.

| Method | When called | Base behaviour |
|--------|-------------|----------------|
| `_extend_phen_column_names()` | Once, at end of gen-0 init | no-op |
| `_pre_mating_hook(gen_abs_idx)` | Before `_assort_mate` every generation | no-op |
| `_post_offspring_hook(offspring_data, gen_abs_idx) → dict` | After `_reproduce` every generation | returns `offspring_data` unchanged |

`gen_abs_idx` is 0-based and counts across burn-in + main phase: e.g. with `n_burn_in=5`, main generation 1 has `gen_abs_idx=5`.

---

### Return Value of `run_simulation()`

```python
{
    'SUMMARY.RES':  list of dicts,   # one entry per recorded generation
    'PHEN':         pd.DataFrame,    # final generation individual data
    'XO':           np.ndarray,      # final genotype matrix (N × n_CV)
    'XL':           np.ndarray,      # placeholder (zeros)
    'HISTORY':      dict or None,    # if save_each_gen=True
    'COVARIANCES':  list or None,    # if save_covs=True
}
```

**`SUMMARY.RES[i]` keys**

| Key | Description |
|-----|-------------|
| `GEN` | 1-based main-phase generation number |
| `GEN_ABS` | 0-based absolute generation (includes burn-in offset) |
| `POPSIZE` | Number of individuals |
| `MATE.COR` | AM correlation used for the *next* generation (look-ahead) |
| `VAO`, `VAL`, `VF`, `VE`, `VP` | 2×2 covariance matrices for each component |
| `VMT` | 2×2 covariance matrix for the mtDNA component |
| `h2`, `h2.obs`, `h2.lat` | Two-element lists `[trait1, trait2]` |
| `h2.mt` | Two-element list `[trait1, trait2]` — $V_{MT} / V_P$ per trait |

**`HISTORY` keys** (only when `save_each_gen=True`)

```python
results['HISTORY']['PHEN'][gen]    # phen_df snapshot, gen 0 = founders
results['HISTORY']['XO'][gen]      # genotype matrix snapshot
results['HISTORY']['MATES'][gen]   # mates dict (None for gen 0)
```

---

## 6. IslandMigrationSimulation

```python
from IslandSimulation import IslandMigrationSimulation
```

Inherits everything from `AssortativeMatingSimulation`. All base-class parameters are passed through `**kwargs`. The only overridden behaviours are:

1. Population initialisation adds balanced `island_id` assignment
2. `_pre_mating_hook` runs vectorised migration before each mating call
3. `_assort_mate` is replaced with per-island independent AM
4. `_post_offspring_hook` propagates mother's post-migration island to offspring

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_islands` | int | 10 | Number of islands |
| `move_p` | float | 0.10 | Fraction of each sex on each island that migrates each generation |
| `within_island_am` | float or list | 0.4 | AM correlation used for within-island mating. Float = constant across all generations (burn-in + main). List = per-generation schedule; padded with last value if too short |
| `migration_trait` | int | 2 | `1` or `2` — which trait's phenotype drives migration sorting |
| `mating_trait` | int | 1 | `1` or `2` — which trait is used for within-island assortment |
| `n_jobs` | int | 1 | joblib parallel jobs for per-island mating. `-1` = all CPUs. Requires `pip install joblib` |

**Constraint**: `pop_size` must be divisible by `n_islands * 2`.

**Note**: `mate_on_trait` is set internally from `mating_trait`. Do not pass `mate_on_trait` separately when using `IslandMigrationSimulation`.

---

### Public Methods

All base-class public methods, plus:

#### `island_summary() → pd.DataFrame`

Returns per-island means, SDs, and counts for Y1, Y2, AO1, AO2. Useful for checking stratification after burn-in.

```python
sim.island_summary()
#    island_id  Y1_mean  Y1_std  Y1_count  Y2_mean  Y2_std  ...
```

---

### Migration Logic

Each generation, before mating:

1. For each sex (female, then male) independently:
   - From each island, `move_p × (n_per_island / 2)` individuals are randomly selected as movers
   - All movers across all islands are pooled and sorted by `Y{migration_trait}` (ascending)
   - The sorted list is divided into `n_islands` equal chunks; chunk 1 → island 1, chunk `n_islands` → island `n_islands`

This mirrors the R `island x.R` script: individuals with high Y2 (or Y1) accumulate on high-numbered islands over time.

Offspring inherit their **mother's island at the time of mating** (i.e. after that generation's migration, before the next). The next generation's migration reassigns them independently.

---

## 7. Scenario Cookbook

### Scenario A — Minimal single-trait AM, no islands

Single-trait mating on Trait 2, constant AM of 0.4, 10 generations.

```python
import numpy as np
from SimulationFunctions import AssortativeMatingSimulation

N = 10000
N_GEN = 10
h2 = 0.5
vg = h2
ve = 1 - h2

# --- Model matrices ---
cove_mat   = np.diag([ve, ve])
f_mat      = np.zeros((2, 2))          # no vertical transmission
s_mat      = np.zeros((2, 2))
a_mat      = np.zeros((2, 2))          # no latent genetic component
d_mat      = np.array([[np.sqrt(vg), 0], [0, np.sqrt(vg)]])
covy_mat   = np.eye(2)
k2_matrix  = np.zeros((2, 2))

am_val = 0.4
am_list = [am_val] * N_GEN

sim = AssortativeMatingSimulation(
    n_CV=2000, rg_effects=0.5,
    maf_min=0.01, maf_max=0.50,
    num_generations=N_GEN,
    pop_size=N,
    mating_type="phenotypic",
    mate_on_trait=2,       # single-trait mode
    am_list=am_list,
    cove_mat=cove_mat, f_mat=f_mat, s_mat=s_mat,
    a_mat=a_mat, d_mat=d_mat,
    covy_mat=covy_mat, k2_matrix=k2_matrix,
    seed=42,
)
results = sim.run_simulation()

final_phen = results['PHEN']
print(final_phen[['Y1', 'Y2']].describe())
```

---

### Scenario B — Two-trait AM with correlated traits

Two-trait AM uses a 2×2 mate correlation matrix per generation.

```python
import numpy as np
from SimulationFunctions import AssortativeMatingSimulation

N = 10000; N_GEN = 10

# mate_on_trait = None → two-trait AM mode
# am_list entries must be 2×2 arrays
mu = np.array([[0.4, 0.1],
               [0.1, 0.3]])   # cross-spouse correlation matrix
am_list = [mu] * N_GEN

# (same matrices as Scenario A)
cove_mat = np.diag([0.5, 0.5])
f_mat = s_mat = a_mat = np.zeros((2, 2))
d_mat = np.diag([np.sqrt(0.5)] * 2)
covy_mat = np.eye(2); k2_matrix = np.zeros((2, 2))

sim = AssortativeMatingSimulation(
    n_CV=2000, rg_effects=0.4,
    maf_min=0.01, maf_max=0.50,
    num_generations=N_GEN, pop_size=N,
    mating_type="phenotypic",
    mate_on_trait=None,   # two-trait mode (default)
    am_list=am_list,
    cove_mat=cove_mat, f_mat=f_mat, s_mat=s_mat,
    a_mat=a_mat, d_mat=d_mat,
    covy_mat=covy_mat, k2_matrix=k2_matrix,
    seed=1,
)
results = sim.run_simulation()
```

---

### Scenario C — Island stratification with burn-in

10 islands, 10% migration, 30 generation burn-in, 10 recorded.
Mating on Trait 1 within islands; migration driven by Trait 2.

```python
import numpy as np
from IslandSimulation import IslandMigrationSimulation

N_ISLANDS = 10
N_PER_ISLAND = 2000              # must be even
N = N_PER_ISLAND * N_ISLANDS    # 20 000

N_BURN  = 30
N_MAIN  = 10

cove_mat = np.diag([0.55, 0.60])
f_mat    = np.zeros((2, 2))
s_mat    = np.zeros((2, 2))
a_mat    = np.zeros((2, 2))
d_mat    = np.diag([np.sqrt(0.45), np.sqrt(0.40)])
covy_mat = np.eye(2)
k2_matrix = np.zeros((2, 2))

# am_list governs the base-class genetic init (gen 0 structure)
# within_island_am is the within-island mating correlation
am_list = [0.4] * N_MAIN

sim = IslandMigrationSimulation(
    # island parameters
    n_islands=N_ISLANDS,
    move_p=0.10,
    within_island_am=0.4,
    migration_trait=2,    # Y2 drives which island you move to
    mating_trait=1,       # mate on Y1 within islands
    n_jobs=1,
    # base class parameters
    n_CV=2000, rg_effects=0.5,
    maf_min=0.01, maf_max=0.50,
    num_generations=N_MAIN,
    n_burn_in=N_BURN,
    pop_size=N,
    mating_type="phenotypic",
    am_list=am_list,
    cove_mat=cove_mat, f_mat=f_mat, s_mat=s_mat,
    a_mat=a_mat, d_mat=d_mat,
    covy_mat=covy_mat, k2_matrix=k2_matrix,
    seed=2025,
)
results = sim.run_simulation()

# Check stratification
print(sim.island_summary()[['island_id', 'Y2_mean', 'Y1_mean']])
```

---

### Scenario D — Island simulation with changing AM over generations

Pass a list of per-generation values to `within_island_am`.
Generation numbering covers burn-in + main: index 0 = burn-in gen 0, index `n_burn_in` = main gen 0.

```python
import numpy as np
from IslandSimulation import IslandMigrationSimulation

N_BURN = 20; N_MAIN = 10

# AM ramps from 0.2 to 0.5 during burn-in, stays at 0.5 in main phase
burn_schedule = list(np.linspace(0.2, 0.5, N_BURN))
main_schedule = [0.5] * N_MAIN
within_island_am = burn_schedule + main_schedule   # length = N_BURN + N_MAIN

sim = IslandMigrationSimulation(
    n_islands=10, move_p=0.10,
    within_island_am=within_island_am,
    migration_trait=2, mating_trait=1,
    n_CV=2000, rg_effects=0.5,
    maf_min=0.01, maf_max=0.50,
    num_generations=N_MAIN, n_burn_in=N_BURN,
    pop_size=20000,
    am_list=[0.5] * N_MAIN,
    mating_type="phenotypic",
    cove_mat=np.diag([0.55, 0.60]),
    f_mat=np.zeros((2,2)), s_mat=np.zeros((2,2)),
    a_mat=np.zeros((2,2)), d_mat=np.diag([np.sqrt(0.45), np.sqrt(0.40)]),
    covy_mat=np.eye(2), k2_matrix=np.zeros((2,2)),
    seed=99,
)
results = sim.run_simulation()
```

---

### Scenario E — Parallel per-island mating

Set `n_jobs=-1` to use all CPUs. Requires `pip install joblib`. Each island gets a deterministic seed derived from `global_seed + island_id * 1000 + gen_abs_idx * 37`, so results are reproducible.

```python
sim = IslandMigrationSimulation(
    n_islands=10, move_p=0.10,
    within_island_am=0.4,
    migration_trait=2, mating_trait=1,
    n_jobs=-1,               # all CPUs
    n_CV=2000, rg_effects=0.5,
    maf_min=0.01, maf_max=0.50,
    num_generations=10, n_burn_in=20,
    pop_size=20000,
    am_list=[0.4] * 10,
    mating_type="phenotypic",
    cove_mat=np.diag([0.55, 0.60]),
    f_mat=np.zeros((2,2)), s_mat=np.zeros((2,2)),
    a_mat=np.zeros((2,2)), d_mat=np.diag([np.sqrt(0.45), np.sqrt(0.40)]),
    covy_mat=np.eye(2), k2_matrix=np.zeros((2,2)),
    seed=42,
)
results = sim.run_simulation()
```

---

### Scenario F — Pre-computed CV info (reproducible runs)

Generate CVs once and reuse across parameter sweeps.

```python
import numpy as np
from SimulationFunctions import AssortativeMatingSimulation

np.random.seed(0)
cv_df = AssortativeMatingSimulation.prepare_CV_random_selection(
    n_CV=5000, rg_effects=0.3,
    maf_min=0.01, maf_max=0.50,
    prop_h2_obs_1=0.8, prop_h2_obs_2=0.7,
)

for am_val in [0.2, 0.4, 0.6]:
    sim = AssortativeMatingSimulation(
        cv_info=cv_df,           # same CVs every run
        num_generations=10, pop_size=5000,
        mating_type="phenotypic",
        mate_on_trait=1,
        am_list=[am_val] * 10,
        cove_mat=np.diag([0.5, 0.5]),
        f_mat=np.zeros((2,2)), s_mat=np.zeros((2,2)),
        a_mat=np.zeros((2,2)), d_mat=np.diag([np.sqrt(0.5)] * 2),
        covy_mat=np.eye(2), k2_matrix=np.zeros((2,2)),
        seed=1,
    )
    r = sim.run_simulation()
    h2_final = r['SUMMARY.RES'][-1]['h2']
    print(f"am={am_val}  h2={h2_final}")
```

---

### Scenario G — Burn-in without islands (base class)

`n_burn_in` works identically in `AssortativeMatingSimulation`. Only `summary_results` and `history` are unaffected by burn-in; population state is fully evolved.

```python
sim = AssortativeMatingSimulation(
    n_CV=2000, rg_effects=0.5,
    maf_min=0.01, maf_max=0.50,
    num_generations=5,
    n_burn_in=20,          # 20 unrecorded warm-up generations
    pop_size=5000,
    mating_type="phenotypic",
    mate_on_trait=1,
    am_list=[0.4] * 5,
    cove_mat=np.diag([0.5, 0.5]),
    f_mat=np.zeros((2,2)), s_mat=np.zeros((2,2)),
    a_mat=np.zeros((2,2)), d_mat=np.diag([np.sqrt(0.5)] * 2),
    covy_mat=np.eye(2), k2_matrix=np.zeros((2,2)),
    seed=0,
)
results = sim.run_simulation()
# len(results['SUMMARY.RES']) == 6  (gen 0 + 5 main gens; burn-in not stored)
```

---

### Scenario H — Accessing results and post-processing

```python
results = sim.run_simulation()

# Final population frame
phen = results['PHEN']
phen['PGS1'] = phen['TPO1'] + phen['TMO1']

# Heritability trajectory across generations
import pandas as pd
h2_df = pd.DataFrame([
    {'gen': s['GEN'], 'h2_t1': s['h2'][0], 'h2_t2': s['h2'][1]}
    for s in results['SUMMARY.RES'] if 'h2' in s
])
print(h2_df)

# Variance components in the final generation
last = results['SUMMARY.RES'][-1]
print("VP:", last['VP'])
print("VAO:", last['VAO'])

# Generation-by-generation phenotype history
for gen_idx, df in enumerate(results['HISTORY']['PHEN']):
    print(f"Gen {gen_idx}: n={len(df)}, var_Y1={df['Y1'].astype(float).var():.3f}")

# Per-island stratification (IslandMigrationSimulation only)
print(sim.island_summary())

# Export final phen_df
phen.to_csv("final_generation.csv", index=False)
```

---

### Scenario I — mtDNA maternal effects

Add a mtDNA component where mothers transmit 30% of their Trait 1 phenotype and 20% of their Trait 2 phenotype to offspring. No cross-trait transmission, no paternal contribution.

```python
import numpy as np
from SimulationFunctions import AssortativeMatingSimulation

N = 10000
N_GEN = 10

# Standard matrices (h2 ≈ 0.45 / 0.40)
h2_1, h2_2 = 0.45, 0.40
ve_1, ve_2 = 1 - h2_1 - 0.09, 1 - h2_2 - 0.04  # leave room for MT variance

cove_mat  = np.diag([ve_1, ve_2])
f_mat     = np.zeros((2, 2))
s_mat     = np.zeros((2, 2))
a_mat     = np.zeros((2, 2))
d_mat     = np.diag([np.sqrt(h2_1), np.sqrt(h2_2)])
covy_mat  = np.eye(2)
k2_matrix = np.zeros((2, 2))

# mtDNA path matrix — diagonal only, no cross-trait effects
# mt_mat[0,0] = 0.30 means offspring MT1 = 0.30 * mother_Y1
# mt_mat[1,1] = 0.20 means offspring MT2 = 0.20 * mother_Y2
mt_mat = np.diag([0.30, 0.20])

am_list = [0.4] * N_GEN

sim = AssortativeMatingSimulation(
    n_CV=2000, rg_effects=0.5,
    maf_min=0.01, maf_max=0.50,
    num_generations=N_GEN,
    pop_size=N,
    mating_type="phenotypic",
    mate_on_trait=1,
    am_list=am_list,
    cove_mat=cove_mat, f_mat=f_mat, s_mat=s_mat,
    a_mat=a_mat, d_mat=d_mat,
    covy_mat=covy_mat, k2_matrix=k2_matrix,
    mt_mat=mt_mat,   # <-- mtDNA component
    seed=42,
)
results = sim.run_simulation()

# Inspect the mtDNA variance component and its contribution to heritability
last = results['SUMMARY.RES'][-1]
print("VMT:", last['VMT'])
print("h2.mt:", last['h2.mt'])   # [trait1, trait2]

# Verify maternal transmission: MT values should correlate with mother's Y
phen = results['PHEN']
mother_y = phen.set_index('ID')[['Y1', 'Y2']].rename(columns={'Y1': 'mY1', 'Y2': 'mY2'})
with_mom = phen[phen['Mother.ID'].notna()].copy()
with_mom['mY1'] = with_mom['Mother.ID'].map(mother_y['mY1'])
with_mom['mY2'] = with_mom['Mother.ID'].map(mother_y['mY2'])
print("MT1 ~ mother Y1 cor:", np.corrcoef(with_mom['MT1'].astype(float),
                                           with_mom['mY1'].astype(float))[0, 1])
print("MT2 ~ mother Y2 cor:", np.corrcoef(with_mom['MT2'].astype(float),
                                           with_mom['mY2'].astype(float))[0, 1])
```

**Combining mtDNA with island stratification** — pass `mt_mat` to `IslandMigrationSimulation` in exactly the same way:

```python
from IslandSimulation import IslandMigrationSimulation

sim = IslandMigrationSimulation(
    n_islands=10, move_p=0.10,
    within_island_am=0.4,
    migration_trait=2, mating_trait=1,
    n_CV=2000, rg_effects=0.5,
    maf_min=0.01, maf_max=0.50,
    num_generations=10, n_burn_in=20,
    pop_size=20000,
    am_list=[0.4] * 10,
    mating_type="phenotypic",
    cove_mat=np.diag([0.46, 0.56]),
    f_mat=np.zeros((2,2)), s_mat=np.zeros((2,2)),
    a_mat=np.zeros((2,2)), d_mat=np.diag([np.sqrt(0.45), np.sqrt(0.40)]),
    covy_mat=np.eye(2), k2_matrix=np.zeros((2,2)),
    mt_mat=np.diag([0.30, 0.20]),
    seed=2025,
)
results = sim.run_simulation()
```

---

## 8. Matrix Construction Guide

Given heritabilities `h2_1`, `h2_2` for Traits 1 and 2, a genetic correlation `rg`, and an environmental correlation `re`:

```python
import numpy as np

h2_1 = 0.45; h2_2 = 0.40
rg   = 0.30; re   = 0.0
ve_1 = 1 - h2_1; ve_2 = 1 - h2_2

# No vertical transmission, no latent genetic component
f_mat   = np.zeros((2, 2))
s_mat   = np.zeros((2, 2))
a_mat   = np.zeros((2, 2))   # latent path = 0
k2_matrix = np.zeros((2, 2))

# d_mat: observed genetic paths (diagonal = sqrt of heritability)
d_mat = np.array([[np.sqrt(h2_1), 0],
                  [0, np.sqrt(h2_2)]])

# cove_mat: environment covariance
cove_off = re * np.sqrt(ve_1 * ve_2)
cove_mat = np.array([[ve_1, cove_off],
                     [cove_off, ve_2]])

# covy_mat: target phenotypic covariance (unit variance + genetic covariance)
covy_off = rg * np.sqrt(h2_1 * h2_2)
covy_mat = np.array([[1.0,      covy_off],
                     [covy_off, 1.0     ]])

# rg_effects for CV generation = rg
```

**Adding vertical transmission**

```python
f11 = 0.10   # path: parent Y1 → offspring F1
f22 = 0.08

f_mat = np.array([[f11, 0  ],
                  [0,   f22]])
# Adjust cove_mat accordingly so that total Var(Y) = 1 at equilibrium
# (requires solving the Rao-Huang equations, or iterate the simulation)
```

**Splitting heritability into observed and latent**

```python
prop_obs_1 = 0.80   # 80% of h2 captured by SNPs
prop_lat_1 = 1 - prop_obs_1

d_mat = np.array([[np.sqrt(h2_1 * prop_obs_1), 0],
                  [0, np.sqrt(h2_2 * 0.70)]])

a_mat = np.array([[np.sqrt(h2_1 * prop_lat_1), 0],
                  [0, np.sqrt(h2_2 * 0.30)]])
```

**Adding mtDNA maternal transmission**

`mt_mat` is a diagonal path matrix where `mt_mat[i,i]` scales how much of the mother's phenotype for trait $i$ is passed to the offspring's $MT_i$ component. It must be diagonal — cross-trait mtDNA effects are not biologically supported and will raise a `ValueError`.

```python
# mt_mat[0,0]: mother Y1 → offspring MT1 path
# mt_mat[1,1]: mother Y2 → offspring MT2 path
mt_mat = np.diag([0.30, 0.20])

# Reduce cove_mat to keep total Var(Y) ≈ 1
# Var(MT_i) ≈ mt_mat[i,i]² × Var(Y_i) at equilibrium
mt_var_1 = 0.30**2 * 1.0   # ≈ 0.09
mt_var_2 = 0.20**2 * 1.0   # ≈ 0.04
ve_1_adjusted = (1 - h2_1) - mt_var_1
ve_2_adjusted = (1 - h2_2) - mt_var_2
cove_mat = np.diag([ve_1_adjusted, ve_2_adjusted])
```

> **Note:** The approximation above is only exact when `Var(Y_i) = 1`. For more precise calibration, run the simulation once with a zero phenotypic variance target and measure the empirical `VMT` from `SUMMARY.RES[-1]`.

---

## 9. FAQ / Gotchas

**Q: `ValueError: pop_size must be divisible by n_islands*2`**  
A: With 10 islands, pop_size must be a multiple of 20. E.g. 20 000, 40 000.

**Q: `am_list` length errors**  
A: `am_list` must have length exactly `num_generations`. It does *not* cover burn-in; burn-in always uses `am_list[0]`.

**Q: `within_island_am` as a list — what length?**  
A: It covers `n_burn_in + num_generations` total entries. Index 0 applies to burn-in generation 0, index `n_burn_in` applies to main generation 0. Short lists are padded with the last value.

**Q: How do I check that stratification worked?**  
A: Call `sim.island_summary()` after `run_simulation()`. Island means for `Y2` should be monotonically increasing by `island_id` if `migration_trait=2`.

**Q: How do I get the mate correlation actually achieved?**  
A: `results['SUMMARY.RES'][i]['MATE.COR']` records the look-ahead AM correlation (what was applied *to produce* generation `i+1`). To measure the achieved spousal phenotypic correlation empirically:
```python
mates = results['HISTORY']['MATES'][gen_idx]
m = mates['males.PHENDATA']; f = mates['females.PHENDATA']
print(np.corrcoef(m['Y1'].values.astype(float),
                  f['Y1'].values.astype(float))[0, 1])
```

**Q: Can I add a third trait?**  
A: Not without significant refactoring. The entire codebase is hardwired to a 2-trait model (2×2 matrices, column pairs like `Y1`/`Y2`).

**Q: Does `save_each_gen=False` affect the population trajectory?**  
A: No. It only affects memory: `history` will be `None`. The simulation still runs identically; only the snapshots are skipped.

**Q: How does `n_burn_in` interact with the seed?**  
A: The global NumPy seed is set once at init. Burn-in consumes random draws just like main-phase generations. Two runs with the same `seed` and `n_burn_in` will produce identical results. If you want the same end-of-burn-in state regardless of whether you then record 5 or 10 generations, that is already the case (burn-in is deterministic given `seed`).

**Q: joblib parallel mode — same results as serial?**  
A: Yes, if `seed != 0`. Each island uses a deterministic seed `= global_seed + island_id * 1000 + gen_abs_idx * 37`, so results are independent of `n_jobs`.

**Q: Why must `mt_mat` be diagonal (no cross-trait effects)?**  
A: mtDNA is a single circular chromosome with no recombination. Cross-trait mtDNA effects would imply a single locus simultaneously affecting two independent phenotypic pathways in a quantitative way — not the intended biological model. The diagonal restriction enforces the assumption that mtDNA contributes an independent additive path for each trait. If you need cross-trait maternal transmission, model it through `f_mat` (vertical transmission) instead.

**Q: How does mtDNA differ from vertical transmission (`f_mat`)?**  
A: Both involve maternal phenotype → offspring phenotype paths, but:
- `f_mat` applies to **both parents** symmetrically (path from paternal *and* maternal phenotype to offspring F component)
- `mt_mat` applies **only to mothers** and contributes to a separate `MT` component, not the `F` component
- `mt_mat` must be diagonal; `f_mat` can have cross-trait entries
- `h2.mt` in `SUMMARY.RES` isolates the mtDNA variance; `f_mat` effects are folded into `VF`
