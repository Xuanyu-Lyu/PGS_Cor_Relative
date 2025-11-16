# SLURM Batch Simulation Scripts

## Overview

This directory contains scripts for running approximation simulations on the CU Boulder Research Computing (RC) Alpine cluster using SLURM job arrays.

## File Structure

### Simulation Scripts
- `run_approximation_uni_rc.py` - Univariate approximation (bivariate simulation with 2 independent traits)
- `run_approximation_bi_rc.py` - Bivariate approximation (4 conditions with cross-trait effects)

### Job Submission Scripts
- `submit_approximation_uni.sh` - Submit univariate simulations (10 array tasks)
- `submit_approximation_bi.sh` - Submit bivariate simulations (40 array tasks)

### Post-Processing
- `combine_batch_results.py` - Combine batch results into final summary files

## Configuration

### Population and Simulation Parameters
- **Population size**: N = 40,000 (increased from 20,000 for better power)
- **Generations**: 15 total (saving final 3 generations: 12, 13, 14)
- **Causal variants**: 1,000 SNPs
- **MAF range**: 0.01 to 0.5

### Batch Parameters
- **Total iterations per condition**: 200
- **Iterations per SLURM task**: 20
- **Univariate**: 1 condition × 10 tasks = 10 total tasks
- **Bivariate**: 4 conditions × 10 tasks each = 40 total tasks

### Resource Allocation (per task)
- **Memory**: 150 GB
- **CPUs**: 4 cores
- **Time limit**: 12 hours
- **Queue**: preemptable

## Data Storage

### Scratch Directory (Large Files)
Raw iteration data saved to: `/scratch/alpine/xuly4739/PGS_Cor_Relative/Data/`
- `approximation_uni/Iteration_XXX/` - Individual iteration data
- `approximation_bi/Condition_XX/Iteration_XXX/` - Individual iteration data per condition

### Project Directory (Summary Files)
Summary statistics saved to: `/projects/xuly4739/Py_Projects/PGS_Cor_Relative/Data/`
- `approximation_uni/` - Batch-level summaries (mate_pgs_correlation_*_batch_XXX.csv)
- `approximation_bi/Condition_XX/` - Batch-level summaries per condition

## Usage

### 1. Submit Univariate Simulations

```bash
cd /projects/xuly4739/Py_Projects/PGS_Cor_Relative/Scripts/run_rc
sbatch submit_approximation_uni.sh
```

This submits 10 array tasks, each running 20 iterations (200 total).

### 2. Submit Bivariate Simulations

```bash
sbatch submit_approximation_bi.sh
```

This submits 40 array tasks:
- Tasks 1-10: Condition_01 (200 iterations)
- Tasks 11-20: Condition_02 (200 iterations)
- Tasks 21-30: Condition_03 (200 iterations)
- Tasks 31-40: Condition_04 (200 iterations)

### 3. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# View output logs
tail -f slurm_logs/approx_uni_<JOB_ID>_<TASK_ID>.out
tail -f slurm_logs/approx_bi_<JOB_ID>_<TASK_ID>.out
```

### 4. Combine Results

After all tasks complete successfully:

```bash
python combine_batch_results.py
```

This combines all batch-level files into final summary files:
- `mate_pgs_correlation_trait1_summary.csv` - All iterations combined
- `mate_pgs_correlation_trait2_summary.csv` - All iterations combined  
- `all_iterations_correlations.csv` - All relationship correlations
- `relationship_summary_statistics.csv` - Summary by relationship type
- `overall_conditions_summary.csv` - Summary across all conditions (bivariate only)

## Bivariate Conditions

### Condition_01
- f11=0.10, vg1=0.60, vg2=1.00
- f22=0.20, am22=0.65, rg=0.75

### Condition_02
- f11=0.20, vg1=0.80, vg2=0.50
- f22=0.30, am22=0.45, rg=0.85

### Condition_03
- f11=0.15, vg1=0.65, vg2=0.50
- f22=0.25, am22=0.55, rg=0.85

### Condition_04
- f11=0.10, vg1=0.60, vg2=0.75
- f22=0.15, am22=0.75, rg=0.75

## Output Files

### Per Iteration (in scratch)
- `individual_measures_*.tsv` - Individual phenotypes/genotypes
- `genealogy_*.tsv` - Family structure
- `correlations_iteration_*.csv` - Relationship correlations

### Per Batch (in project)
- `mate_pgs_correlation_trait1_batch_XXX.csv` - Mate PGS correlations for this batch
- `correlations_batch_XXX.csv` - All relationship correlations for this batch
- `relationship_summary_batch_XXX.csv` - Summary statistics for this batch

### Final Combined (in project)
- `mate_pgs_correlation_trait1_summary.csv` - All 200 iterations
- `all_iterations_correlations.csv` - All relationships, all iterations
- `relationship_summary_statistics.csv` - Mean/SD across all iterations

## Troubleshooting

### Job Fails with Memory Error
Increase `#SBATCH --mem=` in submission script (currently 300G)

### Job Times Out
Increase `#SBATCH --time=` in submission script (currently 12:00:00)

### Import Errors
Check that SimulationFunctions directory is properly added to path in Python scripts

### Missing Output Files
Check SLURM error logs: `slurm_logs/approx_*_<JOB_ID>_<TASK_ID>.err`

## Notes

- Each SLURM task is independent and can be re-run individually if it fails
- Raw iteration data in scratch may be purged after 90 days - copy to project if needed
- Batch summary files in project directory are permanent and can be recombined anytime
- The improved positive definite matrix correction is used (Higham's algorithm + shrinkage)
