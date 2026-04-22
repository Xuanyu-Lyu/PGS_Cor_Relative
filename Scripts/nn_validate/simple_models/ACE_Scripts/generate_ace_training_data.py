"""
Generate ACE Training Data

Simulates MZ and DZ twin covariance matrices for various A, C, E combinations
and saves them to a CSV file for neural network training.

The ACE model:
  - A: additive genetic variance (heritability)
  - C: shared/common environment variance
  - E: unique environment variance (+ measurement error)
  - Total = A + C + E = 1 (standardized; E is set as 1 - A - C exactly)

Theoretical covariance matrices:
  MZ twins: Var = A+C+E, Cov = A+C      (share 100% genes + 100% shared env)
  DZ twins: Var = A+C+E, Cov = 0.5A+C   (share 50% genes + 100% shared env)

Features saved:
  mz_var, mz_cov, dz_var, dz_cov  (unique elements of the 2x2 covariance matrices)

Targets saved:
  A, C, E

Usage:
    # Default: draw N randomly from [50, 100, 200, 500, 1000, 2000, 5000]
    python generate_ace_training_data.py --n_samples 20000 --output ace_training_data.csv

    # Fixed N for all samples (useful for training a no-N model)
    python generate_ace_training_data.py --n_pairs 2000 --n_samples 20000 --output ace_training_data_N500.csv

    # Draw randomly from a custom vector
    python generate_ace_training_data.py --n_pairs 200 500 1000 2000 --output ace_training_data.csv
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Allow importing simulate_covariances from ACEnn in the same folder
sys.path.insert(0, str(Path(__file__).parent))
from ACEnn import simulate_covariances


def generate_training_data(n_samples=20000, n_pairs_options=None, seed=42):
    """
    Generate training data for ACE neural network.

    For each sample:
      1. Sample A, C, E from the unit simplex (Dirichlet, so A+C+E = 1)
      2. Determine N_pairs — fixed if n_pairs_options is a scalar, else drawn
         randomly from the provided vector.
      3. Simulate MZ and DZ pairs and compute sample covariance matrices
      4. Record the four unique covariance elements plus the true A, C, E

    Args:
        n_samples:        Number of training samples to generate
        n_pairs_options:  A single int (fixed N for all samples) or a list of
                          ints to draw from randomly.  Defaults to the vector
                          [50, 100, 200, 500, 1000, 2000, 5000].
        seed:             NumPy random seed

    Returns:
        pd.DataFrame with columns [mz_var, mz_cov, dz_var, dz_cov, N_pairs, A, C, E]
    """
    if n_pairs_options is None:
        n_pairs_options = [50, 100, 200, 500, 1000, 2000, 5000]

    # Normalise to a list so we can check its length
    if isinstance(n_pairs_options, (int, np.integer)):
        n_pairs_options = [int(n_pairs_options)]
    fixed_n = len(n_pairs_options) == 1

    np.random.seed(seed)

    records = []

    for i in range(n_samples):
        # Sample uniformly over the simplex {A+C+E=1, A≥-0.2, C≥-0.2, E≥-0.2}.
        # Shift trick: Dirichlet gives (a,c,e) in [0,1] summing to 1; scaling by
        # 1.6 and subtracting 0.2 maps each component to [-0.2, 1.4] while
        # preserving A+C+E = 1.6*1 - 3*0.2 = 1.0.
        ace = np.random.dirichlet([1, 1, 1])
        A = float(ace[0]) * 1.6 - 0.2
        C = float(ace[1]) * 1.6 - 0.2
        E = 1.0 - A - C  # ensures sum = 1 exactly; equivalent to ace[2]*1.6 - 0.2

        N_pairs = n_pairs_options[0] if fixed_n else int(np.random.choice(n_pairs_options))

        mz_cov, dz_cov = simulate_covariances(A, C, E, N_pairs)

        records.append({
            'mz_var': mz_cov[0, 0],
            'mz_cov': mz_cov[0, 1],
            'dz_var': dz_cov[0, 0],
            'dz_cov': dz_cov[0, 1],
            'N_pairs': N_pairs,
            'log_N_pairs': np.log(N_pairs),
            'A': A,
            'C': C,
            'E': E,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Generate ACE training data')
    parser.add_argument('--n_samples', type=int, default=20000,
                        help='Number of training samples (default: 20000)')
    parser.add_argument('--n_pairs', type=int, nargs='+', default=None,
                        help='Twin-pair sample size(s). Pass a single integer '
                             'to fix N for all samples (e.g. --n_pairs 500), '
                             'or multiple integers to draw randomly from that '
                             'vector (e.g. --n_pairs 50 100 200 500 1000 2000). '
                             'Defaults to [50, 100, 200, 500, 1000, 2000, 5000].')
    parser.add_argument('--output', type=str, default='ace_training_data.csv',
                        help='Output CSV filename (default: ace_training_data.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Resolve output path relative to this script's directory
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    # Determine n_pairs_options from CLI argument
    n_pairs_options = args.n_pairs  # None → use default inside generate_training_data

    print("=" * 60)
    print("GENERATING ACE TRAINING DATA")
    print("=" * 60)
    print(f"  Samples:    {args.n_samples}")
    print(f"  N pairs:    {n_pairs_options if n_pairs_options is not None else '[50,100,200,500,1000,2000,5000] (default)'}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {output_path}")
    print()

    df = generate_training_data(n_samples=args.n_samples,
                                n_pairs_options=n_pairs_options,
                                seed=args.seed)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} rows to {output_path}")
    print(f"\nColumn summary:")
    print(df.describe().to_string())

    # Quick sanity check: theoretical MZ cov ≈ A+C, DZ cov ≈ 0.5A+C
    print("\nSanity check (mean absolute deviation from theoretical values):")
    print(f"  |mz_cov - (A+C)|   mean: {(df['mz_cov'] - (df['A'] + df['C'])).abs().mean():.4f}")
    print(f"  |dz_cov - (.5A+C)| mean: {(df['dz_cov'] - (0.5*df['A'] + df['C'])).abs().mean():.4f}")


if __name__ == "__main__":
    main()
