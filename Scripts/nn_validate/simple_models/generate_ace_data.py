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
    python generate_ace_data.py --n_samples 20000 --output ace_training_data.csv
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
      2. Sample N_pairs from n_pairs_options (controls estimation noise)
      3. Simulate MZ and DZ pairs and compute sample covariance matrices
      4. Record the four unique covariance elements plus the true A, C, E

    Args:
        n_samples:        Number of training samples to generate
        n_pairs_options:  List of twin-pair sample sizes to draw from randomly
        seed:             NumPy random seed

    Returns:
        pd.DataFrame with columns [mz_var, mz_cov, dz_var, dz_cov, N_pairs, A, C, E]
    """
    if n_pairs_options is None:
        n_pairs_options = [200, 500, 1000, 2000, 5000]

    np.random.seed(seed)

    records = []

    for i in range(n_samples):
        # Sample A and C uniformly, then set E = 1 - A - C (enforces A+C+E = 1 exactly)
        ace = np.random.dirichlet([1, 1, 1])
        A, C = float(ace[0]), float(ace[1])
        E = 1.0 - A - C  # E is the remainder so the sum is exactly 1

        N_pairs = int(np.random.choice(n_pairs_options))

        mz_cov, dz_cov = simulate_covariances(A, C, E, N_pairs)

        records.append({
            'mz_var': mz_cov[0, 0],
            'mz_cov': mz_cov[0, 1],
            'dz_var': dz_cov[0, 0],
            'dz_cov': dz_cov[0, 1],
            'N_pairs': N_pairs,
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
    parser.add_argument('--output', type=str, default='ace_training_data.csv',
                        help='Output CSV filename (default: ace_training_data.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Resolve output path relative to this script's directory
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    print("=" * 60)
    print("GENERATING ACE TRAINING DATA")
    print("=" * 60)
    print(f"  Samples:    {args.n_samples}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {output_path}")
    print()

    df = generate_training_data(n_samples=args.n_samples, seed=args.seed)

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
