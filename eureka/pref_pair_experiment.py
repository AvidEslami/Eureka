#!/usr/bin/env python3
# Preference-pair generation with minimal checks via transitive reasoning
# and comparison to the naive (check-every-pair) approach.

from dataclasses import dataclass
from typing import List, Tuple
import argparse
import random
import math

@dataclass
class ComparisonCounter:
    count: int = 0
    def compare(self, a: str, b: str) -> int:
        """Return -1 if a<b, 0 if a==b, 1 if a>b under the ground truth:
        preference goes to the numerically higher string."""
        self.count += 1
        ai, bi = int(a), int(b)
        if ai < bi: return -1
        if ai > bi: return 1
        return 0

def binary_insert_sorted(items: List[str]) -> Tuple[List[str], int]:
    """
    Build a total order using only comparisons and transitive reasoning.
    Uses binary search insertion to minimize checks.
    Returns (ordered_items, num_checks).
    """
    cmp = ComparisonCounter()
    ordered: List[str] = []
    for x in items:
        lo, hi = 0, len(ordered)
        while lo < hi:
            mid = (lo + hi) // 2
            c = cmp.compare(x, ordered[mid])
            if c <= 0:
                hi = mid
            else:
                lo = mid + 1
        ordered.insert(lo, x)
    return ordered, cmp.count

def build_all_pairs_from_total_order(ordered: List[str]) -> List[Tuple[str, str]]:
    """
    Given the total order (ascending numerically), produce all preference pairs (winner, loser),
    inferred transitively without further checks.
    """
    pairs: List[Tuple[str, str]] = []
    n = len(ordered)
    for i in range(n):
        for j in range(i + 1, n):
            # ordered[j] > ordered[i] so winner is ordered[j], loser is ordered[i]
            pairs.append((ordered[j], ordered[i]))
    return pairs

def naive_checks(n: int) -> int:
    """Naive approach needs one check per unordered pair."""
    return n * (n - 1) // 2

def run_experiment(n: int, shuffle: bool = True, seed: int = 0, show_sample_pairs: int = 10):
    # Create items as strings "1".."n"
    items = [str(i) for i in range(1, n + 1)]
    if shuffle:
        random.seed(seed)
        random.shuffle(items)

    # Efficient approach: sort via comparisons (binary insertion)
    ordered, efficient_checks = binary_insert_sorted(items)

    # Sanity check: ordered should be ascending numerically
    assert ordered == sorted(items, key=lambda s: int(s)), "Ordering failed sanity check."

    # Infer all pairs without extra checks
    inferred_pairs = build_all_pairs_from_total_order(ordered)
    total_pairs = len(inferred_pairs)
    assert total_pairs == naive_checks(n), "Pair count mismatch."

    # Baseline
    naive = naive_checks(n)

    # Report
    reduction = 100.0 * (naive - efficient_checks) / naive if naive > 0 else 0.0

    print(f"\n=== Preference Pair Experiment ===")
    print(f"N: {n}")
    print(f"Total pairs (C(N,2)): {total_pairs:,}")
    print(f"Checks (Naive):        {naive:,}")
    print(f"Checks (Efficient):    {efficient_checks:,}")
    print(f"Savings vs Naive:      {naive - efficient_checks:,} checks")
    print(f"Reduction:             {reduction:.2f}%")

    if show_sample_pairs > 0:
        print("\nSample inferred pairs (winner, loser):")
        for w, l in inferred_pairs[:show_sample_pairs]:
            print(f"  ({w}, {l})")
        if total_pairs > show_sample_pairs:
            print(f"  ... [{total_pairs - show_sample_pairs} more]")

    return {
        "ordered": ordered,
        "inferred_pairs": inferred_pairs,
        "checks_efficient": efficient_checks,
        "checks_naive": naive
    }

def run_sweep(Ns: List[int], seed: int = 0):
    print("\n=== Sweep ===")
    print(f"{'N':>6} | {'Pairs':>12} | {'Naive':>12} | {'Efficient':>12} | {'Savings':>12} | {'Reduction %':>11}")
    print("-" * 79)
    for n in Ns:
        res = run_experiment(n, shuffle=True, seed=seed, show_sample_pairs=0)
        pairs = naive_checks(n)
        print(f"{n:6d} | {pairs:12,d} | {res['checks_naive']:12,d} | {res['checks_efficient']:12,d} | "
              f"{(res['checks_naive']-res['checks_efficient']):12,d} | "
              f"{100.0*(res['checks_naive']-res['checks_efficient'])/res['checks_naive']:11.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient preference-pair generation experiment.")
    parser.add_argument("--n", type=int, default=1000, help="Number of items (strings '1'..'N').")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for shuffling input order.")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of insertion order.")
    parser.add_argument("--sweep", action="store_true", help="Run a sweep over preset N values.")
    args = parser.parse_args()

    if args.sweep:
        run_sweep([50, 100, 200, 500, 1000], seed=args.seed)
    else:
        run_experiment(args.n, shuffle=not args.no_shuffle, seed=args.seed, show_sample_pairs=10)
