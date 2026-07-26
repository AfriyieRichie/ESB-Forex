"""Score a blind review pack: does selectivity beat taking everything?

The comparison is taken vs skipped **within the same pool**, so the pool's own
expectancy cancels out. If the answer is yes, the edge is in the judgement, not
in the rules - which changes what is worth building next.

Significance comes from a permutation test rather than a normal approximation:
shuffle the take/skip labels many times and see how often chance produces a gap
this large. It is exact, assumes nothing about the shape of R, and copes with
the small samples this experiment will realistically have.

Caveat it cannot fix: trades overlapping in time or sharing a currency are not
independent, so the true p-value is somewhat worse than the printed one.

    uv run python scripts/review_report.py --pack review/pack-01
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

PERMUTATIONS = 20000


def load_pack(pack: Path):
    key = {}
    with (pack / "ANSWER_KEY_do_not_open.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key[row["review_id"]] = row

    decisions = {}
    with (pack / "decisions.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            verdict = (row.get("decision") or "").strip().lower()
            if verdict in {"take", "skip"}:
                decisions[row["review_id"].strip()] = {
                    "decision": verdict,
                    "confidence": (row.get("confidence") or "").strip(),
                    "note": (row.get("note") or "").strip(),
                }
    return key, decisions


def describe(label: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{label:<12} n=   0"
    wins = int((values > 0).sum())
    return (
        f"{label:<12} n={values.size:>4}  win={wins / values.size:>5.1%}  "
        f"E[R]={values.mean():>+6.3f}  totalR={values.sum():>+7.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("review/pack-01"))
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    key, decisions = load_pack(args.pack)
    if not decisions:
        raise SystemExit("no decisions recorded yet - fill in decisions.csv first")

    missing = len(key) - len(decisions)
    taken = np.array([key[i]["r_multiple"] for i in decisions if decisions[i]["decision"] == "take"])
    skipped = np.array([key[i]["r_multiple"] for i in decisions if decisions[i]["decision"] == "skip"])
    everything = np.concatenate([taken, skipped]) if taken.size + skipped.size else np.array([])

    print("=" * 76)
    print(f"BLIND REVIEW: {args.pack}")
    print("=" * 76)
    print(f"decided {len(decisions)}/{len(key)}" + (f"  ({missing} undecided)" if missing else ""))
    print()
    print(describe("TAKEN", taken))
    print(describe("SKIPPED", skipped))
    print(describe("ALL", everything))

    if taken.size == 0 or skipped.size == 0:
        print("\nneed both takes and skips to compare")
        return

    observed = taken.mean() - skipped.mean()
    print(f"\nselection effect (taken - skipped): {observed:+.3f}R")

    # Permutation test: how often does a random split of the same sizes
    # produce a gap at least this large?
    rng = np.random.default_rng(args.seed)
    pooled = everything.copy()
    n_taken = taken.size
    diffs = np.empty(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        rng.shuffle(pooled)
        diffs[i] = pooled[:n_taken].mean() - pooled[n_taken:].mean()

    p_two_sided = float((np.abs(diffs) >= abs(observed)).mean())
    print(f"permutation p = {p_two_sided:.4f}  ({PERMUTATIONS} shuffles)")
    print(
        f"random splits of this size land in "
        f"[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}] 95% of the time"
    )

    if p_two_sided < 0.05 and observed > 0:
        print("\n  -> selectivity beat the pool. The edge is in the judgement.")
        print("     Next: characterise WHAT was selected, then test that rule out-of-sample.")
    elif p_two_sided < 0.05:
        print("\n  -> selection was significantly WORSE than random.")
        print("     Worth knowing: the instinct is anti-predictive here.")
    else:
        print("\n  -> no detectable selection effect. Judgement did not beat taking everything.")
        print("     Note this only rules out effects this sample was large enough to see.")

    print("\n--- what was taken ---")
    for field in ("pattern", "zone_tier", "zone_origin"):
        counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for review_id, decision in decisions.items():
            bucket = counts[key[review_id][field]]
            bucket[0 if decision["decision"] == "take" else 1] += 1
        summary = "  ".join(
            f"{k}={v[0]}/{v[0] + v[1]}" for k, v in sorted(counts.items())
        )
        print(f"{field:<12} {summary}")

    confidences = {
        review_id: int(decision["confidence"])
        for review_id, decision in decisions.items()
        if decision["confidence"].isdigit()
    }
    if len(confidences) >= 20:
        print("\n--- does stated confidence track outcome? ---")
        by_level: dict[int, list[float]] = collections.defaultdict(list)
        for review_id, level in confidences.items():
            by_level[level].append(key[review_id]["r_multiple"])
        for level in sorted(by_level):
            print(describe(f"conf {level}", np.array(by_level[level])))


if __name__ == "__main__":
    main()
