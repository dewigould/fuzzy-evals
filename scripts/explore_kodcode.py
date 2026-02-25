"""
Explore the KodCode/KodCode-Light-RL-10K dataset from HuggingFace.
"""
import sys
import re
from collections import Counter

sys.path.insert(0, "/workspace/tinker-cookbook")

from dotenv import load_dotenv
load_dotenv("/workspace/.env")

import datasets

print("=" * 80)
print("LOADING DATASET: KodCode/KodCode-Light-RL-10K")
print("=" * 80)

ds = datasets.load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
print(f"\nDataset size: {len(ds)} examples")
print(f"Columns: {ds.column_names}")
print(f"Features: {ds.features}")

# ──────────────────────────────────────────────────────────────────────
# 1. gpt_difficulty distribution
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1. GPT_DIFFICULTY DISTRIBUTION")
print("=" * 80)

difficulty_counts = Counter(ds["gpt_difficulty"])
for diff, count in sorted(difficulty_counts.items(), key=lambda x: -x[1]):
    print(f"  {diff!r:>20s}: {count:5d}  ({count/len(ds)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# 2. gpt_pass_percentage distribution (histogram buckets)
#    NOTE: Values are in [0, 1] scale (not 0-100)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. GPT_PASS_PERCENTAGE DISTRIBUTION")
print("=" * 80)

pass_pcts = ds["gpt_pass_percentage"]
print(f"  Type of first value: {type(pass_pcts[0])}, value: {pass_pcts[0]!r}")
print(f"  Sample values (first 20): {pass_pcts[:20]}")

numeric_pcts = []
none_count = 0
for v in pass_pcts:
    if v is None:
        none_count += 1
    else:
        numeric_pcts.append(float(v))

# Values are 0-1, so adjust bucket boundaries
buckets = {
    "exactly 0% (0.0)": 0,
    "(0%, 25%] (0.0-0.25]": 0,
    "(25%, 50%] (0.25-0.50]": 0,
    "(50%, 75%] (0.50-0.75]": 0,
    "(75%, 100%) (0.75-1.0)": 0,
    "exactly 100% (1.0)": 0,
}

for v in numeric_pcts:
    if v == 0.0:
        buckets["exactly 0% (0.0)"] += 1
    elif v == 1.0:
        buckets["exactly 100% (1.0)"] += 1
    elif v <= 0.25:
        buckets["(0%, 25%] (0.0-0.25]"] += 1
    elif v <= 0.50:
        buckets["(25%, 50%] (0.25-0.50]"] += 1
    elif v <= 0.75:
        buckets["(50%, 75%] (0.50-0.75]"] += 1
    else:
        buckets["(75%, 100%) (0.75-1.0)"] += 1

print(f"\n  None/missing: {none_count}")
for label, count in buckets.items():
    print(f"  {label:>35s}: {count:5d}  ({count/len(ds)*100:.1f}%)")

if numeric_pcts:
    sorted_pcts = sorted(numeric_pcts)
    print(f"\n  Min: {min(numeric_pcts)}, Max: {max(numeric_pcts)}, "
          f"Mean: {sum(numeric_pcts)/len(numeric_pcts):.4f}")
    # Show unique values
    unique_vals = sorted(set(numeric_pcts))
    print(f"  Number of unique values: {len(unique_vals)}")
    if len(unique_vals) <= 30:
        val_counts = Counter(numeric_pcts)
        print(f"  Value counts:")
        for v in unique_vals:
            print(f"    {v:.2f}: {val_counts[v]:5d}")
    else:
        # Show percentiles
        n = len(sorted_pcts)
        for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
            idx = min(int(n * p / 100), n - 1)
            print(f"    P{p:3d}: {sorted_pcts[idx]:.4f}")

# Also check gpt_pass_sequence and gpt_pass_trial_num
print(f"\n  gpt_pass_trial_num unique values: {sorted(set(ds['gpt_pass_trial_num']))[:20]}")
print(f"  gpt_pass_sequence example (row 0): {ds[0]['gpt_pass_sequence']}")
print(f"  gpt_pass_sequence example (row 2): {ds[2]['gpt_pass_sequence']}")

# ──────────────────────────────────────────────────────────────────────
# 3. Example problems at different difficulties
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. EXAMPLE PROBLEMS AT DIFFERENT DIFFICULTIES")
print("=" * 80)

sorted_diffs = sorted(difficulty_counts.keys())
print(f"  All difficulty levels: {sorted_diffs}")

# For each difficulty, find a representative example
targets = sorted_diffs  # easy, hard, medium
chosen = []
for target in targets:
    for i, row in enumerate(ds):
        if row["gpt_difficulty"] == target:
            chosen.append((i, row))
            break

for idx, (i, row) in enumerate(chosen):
    print(f"\n  --- Example {idx+1} (index={i}) ---")
    print(f"  gpt_difficulty: {row['gpt_difficulty']!r}")
    print(f"  gpt_pass_percentage: {row['gpt_pass_percentage']!r}")
    print(f"  gpt_pass_sequence: {row['gpt_pass_sequence']}")
    print(f"  gpt_pass_trial_num: {row['gpt_pass_trial_num']}")

    q = row.get("question", "N/A")
    if isinstance(q, str):
        print(f"  question (first 300 chars):\n    {q[:300]}")
    else:
        print(f"  question: {q!r}")

    t = row.get("test", "N/A")
    if isinstance(t, str):
        print(f"  test (first 500 chars):\n    {t[:500]}")
    else:
        print(f"  test: {t!r}")

    ti = row.get("test_info", "N/A")
    print(f"  test_info: {ti}")

# ──────────────────────────────────────────────────────────────────────
# 4. Test format analysis: "from solution import ..." patterns
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. TEST FORMAT ANALYSIS")
print("=" * 80)

from_solution_count = 0
other_start_patterns = Counter()
total_with_test = 0

# Also track what exactly is imported
from_solution_imports = Counter()

for row in ds:
    test = row.get("test", "")
    if not test or not isinstance(test, str):
        continue
    total_with_test += 1
    stripped = test.strip()

    if stripped.startswith("from solution import"):
        from_solution_count += 1
        # Extract what's imported
        first_line = stripped.split("\n")[0]
        from_solution_imports[first_line[:100]] += 1
    else:
        first_line = stripped.split("\n")[0][:80]
        other_start_patterns[first_line] += 1

print(f"  Total examples with test field: {total_with_test}")
print(f"  Tests starting with 'from solution import': {from_solution_count} "
      f"({from_solution_count/total_with_test*100:.1f}%)")
print(f"  Tests with other patterns: {total_with_test - from_solution_count} "
      f"({(total_with_test - from_solution_count)/total_with_test*100:.1f}%)")

print(f"\n  Top 15 other starting patterns:")
for pattern, count in other_start_patterns.most_common(15):
    print(f"    {count:4d}x  {pattern}")

# Check how many also contain from solution import later
contains_from_solution_anywhere = 0
from_solution_re = re.compile(r"from\s+solution\s+import")
for row in ds:
    test = row.get("test", "")
    if not test or not isinstance(test, str):
        continue
    if from_solution_re.search(test):
        contains_from_solution_anywhere += 1

print(f"\n  Contains 'from solution import' anywhere: {contains_from_solution_anywhere}")
print(f"  Does NOT contain 'from solution import': {total_with_test - contains_from_solution_anywhere}")

# Show some examples that DON'T have from solution import
print(f"\n  Examples WITHOUT 'from solution import' (first 3):")
no_import_count = 0
for i, row in enumerate(ds):
    test = row.get("test", "")
    if not test or not isinstance(test, str):
        continue
    if not from_solution_re.search(test):
        no_import_count += 1
        if no_import_count <= 3:
            print(f"\n    --- No-import example {no_import_count} (index={i}) ---")
            print(f"    difficulty: {row['gpt_difficulty']}")
            print(f"    test (first 600 chars):")
            for line in test[:600].split("\n"):
                print(f"      {line}")

# ──────────────────────────────────────────────────────────────────────
# 5. Filtering for 8B-achievable problems
#    Values are in [0, 1] scale
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5. FILTERING FOR 8B-MODEL-ACHIEVABLE PROBLEMS")
print("=" * 80)

thresholds = [
    ("gpt_pass_percentage > 0", lambda r: r["gpt_pass_percentage"] is not None and r["gpt_pass_percentage"] > 0),
    ("gpt_pass_percentage >= 0.25", lambda r: r["gpt_pass_percentage"] is not None and r["gpt_pass_percentage"] >= 0.25),
    ("gpt_pass_percentage >= 0.50", lambda r: r["gpt_pass_percentage"] is not None and r["gpt_pass_percentage"] >= 0.50),
    ("gpt_pass_percentage >= 0.75", lambda r: r["gpt_pass_percentage"] is not None and r["gpt_pass_percentage"] >= 0.75),
    ("gpt_pass_percentage == 1.0", lambda r: r["gpt_pass_percentage"] is not None and r["gpt_pass_percentage"] == 1.0),
]

for label, predicate in thresholds:
    count = sum(1 for r in ds if predicate(r))
    print(f"  {label:>35s}: {count:5d} examples ({count/len(ds)*100:.1f}%)")

# Cross-tab: difficulty x pass_percentage buckets
print("\n  Cross-tabulation: difficulty x pass_percentage bucket")
header = f"  {'Difficulty':<12s} | {'0.0':>6s} | {'0-0.25':>6s} | {'0.25-0.5':>8s} | {'0.5-0.75':>8s} | {'0.75-1.0':>8s} | {'1.0':>6s} | {'Total':>6s}"
print(header)
print("  " + "-" * len(header))

for diff in sorted(difficulty_counts.keys()):
    b = {"0.0": 0, "0-0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0, "0.75-1.0": 0, "1.0": 0}
    total = 0
    for row in ds:
        if row["gpt_difficulty"] != diff:
            continue
        total += 1
        v = row["gpt_pass_percentage"]
        if v is None:
            continue
        if v == 0.0:
            b["0.0"] += 1
        elif v == 1.0:
            b["1.0"] += 1
        elif v <= 0.25:
            b["0-0.25"] += 1
        elif v <= 0.50:
            b["0.25-0.5"] += 1
        elif v <= 0.75:
            b["0.5-0.75"] += 1
        else:
            b["0.75-1.0"] += 1
    print(f"  {diff:<12s} | {b['0.0']:>6d} | {b['0-0.25']:>6d} | {b['0.25-0.5']:>8d} | {b['0.5-0.75']:>8d} | {b['0.75-1.0']:>8d} | {b['1.0']:>6d} | {total:>6d}")

# Also check r1 fields (DeepSeek R1 results)
print("\n  --- R1 (DeepSeek R1) correctness distribution ---")
r1_correctness = Counter(ds["r1_correctness"])
for val, count in sorted(r1_correctness.items(), key=lambda x: -x[1]):
    print(f"    {val!r:>20s}: {count:5d}  ({count/len(ds)*100:.1f}%)")

# Filter recommendations
print("\n  RECOMMENDATION for 8B model filtering:")
print("  - gpt_pass_percentage is on 0-1 scale (fraction, not percent)")
print("  - gpt_pass_percentage > 0 gives 9994/10000 problems (almost all)")
print("  - For an 8B model, suggested filters:")
print("    * difficulty='easy' AND gpt_pass_percentage >= 0.5 for most achievable")
print("    * difficulty in ('easy','medium') AND gpt_pass_percentage > 0 for broader set")
print("    * Use r1_correctness to cross-reference (R1 is closer to 8B capability)")

# Count combined filters
easy_high = sum(1 for r in ds if r["gpt_difficulty"] == "easy" and r["gpt_pass_percentage"] and r["gpt_pass_percentage"] >= 0.5)
easy_med = sum(1 for r in ds if r["gpt_difficulty"] in ("easy", "medium") and r["gpt_pass_percentage"] and r["gpt_pass_percentage"] > 0)
r1_correct = sum(1 for r in ds if r["r1_correctness"] == "correct")

print(f"\n  Combined filter counts:")
print(f"    easy + pass >= 0.5:         {easy_high:5d}")
print(f"    easy/medium + pass > 0:     {easy_med:5d}")
print(f"    r1_correctness == 'correct': {r1_correct:5d}")

# Show subset and style distributions
print("\n  --- 'subset' distribution ---")
subset_counts = Counter(ds["subset"])
for val, count in sorted(subset_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"    {val!r:>30s}: {count:5d}")

print("\n  --- 'style' distribution ---")
style_counts = Counter(ds["style"])
for val, count in sorted(style_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"    {val!r:>30s}: {count:5d}")

print("\n  --- 'version' distribution ---")
version_counts = Counter(ds["version"])
for val, count in sorted(version_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"    {val!r:>30s}: {count:5d}")

# Check conversations field
print("\n  --- 'conversations' field sample ---")
conv = ds[0]["conversations"]
print(f"    Length: {len(conv)}")
for c in conv:
    print(f"    from={c['from']!r}, value[:100]={c['value'][:100]!r}")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
