# Code Extraction Bug: Wrong Block Selected

## The Bug

`extract_code()` in `utils.py:78` always takes the **last** fenced code block:

```python
def extract_code(text: str) -> str | None:
    """Extract the last fenced code block from text."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()  # <-- always takes LAST block
    return None
```

Math-distilled models learned verbose reasoning from Sonnet/R1 traces. They often:
1. Output the solution in a code block
2. Write verification text ending with a closing ` ``` `
3. Re-state the solution in another ` ```python ` block

The regex matches the closing ``` of step 2 and the opening ``` of step 3 as a **zero-length code block**. This empty block becomes the "last" block, so `extract_code` returns an empty string. The real code is never extracted.

**Impact on `sonnet_math_qwen_4k_step300`:** MBPP+ goes from 57% to 73% with a fix (+16 points).
The apparent cross-domain forgetting (baseline 75.7% → 57%) is mostly an extraction artifact.
Real drop is 75.7% → 73%, only ~3 points.

---

## Example 1: `next_smallest_palindrome` (task 100)

**Prompt:** Write a function to find the next smallest palindrome of a specified integer, returned as an integer.

**Error:** `NameError: name 'next_smallest_palindrome' is not defined`

### Model output (after `</think>`)

The model produces this text (with 3 fenced regions marked):

    ▼ BLOCK 0 — the actual solution (354 chars)
    ┌─────────────────────────────────────────────────
    │ ```python
    │ def is_palindrome(num):
    │     """Check if a number is a palindrome"""
    │     s = str(num)
    │     return s == s[::-1]
    │
    │ def next_smallest_palindrome(n):
    │     """Find the next smallest palindrome after n"""
    │     candidate = n + 1
    │     while True:
    │         if is_palindrome(candidate):
    │             return candidate
    │         candidate += 1
    │ ```
    └─────────────────────────────────────────────────

    Let me verify with the test cases:

    ▼ BLOCK 1 — verification assertions (133 chars)
    ┌─────────────────────────────────────────────────
    │ ```python
    │ assert next_smallest_palindrome(99) == 101
    │ assert next_smallest_palindrome(1221) == 1331
    │ assert next_smallest_palindrome(120) == 121
    │ ```
    └─────────────────────────────────────────────────

    For `next_smallest_palindrome(99)`:
    - Start at 100: not a palindrome
    - Check 101: is a palindrome ✓
    ...
    ```                               ← stray closing fence

    ```python                         ← intended to start a new block
    def is_palindrome(num):
        ...
    def next_smallest_palindrome(n):
        ...
    ```

### What the regex sees

The regex `r"```(?:python)?\s*\n(.*?)```"` matches 3 regions:

| Match | Content | Size |
|-------|---------|------|
| Block 0 | `def is_palindrome...candidate += 1` | 354 chars |
| Block 1 | `assert next_smallest_palindrome(99) == 101...` | 133 chars |
| **Block 2** | **(empty string)** | **0 chars** |

Block 2 is 0 chars because the regex matches the stray closing ` ``` ` and the opening ` ```python ` of the re-stated solution as a single fence pair with nothing inside. The real final code block is consumed by this match and never captured.

### What `extract_code()` returns

**Empty string** (block 2, the last match). This gets written to `solution.py`. The test suite tries to call `next_smallest_palindrome()` → `NameError`.

### What it should return

Block 0 — the actual solution with both function definitions. It passes all tests.

---

## Example 2: `check_monthnumb_number` (task 455)

**Prompt:** Write a function to check whether the given month number contains 31 days or not.

**Error:** `NameError: name 'check_monthnumb_number' is not defined`

### Model output (after `</think>`)

    I need to write a function that checks if a given month number contains 31 days.
    Months with 31 days: 1, 3, 5, 7, 8, 10, 12

    Now I'll write the function:

    ▼ BLOCK 0 — the actual solution
    ┌─────────────────────────────────────────────────
    │ ```python
    │ def check_monthnumb_number(month):
    │     months_with_31_days = {1, 3, 5, 7, 8, 10, 12}
    │     return month in months_with_31_days
    │ ```
    └─────────────────────────────────────────────────

    Let me verify this works:
    - check_monthnumb_number(5) → True ✓
    - check_monthnumb_number(2) → False ✓
    - check_monthnumb_number(6) → False ✓
    ```                               ← stray closing fence

    ```python                         ← intended re-statement
    def check_monthnumb_number(month):
        months_with_31_days = {1, 3, 5, 7, 8, 10, 12}
        return month in months_with_31_days
    ```

### What the regex sees

| Match | Content | Size |
|-------|---------|------|
| Block 0 | `def check_monthnumb_number...` | 195 chars |
| **Block 1** | **(empty string)** | **0 chars** |

Same pattern: stray ``` + opening ```python = zero-length match. The re-stated solution is eaten.

**Extracted:** empty string → `NameError`
**Should extract:** Block 0 (passes all tests)

---

## Example 3: `min_k` (task 753)

**Prompt:** Write a function to find minimum k records from tuple list.

**Error:** `NameError: name 'min_k' is not defined`

### Model output (after `</think>`)

    ▼ BLOCK 0 — the actual solution
    ┌─────────────────────────────────────────────────
    │ ```python
    │ def min_k(lst, k):
    │     sorted_lst = sorted(lst, key=lambda x: x[1])
    │     return sorted_lst[:k]
    │ ```
    └─────────────────────────────────────────────────

    Let me verify with the test cases:

    Test case 1:
    ▼ BLOCK 1 — assertion
    ┌─────────────────────────────────────────────────
    │ ```python
    │ assert min_k([('Manjeet', 10), ...], 2) == [('Akash', 2), ('Akshat', 4)]
    │ ```
    └─────────────────────────────────────────────────
    - Sorted: [...] - First 2: [...] ✓

    Test case 2:
    ▼ BLOCK 2 — assertion
    ...

    Test case 3:
    ▼ BLOCK 3 — assertion
    ...
    ```                               ← stray closing fence

    ```python                         ← intended re-statement of solution
    def min_k(lst, k):
        ...
    ```

### What the regex sees

| Match | Content | Size |
|-------|---------|------|
| Block 0 | `def min_k(lst, k):...` | 189 chars |
| Block 1 | `assert min_k(...)` | 112 chars |
| Block 2 | `assert min_k(...)` | 123 chars |
| Block 3 | `assert min_k(...)` | 95 chars |
| **Block 4** | **(empty string)** | **0 chars** |

**Extracted:** empty string → `NameError`
**Should extract:** Block 0 (passes all tests)

---

## Aggregate Impact

Tested across all Qwen eval parquets (5880 total questions):

| Category | Count | % of failures |
|----------|-------|---------------|
| No code block (truncation) | 928 | 43.3% |
| Wrong answer | 842 | 39.3% |
| Runtime/other errors | 257 | 12.0% |
| **Wrong block extracted** | **117** | **5.5%** |

Of the 117 wrong-block cases, **80 would pass** if we used the first `def`-containing block.

### Per-model impact (most affected)

| Model | Current MBPP+ | Fixed MBPP+ | Delta |
|-------|--------------|-------------|-------|
| sonnet_math_qwen_4k_step300 | 57.0% | 73.0% | +16 |
| math_4k_lr5_step400 | 48.0% | 60.0% | +12 |
| sonnet_math_qwen_4k_step200 | 62.0% | 72.0% | +10 |
| sonnet_math_qwen_4k_final | 62.0% | 71.0% | +9 |
| sonnet_math_qwen_4k_step400 | 63.0% | 72.0% | +9 |

The bug disproportionately affects **math-distilled models evaluated on code** — exactly the scenario where we're measuring cross-domain transfer. This inflates the apparent catastrophic forgetting.

## Suggested Fix

Replace `blocks[-1]` with: find the first block containing a `def` statement, or filter out empty/assertion-only blocks.
