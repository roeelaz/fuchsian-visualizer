# Plan: restrict custom input to Theorem 5.2.5 cases

## Goal

Make `check_division_algebra` fully exact by accepting only inputs covered by
Theorem 5.2.5 (Katok §5.2). Remove the heuristic zero-divisor search entirely.

## Why the restriction is necessary

Theorem 5.2.5 is an exact iff criterion **only when b is prime**:

> b prime, a not a perfect square, a QNR mod b  ↔  A = (a,b/ℚ) is a division algebra

For composite b, or for prime b with a QR mod b, no theorem in §5.2 gives an exact
answer. The 4-variable zero-divisor search used as a fallback is a heuristic that
can give wrong results (counterexample: a=97, b=6 — A splits but the smallest zero
divisor has max|xᵢ|=10, outside the search radius of 8).

Even for prime b, "a QR mod b" does not imply A is not a division algebra:
A=(3,11/ℚ) is a division algebra despite b=11 prime and 3 being a QR mod 11
(the Hilbert symbol at p=3 is −1). So the contrapositive of Thm 5.2.5 does not hold.

## Changes

### 1. `_is_prime` → deterministic Miller-Rabin

Replace trial division with a deterministic Miller-Rabin test using witnesses
{2, 3, 5, 7}. This is provably correct for all n < 3,215,031,751 — far beyond any
b a user will type — and runs in O(log² n) time.

### 2. `check_division_algebra(a, b)` rewrite

Remove the zero-divisor search. Four exact branches, no heuristic:

| Condition | Returns | Theorem |
|-----------|---------|---------|
| a ≤ 0 or a is a perfect square | `(False, reason)` | Thm 5.2.1(i) |
| b not prime | `(False, "b must be prime — Thm 5.2.5 requires prime b")` | scope |
| b prime, a QR mod b | `(False, "a is a QR mod b — Thm 5.2.5 does not apply; choose a with a QNR mod b")` | scope |
| b prime, a QNR mod b | `(True, "Theorem 5.2.5: …")` | Thm 5.2.5 exact |

The word "likely" never appears.

### 3. `_on_generate_quaternion` — block on False

When `check_division_algebra` returns `(False, msg)`, display `msg` in the
description panel and return without generating the group. Only a `True` result
proceeds to generate.

### 4. Tests

Remove: `test_check_div_alg_not_division` (tested the heuristic path).

Add:
- `test_composite_b_rejected` — b=6 → False
- `test_prime_b_qr_rejected` — a=3, b=11 (3 is QR mod 11) → False
- `test_qr_counterexample_not_caught_by_525` — documents that A=(3,11/Q) is
  actually a division algebra but Thm 5.2.5 cannot confirm it

## What is NOT changed

- `gen_quaternion` — unchanged; it operates on any valid a, b
- The two built-in presets Γ(3,5/ℚ) and Γ(2,3/ℚ) — both satisfy Thm 5.2.5 exactly
- The Hilbert symbol approach remains documented in `docs/hilbert_symbol.md` for
  reference; it is not implemented here
