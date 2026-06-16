# Division algebras and the Hilbert symbol

This note explains the exact criterion for deciding whether the quaternion algebra
A = (a,b/ℚ) is a division algebra — going one theorem beyond what Katok §5.2 proves.

## What Katok proves

**Theorem 5.2.3**: A is a division algebra if and only if the reduced norm
Nrd(x) = x₀² − ax₁² − bx₂² + abx₃² has no zero other than x = 0 (over ℚ).

**Theorem 5.2.5**: If b is prime and a is a quadratic non-residue mod b, then
A is a division algebra.

For the cases Katok constructs — where b is always chosen prime and a chosen to be
QNR mod b — Theorem 5.2.5 is both sufficient **and** necessary (it is an iff for prime b),
so no further machinery is needed for those examples.

## When Theorem 5.2.5 is not enough

If b is composite, Theorem 5.2.5 gives no information. The 4-variable zero-divisor
search used as a fallback is a heuristic: it can fail to find a zero divisor that
exists but has large coordinates.

**Concrete failure**: A = (97, 6/ℚ). Here b = 6 is not prime, so Theorem 5.2.5
does not apply. A search over |xᵢ| ≤ 8 finds no zero divisors and wrongly reports
"likely a division algebra." In fact A splits (is not a division algebra); the
smallest zero divisor is (x₀,x₁,x₂,x₃) = (10, 2, 7, 1), with max|xᵢ| = 10.

## The exact criterion via the Hilbert symbol

### Step 1 — reduce the norm form (Theorem 5.2.3)

Factor Nrd(x):

```
Nrd(x) = (x₀² − ax₁²) − b · (x₂² − ax₃²)
```

Setting Nrd = 0 with (x₂, x₃) ≠ (0, 0):

```
b = (x₀² − ax₁²) / (x₂² − ax₃²) = N_K(x₀ + x₁√a) / N_K(x₂ + x₃√a)
```

where N_K denotes the norm in K = ℚ(√a). This ratio is itself a norm in K. So:

> A is **not** a division algebra iff (a is a perfect square) **or**
> (b is a norm in the extension ℚ(√a)/ℚ).

"b is a norm in ℚ(√a)/ℚ" means the equation x² − ay² = b has a rational solution.

### Step 2 — reduce to local conditions (Hasse norm theorem)

For a quadratic extension K = ℚ(√a), the **Hasse norm theorem** states:

> b is a global norm N_{K/ℚ}(α) for some α ∈ K* **if and only if** b is a local
> norm N_{K_p/ℚ_p}(αₚ) at **every** prime p (and over ℝ).

The direction "global → local" is immediate. The direction "local everywhere → global"
is the content of the theorem; its proof goes through the product formula for Hilbert
symbols, which is equivalent to quadratic reciprocity.

For a, b > 0 the real condition is always satisfied (the form x² − ay² is indefinite
over ℝ). So only finitely many primes matter.

**Key fact**: at every prime p not dividing 2ab, b is automatically a local norm.
Only the primes dividing 2ab need checking.

### Step 3 — the Hilbert symbol formula

The **Hilbert symbol** (a, b)_p encodes whether b is a local norm from ℚ_p(√a)/ℚ_p:

```
(a, b)_p = +1   iff   b is a local norm at p
(a, b)_p = −1   iff   b is NOT a local norm at p
```

**For an odd prime p**, write a = pʳ · u and b = pˢ · v with u, v not divisible by p:

```
(a, b)_p = Legendre(  (−1)^{rs} · u^s · v^r ,  p  )
```

where Legendre(·, p) is the quadratic residue symbol mod p, computed by Euler's
criterion: Legendre(n, p) = n^{(p−1)/2} mod p, returning +1 or −1.

**For p = 2**, write a = 2^α · u and b = 2^β · v with u, v odd. Define:

```
ε(x) = 1 if x ≡ 3 (mod 4), else 0
ω(x) = (x² − 1)/8 mod 2       [= 1 if x ≡ ±3 (mod 8), else 0]
```

Then:

```
(a, b)_2 = (−1)^{ ε(u)·ε(v) + α·ω(v) + β·ω(u) }
```

### Step 4 — the decision

> A = (a, b/ℚ) is a **division algebra** iff (a, b)_p = −1 for at least one prime p.

(If no symbol is −1, A ≅ M(2, ℚ) and is not a division algebra.)

The **product formula** ∏_p (a,b)_p = 1 (for a, b > 0) guarantees the number of
primes where the symbol equals −1 is always even: either 0 (A splits) or ≥ 2
(A is a division algebra, ramified at those primes).

## Example: A = (97, 6/ℚ)

Primes dividing 2 · 97 · 6 = 1164: p = 2, 3, 97.

**p = 3**: r = v₃(97) = 0, s = v₃(6) = 1, u = 97 ≡ 1 (mod 3), v = 2.
Formula: Legendre((−1)^0 · 1^1 · 2^0, 3) = Legendre(1, 3) = **+1**.

**p = 97**: r = v₉₇(97) = 1, s = v₉₇(6) = 0, u = 1, v = 6.
Formula: Legendre((−1)^0 · 1^0 · 6^1, 97) = Legendre(6, 97).
6 = 2 · 3; Legendre(2, 97) = 1 (since 97 ≡ 1 mod 8); Legendre(3, 97) = Legendre(97, 3) = Legendre(1, 3) = 1.
So Legendre(6, 97) = **+1**.

**p = 2**: α = v₂(97) = 0, β = v₂(6) = 1, u = 97, v = 3.
ε(97) = 0 (97 ≡ 1 mod 4), ε(3) = 1 (3 ≡ 3 mod 4), ω(97) = 0 (97 ≡ 1 mod 8).
Formula: (−1)^{0·1 + 0·ω(3) + 1·0} = (−1)^0 = **+1**.

All symbols +1 → A = (97, 6/ℚ) **splits** (not a division algebra).
Zero divisor: (10, 2, 7, 1) with Nrd = 100 − 388 − 294 + 582 = 0. ✓

## Why the visualizer restricts to prime b

For prime b, Theorem 5.2.5 is an exact iff:

- a QNR mod b  →  (a, b)_b = −1  →  A is a division algebra
- a QR mod b   →  (a, b)_b = +1 and no other prime can give −1 (for Katok's typical choices)

This keeps the implementation within §5.2 of the book. For composite b the exact
check requires the Hilbert symbol formulas above.
