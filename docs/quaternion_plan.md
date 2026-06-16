# Fuchsian Groups from Quaternion Algebras — Plan

## Mathematical Background (Katok §5.2)

### Quaternion algebra A = (a, b/ℚ)

A 4-dimensional ℚ-algebra with basis {1, i, j, k} and multiplication:

```
i² = a,   j² = b,   k = ij = −ji
```

An element x = x₀ + x₁i + x₂j + x₃k has:
- **Reduced norm**: Nrd(x) = x₀² − ax₁² − bx₂² + abx₃²
- **Reduced trace**: Trd(x) = 2x₀

### Embedding into M(2, ℝ)  [Katok eq. 5.2.4]

For a > 0, the map ρ₁: A → M(2, ℝ) defined by

```
ρ₁(x) = [[ x₀ + x₁√a,      x₂ + x₃√a  ],
           [ b(x₂ − x₃√a),  x₀ − x₁√a  ]]
```

satisfies **det(ρ₁(x)) = Nrd(x)** and **tr(ρ₁(x)) = 2x₀**.

### From quaternions to a Fuchsian group [Katok §5.2]

**Standard order**: O = { x | x₀, x₁, x₂, x₃ ∈ ℤ }

**Norm-1 units**: O¹ = { x ∈ O | Nrd(x) = 1 }
= solutions in ℤ⁴ of the **norm equation**:

```
x₀² − ax₁² − bx₂² + abx₃² = 1
```

**Fuchsian group** (Theorem 5.2.7):

```
Γ(A, O) = ρ₁(O¹) / {±I₂}  ⊂  PSL(2, ℝ)
```

ρ₁(O¹) is a discrete subgroup of SL(2, ℝ), hence Γ(A, O) is a Fuchsian group.

### Division algebra ↔ compact quotient

**Theorem 5.2.5** (Katok): If b is prime and a is a quadratic non-residue mod b
(i.e., x² ≡ a (mod b) has no integer solution), then A is a **division algebra**.

**Theorem 5.4.1** (Katok): If A is a division algebra over ℚ, then Γ(A, O)\ℌ is
**compact** — no cusps, no parabolic elements.

Consequence: for a, b > 0 integer with A a division algebra, every non-identity
element of Γ(A, O) is **hyperbolic** (|tr| > 2). The fundamental domain is
a compact hyperbolic polygon — completely different from PSL(2, ℤ).

---

## Concrete Examples

| Preset | a | b | Norm equation | Division alg? | Reason |
|--------|---|---|---------------|---------------|--------|
| Γ(3,5/ℚ) | 3 | 5 | x₀²−3x₁²−5x₂²+15x₃²=1 | ✓ | 3 is QNR mod 5 |
| Γ(2,3/ℚ) | 2 | 3 | x₀²−2x₁²−3x₂²+6x₃²=1 | ✓ | 2 is QNR mod 3 |

### Small generators of Γ(3,5/ℚ)  (coord search, |xᵢ| ≤ 3)

| (x₀, x₁, x₂, x₃) | Matrix ρ₁(x) | tr | type |
|--------------------|-------------|----|------|
| (2, 1, 0, 0)  | [[2+√3, 0], [0, 2−√3]]   | 4 | hyperbolic |
| (3, 1, 1, 0)  | [[3+√3, 1], [5, 3−√3]]   | 6 | hyperbolic |
| (3, −1, 1, 0) | [[3−√3, 1], [5, 3+√3]]   | 6 | hyperbolic |
| (3, 1, −1, 0) | [[3+√3, −1], [−5, 3−√3]] | 6 | hyperbolic |

### Small generators of Γ(2,3/ℚ)  (coord search, |xᵢ| ≤ 3)

| (x₀, x₁, x₂, x₃) | Matrix ρ₁(x) | tr | type |
|--------------------|-------------|----|------|
| (3, 2, 0, 0) | [[3+2√2, 0], [0, 3−2√2]] | 6 | hyperbolic |
| (2, 0, 1, 0) | [[2, 1], [3, 2]]          | 4 | hyperbolic |

---

## Implementation Plan

### New math functions  (visualizer.py)

```python
def quaternion_norm(x0, x1, x2, x3, a, b):
    """Reduced norm: x0²−a·x1²−b·x2²+ab·x3²  (= det of ρ₁(x))."""
    return x0**2 - a*x1**2 - b*x2**2 + a*b*x3**2


def quaternion_embed(x0, x1, x2, x3, a, b):
    """Katok eq. 5.2.4: map quaternion element to 2×2 real matrix.
    Requires a > 0 for real embedding."""
    sa = np.sqrt(float(a))
    return np.array([[x0 + x1*sa,       x2 + x3*sa],
                     [b*(x2 - x3*sa),   x0 - x1*sa]], dtype=float)


def gen_quaternion(a, b, coord_bound=4):
    """Enumerate Γ(A,O) for A=(a,b/ℚ) by exhaustive norm-equation search.

    Finds all (x₀,x₁,x₂,x₃) ∈ [−R,R]⁴ with Nrd(x)=1, maps each
    to a 2×2 matrix via ρ₁, deduplicates by _mat_key.
    O((2R+1)⁴) iterations: R=5 → ~14k, fast in Python.
    """
    elements, seen = [], set()
    for x0 in range(-coord_bound, coord_bound+1):
        for x1 in range(-coord_bound, coord_bound+1):
            for x2 in range(-coord_bound, coord_bound+1):
                for x3 in range(-coord_bound, coord_bound+1):
                    if quaternion_norm(x0, x1, x2, x3, a, b) == 1:
                        M = quaternion_embed(x0, x1, x2, x3, a, b)
                        k = _mat_key(M)
                        if k not in seen:
                            seen.add(k)
                            elements.append(M)
    return elements
```

The depth UI slider maps to `coord_bound = depth + 2` so that depth=1 already
finds the first generators (depth=1 → R=3 for Γ(3,5/ℚ)).

### Division algebra check  (visualizer.py)

Three tests are applied in order:

1. **Perfect-square test**: if a = t² for integer t, then A ≅ M(2,ℚ) (Theorem 5.2.1).
2. **Theorem 5.2.5**: if |b| is prime and a is a quadratic non-residue mod |b|, A is a
   division algebra. Uses Euler's criterion: a is QNR mod p iff a^((p−1)/2) ≡ −1 (mod p).
3. **Heuristic zero-divisor search**: enumerate |xᵢ| ≤ 8; if Nrd(x) = 0 for some x ≠ 0,
   A is not a division algebra.

```python
def _is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    return all(n % i for i in range(3, int(n**0.5) + 1, 2))


def _is_qnr_mod_p(a, p):
    """True iff a is a quadratic non-residue mod prime p > 2 (Euler's criterion)."""
    a_mod = a % p
    if a_mod == 0: return False
    return pow(int(a_mod), (p - 1) // 2, p) == p - 1


def check_division_algebra(a, b):
    """Return (is_division: bool, message: str) for A=(a,b/ℚ), a,b positive ints."""
    import math
    a, b = int(a), int(b)
    t = int(math.isqrt(a))
    if t * t == a:
        return False, f"a={a}={t}² is a perfect square → A≅M(2,ℚ), not a division algebra"
    if _is_prime(abs(b)) and _is_qnr_mod_p(a, abs(b)):
        return True, (f"Theorem 5.2.5: b={b} prime, a={a} QNR mod {b} "
                      f"→ division algebra; Γ\\ℌ is compact (Theorem 5.4.1)")
    R = 8
    for x0 in range(-R, R+1):
        for x1 in range(-R, R+1):
            for x2 in range(-R, R+1):
                for x3 in range(-R, R+1):
                    if x0==x1==x2==x3==0: continue
                    if quaternion_norm(x0, x1, x2, x3, a, b) == 0:
                        return False, (f"Zero divisor ({x0},{x1},{x2},{x3}) found "
                                       f"→ A≅M(2,ℚ), not a division algebra")
    return True, (f"No zero divisors found for |xᵢ|≤{R} "
                  f"→ likely a division algebra; Γ\\ℌ likely compact")
```

No new dependencies — `pow(a, e, m)` is built-in Python.

### Fixed presets  (starting examples)

Two presets remain in GROUP_PRESETS as convenient defaults:

```python
GroupPreset(
    label='Γ(3,5/ℚ)',
    group_desc='A=(3,5/ℚ), standard integer order.  '
               'Norm eq.: x₀²−3x₁²−5x₂²+15x₃²=1.  '
               '3 is QNR mod 5 → division algebra.',
    fd_desc='Compact FD (Thm 5.4.1): no cusps.  All elements hyperbolic.',
    enum_func=lambda d: gen_quaternion(3, 5, d + 2),
    canonical_z0=2j,
),
GroupPreset(
    label='Γ(2,3/ℚ)',
    group_desc='A=(2,3/ℚ), standard integer order.  '
               'Norm eq.: x₀²−2x₁²−3x₂²+6x₃²=1.  '
               '2 is QNR mod 3 → division algebra.',
    fd_desc='Compact FD (Thm 5.4.1): no cusps.  All elements hyperbolic.',
    enum_func=lambda d: gen_quaternion(2, 3, d + 2),
    canonical_z0=2j,
),
```

### Custom (a, b) input — UI

A small row of widgets placed below the Group radio buttons lets the user enter any
positive integers a, b and generate the corresponding quaternion group on the fly.

```
┌─────────────────────────────────────────────────────────────────┐
│  Mode          │ Group                │ Quaternion custom input  │
│  ○ Geodesics   │ ○ PSL(2,ℤ)          │  a: [__3__]  b: [__5__] │
│  ○ Fund. Dom.  │ ○ ⟨z↦2z⟩            │  [Generate Γ(a,b)]       │
│  ○ Dirichlet   │ ○ ⟨z↦z+1⟩           │                          │
│                │ ○ ⟨S, T²⟩           │                          │
│                │ ○ Γ(3,5/ℚ)  [preset]│                          │
│                │ ○ Γ(2,3/ℚ)  [preset]│                          │
└─────────────────────────────────────────────────────────────────┘
```

Widgets used: `matplotlib.widgets.TextBox` for a and b; `Button` for Generate.

`_on_generate_quaternion(event)`:
1. Parse a, b from text boxes as positive integers; show error in description if invalid.
2. Call `check_division_algebra(a, b)` → get `(is_div, msg)`.
3. Build a temporary `GroupPreset` with `enum_func=lambda d: gen_quaternion(a, b, d+2)`.
4. Set this as `self.current_group`; clear plots; redraw if mode is Fund. Domain.
5. Display in description panel:
   ```
   Custom quaternion group Γ(a,b/ℚ): a=<a>, b=<b>
   Norm equation: x₀²−<a>x₁²−<b>x₂²+<ab>x₃²=1
   <msg from check_division_algebra>
   ```
6. Deselect all group radio buttons (set active index to -1 or add a "Custom" dummy entry).

### New tests  (test_math.py)

```python
from visualizer import (quaternion_norm, quaternion_embed, gen_quaternion,
                        check_division_algebra, _mat_key, element_type)

def test_quaternion_norm_identity():
    assert quaternion_norm(1, 0, 0, 0, 3, 5) == 1

def test_quaternion_norm_known():
    assert quaternion_norm(2, 1, 0, 0, 3, 5) == 1   # 4−3 = 1
    assert quaternion_norm(3, 1, 1, 0, 3, 5) == 1   # 9−3−5 = 1

def test_quaternion_embed_det():
    for (x0,x1,x2,x3), a, b in [((2,1,0,0),3,5), ((3,1,1,0),3,5), ((2,0,1,0),2,3)]:
        M = quaternion_embed(x0,x1,x2,x3,a,b)
        assert abs(np.linalg.det(M) - quaternion_norm(x0,x1,x2,x3,a,b)) < 1e-10

def test_quaternion_embed_trace():
    for x0 in [1, 2, 3]:
        M = quaternion_embed(x0, 0, 0, 0, 3, 5)
        assert abs(np.trace(M) - 2*x0) < 1e-10

def test_gen_quaternion_contains_generators():
    elements = gen_quaternion(3, 5, 3)
    keys = {_mat_key(M) for M in elements}
    for x0,x1,x2,x3 in [(2,1,0,0),(3,1,1,0),(3,-1,1,0)]:
        M = quaternion_embed(x0,x1,x2,x3,3,5)
        assert _mat_key(M) in keys

def test_gen_quaternion_unit_det():
    for M in gen_quaternion(3, 5, 3):
        assert abs(np.linalg.det(M) - 1) < 1e-9

def test_gen_quaternion_all_hyperbolic():
    for M in gen_quaternion(3, 5, 4):
        assert element_type(M) in ('identity', 'hyperbolic')

# Division algebra check tests
def test_check_div_alg_perfect_square():
    is_div, msg = check_division_algebra(4, 5)   # a=4=2²
    assert not is_div

def test_check_div_alg_thm525():
    is_div, msg = check_division_algebra(3, 5)   # 3 QNR mod 5
    assert is_div
    is_div2, _ = check_division_algebra(2, 3)    # 2 QNR mod 3
    assert is_div2

def test_check_div_alg_not_division():
    # a=1 is always a perfect square → not division algebra
    is_div, _ = check_division_algebra(1, 7)
    assert not is_div
```

### Implementation order

1. Add `quaternion_norm`, `quaternion_embed`, `gen_quaternion`, `_is_prime`,
   `_is_qnr_mod_p`, `check_division_algebra` to `visualizer.py`
2. Run the 10 new tests (all should pass)
3. Add two `GroupPreset` entries to `GROUP_PRESETS`
4. Add `TextBox` (a, b) + `Button` (Generate) widgets; wire `_on_generate_quaternion`
5. Run all tests, launch visualizer, check tessellation and description text

### File changes

| File | Change |
|------|--------|
| `visualizer.py` | 6 new math functions; 2 new `GroupPreset` entries; 3 new widgets + handler |
| `test_math.py` | 10 new tests; updated imports |

No new dependencies.

---

## Key properties of the resulting groups

- **No parabolic elements**: Trd=±2 with Nrd=1 forces x₀=±1 and 3x₁²+5x₂²=15x₃²,
  which has no integer solutions ⟹ no cusps.
- **No elliptic elements**: Trd=0 requires −3x₁²−5x₂²+15x₃²=1,
  also unsolvable in ℤ for A=(3,5/ℚ).
- **Compact quotient**: Γ(A,O)\ℌ is a compact hyperbolic surface (Theorem 5.4.1).
- **Different from PSL(2,ℤ)**: entries involve √a — these are genuinely new groups,
  not commensurable with the modular group.
