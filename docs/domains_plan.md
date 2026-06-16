# Fundamental Domains & Dirichlet Domains — Plan

## Goal

Add two new visualization modes to the existing hyperbolic visualizer:

1. **Fundamental Domain** — draw the standard PSL(2,ℤ) fundamental domain F and its tessellation (N layers of Γ-tiles) in both panels.
2. **Dirichlet Domain** — for a user-chosen basepoint z₀, compute and draw the hyperbolic Voronoi cell of z₀ with respect to the orbit Γ·z₀.

Group: **PSL(2,ℤ)** only (generators S: z ↦ −1/z, T: z ↦ z+1).

---

## Math

### Standard fundamental domain F

```
F = { z ∈ ℌ : |Re z| ≤ ½,  |z| ≥ 1 }
```

Three boundary pieces (as pairs of endpoints for `sample_geodesic_H`):

| Arc | z_start | z_end | Type |
|-----|---------|-------|------|
| Left ray | −½ + i·(√3/2) | −½ + i·Y_MAX | vertical |
| Right ray | ½ + i·(√3/2) | ½ + i·Y_MAX | vertical |
| Unit-circle arc | −½ + i·(√3/2) | ½ + i·(√3/2) | semicircle (center 0, r=1) |

`Y_MAX = 20` — large enough that all tile boundaries fall within the visible window after Möbius transformation.

### Möbius transformation

```
mobius(M, z) = (a·z + b) / (c·z + d)    for M = [[a, b], [c, d]]
```

Möbius images of geodesics are geodesics, so each transformed arc (z_start, z_end) can be resampled with `sample_geodesic_H(mobius(γ, a), mobius(γ, b))`.

### Tessellation

```
for γ in gen_psl2z(depth):
    for (a, b) in std_fd_arcs():
        draw sample_geodesic_H(mobius(γ, a), mobius(γ, b))
```

Color tiles by BFS generation (depth from identity). Tiles at greater depth are drawn with lower opacity.

### Enumerating PSL(2,ℤ)

BFS from the identity matrix using generators {S, T, T⁻¹}:

```
S   = [[0, −1], [1,  0]]
T   = [[1,  1], [0,  1]]
T⁻¹ = [[1, −1], [0,  1]]
```

State: 2×2 integer matrix. Deduplication via `frozenset` of flattened entries (identify M and −M since PSL = SL/{±I}). Stop after reaching `depth` BFS levels.

### Hyperbolic distance

```
hyp_dist(z, w) = acosh(1 + |z − w|² / (2·Im(z)·Im(w)))
```

### Perpendicular bisector

The bisector of z₁, z₂ ∈ ℌ is the set of points equidistant from both:

```
|z − z₁|² / Im(z₁) = |z − z₂|² / Im(z₂)
```

Expanding: this is a Euclidean circle centered on ℝ (a geodesic), computed as:

```
c_x = (Im(z₂)·Re(z₁) − Im(z₁)·Re(z₂)) / (Im(z₂) − Im(z₁))
r   = distance from c_x to z₁
```

Special case Im(z₁) = Im(z₂): bisector is the vertical line x = (Re(z₁) + Re(z₂)) / 2.

`sample_bisector(z1, z2)` returns the bisector arc clipped to a reasonable y-range.

### Dirichlet domain (analytic, sieve method)

```
D(z₀) = { z ∈ ℌ : ρ(z, z₀) ≤ ρ(z, γ(z₀))  for all γ ∈ Γ }
```

Algorithm:

```
elements = gen_psl2z(depth)
images   = [mobius(γ, z₀) for γ in elements if γ ≠ I]

boundary_arcs = []
for w in images:
    pts = sample_bisector(z₀, w, n=400)
    # sieve: keep points closer to z₀ than to every other orbit image
    valid = [z for z in pts
             if all(hyp_dist(z, z₀) ≤ hyp_dist(z, v) + ε
                    for v in images)]
    boundary_arcs.append(valid)

draw all boundary_arcs
```

The sieve is vectorized with numpy (grid of bisector points × orbit images → distance matrix).

---

## UI

```
┌────────────────────────────────────────────────────────┐
│  [ax_H]                    │  [ax_D]                   │
│                            │                           │
├────────────────────────────────────────────────────────┤
│  Mode: ○ Geodesics  ○ Fund. Domain  ○ Dirichlet        │
│  [Clear]        Depth: [−] 2 [+]                       │
├────────────────────────────────────────────────────────┤
│  [ax_desc]  — text description panel                   │
└────────────────────────────────────────────────────────┘
```

| Mode | Click behavior |
|------|---------------|
| Geodesics | existing: 2 clicks → draw geodesic |
| Fund. Domain | no clicks; [−]/[+] buttons control tessellation depth |
| Dirichlet | 1 click → set z₀ → compute and draw Dirichlet domain |

Depth buttons are hidden in Geodesics and Dirichlet modes.

Implemented with `matplotlib.widgets.RadioButtons` and two `Button` widgets.

### Text description panel

A frameless `ax_desc` axes sits below the controls and holds a single `ax.text` call, updated on every state change. Content per mode:

**Geodesics mode (idle):**
```
Click two points in either panel to draw the hyperbolic geodesic through them.
Points clicked in the disk are mapped to ℌ via the inverse Cayley map f⁻¹(w) = i(1+w)/(1−w).
```

**Geodesics mode (after drawing, one geodesic selected):**
```
Geodesic through z₁ = <x₁> + <y₁>i,  z₂ = <x₂> + <y₂>i
Type: semicircle  |  center c = <c>,  radius r = <r>
Hyperbolic distance: ρ(z₁, z₂) = <d>
```
(or "Type: vertical line  |  x = <x>" for the vertical case)

**Fundamental Domain mode:**
```
Standard fundamental domain of PSL(2,ℤ):
  F = { z ∈ ℌ : |Re z| ≤ ½,  |z| ≥ 1 }
Bounded by: two vertical rays x = ±½ and the unit-circle arc |z| = 1.
Tessellation depth <N>: showing <k> tiles  γ(F),  γ ∈ PSL(2,ℤ).
Every point in ℌ is PSL(2,ℤ)-equivalent to exactly one interior point of F.
```

**Dirichlet mode (idle):**
```
Click a point z₀ in either panel to compute its Dirichlet domain.
D(z₀) = { z ∈ ℌ : ρ(z, z₀) ≤ ρ(z, γ(z₀))  for all γ ∈ PSL(2,ℤ) }
The boundary consists of hyperbolic perpendicular bisectors of segments z₀ → γ(z₀).
```

**Dirichlet mode (after click):**
```
Dirichlet domain centered at z₀ = <x₀> + <y₀>i
D(z₀) is a fundamental domain for PSL(2,ℤ): its PSL(2,ℤ)-translates tile ℌ without overlap.
Boundary arcs: <k> bisector segments shown  (depth <N>).
```

The description is rendered with `wrap=True` and a small fixed font size so it fits in one line per paragraph. It updates immediately whenever the mode changes or a new geodesic/domain is drawn.

---

## New tests (test_math.py)

| Test | What it checks |
|------|---------------|
| `test_mobius_identity` | `mobius(I, z) == z` for several z |
| `test_mobius_S` | S(i) = i (fixed point), S(2i) = i/2 |
| `test_mobius_composition` | `mobius(S @ T, z) == mobius(S, mobius(T, z))` |
| `test_hyp_dist_known` | ρ(i, 2i) = ln 2 (consistent with existing test 8) |
| `test_bisector_midpoint` | hyperbolic midpoint of z₁, z₂ lies on bisector |
| `test_bisector_equidistant` | two random bisector-sample points are equidistant from z₁, z₂ |
| `test_gen_psl2z_det` | all enumerated matrices have det = 1 (mod ±1) |
| `test_gen_psl2z_contains_generators` | S and T appear in the enumeration |
| `test_dirichlet_contains_z0` | z₀ satisfies the Dirichlet condition (inside its own domain) |

---

## File changes

| File | Change |
|------|--------|
| `visualizer.py` | New math functions; 3-mode UI |
| `test_math.py` | 9 new tests |

No new files. No new dependencies.

---

## Implementation order

1. `mobius`, `hyp_dist`, `gen_psl2z` + their tests
2. `std_fd_arcs`, tessellation drawing + `test_gen_psl2z_*`
3. `bisector_params`, `sample_bisector` + bisector tests
4. `dirichlet_boundary` (sieve) + `test_dirichlet_contains_z0`
5. UI: RadioButtons, depth controls, mode dispatch
6. `_update_description()`: text panel wired to every state change
