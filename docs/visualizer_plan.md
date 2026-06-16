# Hyperbolic Geometry Visualizer — Plan

## Goal

An interactive Python tool that displays the upper half-plane ℌ and the Poincaré disk 𝔻 side by side. The user clicks two points in either panel; the hyperbolic geodesic through them appears in both panels simultaneously, linked via the Cayley map.

---

## Stack

| Concern | Choice |
|---------|--------|
| Language | Python 3 |
| Plotting | `matplotlib` |
| UI controls | `matplotlib.widgets` (Button) |
| Numerics | `numpy` |
| Build | None — single file, `pip install matplotlib numpy` |

---

## File layout

```
fuchsian_visualizer/
  visualizer.py        ← entire app (~200 lines)
  test_math.py         ← pure-math unit tests (no display)
  visualizer_plan.md   ← this file
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  UI LAYER                                           │
│  matplotlib Figure                                  │
│    ax_H  (upper half-plane)  │  ax_D (Poincaré disk)│
│  Button: "Clear"                                    │
└───────────────────┬─────────────────────────────────┘
                    │ click events
┌───────────────────▼─────────────────────────────────┐
│  APP LAYER — HyperbolicVisualizer                   │
│  state: points_buffer[], geodesics[]                │
│  on_click(event) → collect 2 pts → draw both panels │
└──────┬──────────────────┬───────────────────────────┘
       │                  │
┌──────▼──────┐  ┌────────▼──────────┐
│ MATH LAYER  │  │  MATH LAYER       │
│ cayley(z)   │  │ geodesic_H(z1,z2) │
│ cayley_inv  │  │   → (center, r)   │
│  ℌ ↔ 𝔻     │  │   or vertical line│
└──────┬──────┘  └────────┬──────────┘
       │                  │
┌──────▼──────────────────▼──────────┐
│  RENDER LAYER                      │
│  draw_H(ax, arc_pts)               │
│  draw_D(ax, arc_pts mapped via f)  │
└────────────────────────────────────┘
```

---

## Math

### Models

**Upper half-plane** ℌ = { z ∈ ℂ | Im(z) > 0 }  
Metric: ds = √(dx² + dy²) / y

**Poincaré disk** 𝔻 = { w ∈ ℂ | |w| < 1 }  
Metric: ds = 2|dw| / (1 − |w|²)

### Cayley map  ℌ → 𝔻

```
f(z)   = (z − i) / (z + i)
f⁻¹(w) = i(1 + w) / (1 − w)
```

Key values: f(i) = 0,  f(i+1) = (1+i−i)/(1+i+i) = 1/(1+2i),  f(∞) = 1.

### Geodesics in ℌ

Given z₁, z₂ ∈ ℌ:

1. **Vertical line** — if |Re(z₁) − Re(z₂)| < ε:  
   x = Re(z₁),  y ∈ [0, y_max]

2. **Semicircle** — otherwise:  
   Center on ℝ:  c = (|z₂|² − |z₁|²) / (2 (Re(z₂) − Re(z₁)))  
   Radius:        r = |z₁ − c|  
   Arc:           z(θ) = c + r·e^{iθ},  θ ∈ [θ₁, θ₂] ⊂ (0, π)  
   where θⱼ = arg(zⱼ − c)

### Geodesics in 𝔻

No separate formula needed.  
Parameterize the geodesic in ℌ as above → sample N points → apply f(z) to each.  
The resulting curve is automatically a geodesic arc in 𝔻 (Cayley is an isometry).

---

## Interaction

| Action | Effect |
|--------|--------|
| Click in ax_H | Records z = x+iy (ignores clicks with y≤0) |
| Click in ax_D | Converts w → z via f⁻¹, records z |
| 2nd click | Computes geodesic; draws on both panels; resets buffer |
| "Clear" button | Removes all geodesics, resets state |

Geodesics are drawn in cycling colors so multiple geodesics are distinguishable.

---

## Test Cases

Tests live in `test_math.py` and use only `numpy` + `pytest` (no display). All functions under test are imported from `visualizer.py` with the matplotlib setup guarded by `if __name__ == "__main__"`.

### 1. Cayley map — known values

```python
# f(i) = 0
assert abs(cayley(1j)) < 1e-12

# f(2i) = 1/3
assert abs(cayley(2j) - 1/3) < 1e-12

# f(-1+i) = (-1+i-i)/(-1+i+i) = -1/(-1+2i) = (-1)(−1−2i)/5 = (1+2i)/5
assert abs(cayley(-1+1j) - (1+2j)/5) < 1e-12
```

### 2. Cayley inverse — round-trip

```python
test_points = [1j, 2+3j, -1+0.5j, 0.1+10j]
for z in test_points:
    assert abs(cayley_inv(cayley(z)) - z) < 1e-10
```

### 3. Cayley maps ℌ into 𝔻

```python
# |f(z)| < 1 for all z ∈ ℌ
for z in [1j, 2+3j, 0.001+0.001j, 0+100j]:
    assert abs(cayley(z)) < 1 - 1e-12
```

### 4. Geodesic center and radius

```python
# z1=1+i, z2=3+i
# c = (|3+i|²-|1+i|²)/(2*(3-1)) = (10-2)/4 = 2
# r = |1+i - 2| = |-1+i| = sqrt(2)
c, r = geodesic_H_params(1+1j, 3+1j)
assert abs(c - 2) < 1e-12
assert abs(r - 2**0.5) < 1e-12
```

### 5. Geodesic passes through both input points

```python
for z1, z2 in [(1+1j, 3+2j), (0.5+2j, -0.5+0.7j), (2+0.1j, -3+4j)]:
    c, r = geodesic_H_params(z1, z2)
    assert abs(abs(z1 - c) - r) < 1e-10
    assert abs(abs(z2 - c) - r) < 1e-10
```

### 6. Vertical-line detection

```python
# Same real part → vertical line
result = geodesic_type(2+1j, 2+3j)
assert result == "vertical"

# Different real part → semicircle
result = geodesic_type(1+1j, 3+1j)
assert result == "semicircle"
```

### 7. Geodesic symmetry (order independence)

```python
for z1, z2 in [(1+1j, 3+2j), (0.1+0.5j, -2+3j)]:
    c1, r1 = geodesic_H_params(z1, z2)
    c2, r2 = geodesic_H_params(z2, z1)
    assert abs(c1 - c2) < 1e-12
    assert abs(r1 - r2) < 1e-12
```

### 8. Hyperbolic distance formula

From the book (§1.2):  ρ(z, w) = ln( (|z−w̄| + |z−w|) / (|z−w̄| − |z−w|) )

```python
def hyp_dist(z, w):
    num = abs(z - w.conjugate()) + abs(z - w)
    den = abs(z - w.conjugate()) - abs(z - w)
    return np.log(num / den)

# ρ(i, 2i): |i - (-2i)| = 3,  |i - 2i| = 1  →  ln(4/2) = ln 2
assert abs(hyp_dist(1j, 2j) - np.log(2)) < 1e-12

# Symmetry
assert abs(hyp_dist(1+1j, 3+2j) - hyp_dist(3+2j, 1+1j)) < 1e-12
```

### 9. Geodesic arc stays in ℌ

```python
# All sampled points on the geodesic arc should have Im(z) > 0
pts = sample_geodesic_H(1+1j, 3+2j, n=200)
assert all(z.imag > 0 for z in pts)
```

### 10. Disk arc stays inside 𝔻

```python
# After Cayley transform, all points should satisfy |w| < 1
pts_H = sample_geodesic_H(1+1j, 3+2j, n=200)
pts_D = [cayley(z) for z in pts_H]
assert all(abs(w) < 1 for w in pts_D)
```

---

## Running tests

```bash
pip install pytest numpy matplotlib
pytest test_math.py -v
```

Expected: all 10 test groups pass with no display window opened.

---

## v1 scope (what's excluded)

- No PSL(2,ℝ) action / transformation animation (v2)
- No fundamental domain drawing (v2)
- No geodesic distance readout (v2)
- No save/export (v2)
