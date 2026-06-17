import math as _math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Callable, List

Y_MAX = 200  # truncation height for vertical geodesic rays

# ── Hyperbolic geometry ───────────────────────────────────────────────────────

def cayley(z):
    """Cayley map ℌ → 𝔻: f(z) = (z − i) / (z + i)"""
    return (z - 1j) / (z + 1j)

def cayley_inv(w):
    """Inverse Cayley map 𝔻 → ℌ: f⁻¹(w) = i(1 + w) / (1 − w)"""
    return 1j * (1 + w) / (1 - w)

def geodesic_type(z1, z2, eps=1e-9):
    """Returns 'vertical' if Re(z1) ≈ Re(z2), else 'semicircle'."""
    return "vertical" if abs(z1.real - z2.real) < eps else "semicircle"

def geodesic_H_params(z1, z2):
    """Center on ℝ and radius of the semicircle geodesic through z1, z2 in ℌ."""
    c = (abs(z2)**2 - abs(z1)**2) / (2 * (z2.real - z1.real))
    r = abs(z1 - c)
    return c, r

def sample_geodesic_H(z1, z2, n=400):
    """Sample n points on the geodesic arc through z1, z2 in ℌ."""
    if geodesic_type(z1, z2) == "vertical":
        x = z1.real
        y_min = min(z1.imag, z2.imag)
        y_max = max(z1.imag, z2.imag)
        ys = np.linspace(y_min, y_max, n)
        return (x + 1j * ys).astype(complex)
    else:
        c, r = geodesic_H_params(z1, z2)
        theta1 = np.angle(z1 - c)
        theta2 = np.angle(z2 - c)
        if theta1 > theta2:
            theta1, theta2 = theta2, theta1
        thetas = np.linspace(theta1, theta2, n)
        return (c + r * np.exp(1j * thetas)).astype(complex)

def mobius(M, z):
    """Apply 2×2 matrix M = [[a,b],[c,d]] to z as (az+b)/(cz+d)."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return (a * z + b) / (c * z + d)

def hyp_dist(z, w):
    """Hyperbolic distance in ℌ: acosh(1 + |z−w|²/(2·Im z·Im w))."""
    return np.arccosh(1 + abs(z - w)**2 / (2 * z.imag * w.imag))

def _mat_key(M):
    """Canonical key for a PSL(2,R) element (identify M and −M)."""
    flat = np.round(M.flatten().astype(float), 8)
    if flat[0] < -1e-9 or (abs(flat[0]) < 1e-9 and flat[1] < -1e-9):
        flat = -flat
    return tuple(flat)

# ── Group enumeration ─────────────────────────────────────────────────────────

def gen_psl2z(depth):
    """BFS enumerate PSL(2,ℤ) elements up to given word-length depth."""
    S  = np.array([[ 0, -1], [1,  0]], dtype=int)
    T  = np.array([[ 1,  1], [0,  1]], dtype=int)
    Ti = np.array([[ 1, -1], [0,  1]], dtype=int)
    I = np.eye(2, dtype=int)
    seen = {_mat_key(I)}
    queue = deque([(I, 0)])
    result = [I]
    while queue:
        M, d = queue.popleft()
        if d >= depth:
            continue
        for G in [S, T, Ti]:
            MG = M @ G
            key = _mat_key(MG)
            if key not in seen:
                seen.add(key)
                result.append(MG)
                queue.append((MG, d + 1))
    return result

def gen_dilation(lam, depth):
    """Enumerate ⟨z ↦ λz⟩: returns [I, g, g², …, gᵈ, g⁻¹, …, g⁻ᵈ]."""
    s = lam ** 0.5
    gp = np.array([[s,    0.], [0., 1./s]])
    gm = np.array([[1./s, 0.], [0., s   ]])
    result = [np.eye(2)]
    Mp, Mm = np.eye(2), np.eye(2)
    for _ in range(depth):
        Mp = Mp @ gp;  result.append(Mp.copy())
        Mm = Mm @ gm;  result.append(Mm.copy())
    return result

def gen_translation(t, depth):
    """Enumerate ⟨z ↦ z+t⟩: returns [I, T, T², …, Tᵈ, T⁻¹, …, T⁻ᵈ]."""
    Tp = np.array([[1.,  t], [0., 1.]])
    Tm = np.array([[1., -t], [0., 1.]])
    result = [np.eye(2)]
    Mp, Mm = np.eye(2), np.eye(2)
    for _ in range(depth):
        Mp = Mp @ Tp;  result.append(Mp.copy())
        Mm = Mm @ Tm;  result.append(Mm.copy())
    return result

def gen_bfs(generators, depth):
    """BFS enumerate a Fuchsian group from real 2×2 generator matrices."""
    I = np.eye(2)
    seen = {_mat_key(I)}
    queue = deque([(I.copy(), 0)])
    result = [I.copy()]
    while queue:
        M, d = queue.popleft()
        if d >= depth:
            continue
        for G in generators:
            MG = M @ G
            k = _mat_key(MG)
            if k not in seen:
                seen.add(k)
                result.append(MG.copy())
                queue.append((MG.copy(), d + 1))
    return result

# ── Element classification ────────────────────────────────────────────────────

def element_type(M):
    """Classify a PSL(2,R) element by |trace|: 'elliptic', 'parabolic', or 'hyperbolic'."""
    t = abs(float(np.trace(M)))
    if t < 2 - 1e-9:
        return 'elliptic'
    if t < 2 + 1e-9:
        return 'parabolic'
    return 'hyperbolic'

def fixed_points(M):
    """Return fixed point(s) of M as list of complex numbers.
    Parabolic with c=0: returns [inf].
    Elliptic: returns the point inside ℌ (Im > 0) first.
    Hyperbolic: returns two real points."""
    a, b, c, d = float(M[0,0]), float(M[0,1]), float(M[1,0]), float(M[1,1])
    if abs(c) < 1e-12:
        return [np.inf]
    disc = complex((a - d)**2 + 4 * b * c)
    sq = np.sqrt(disc)
    z1 = ((a - d) + sq) / (2 * c)
    z2 = ((a - d) - sq) / (2 * c)
    if z1.imag < z2.imag:
        z1, z2 = z2, z1
    return [z1, z2]

# ── Geodesic domain geometry ──────────────────────────────────────────────────

_SQRT3_2 = np.sqrt(3) / 2

def std_fd_arcs():
    """Return the 3 boundary arc endpoint-pairs of the standard PSL(2,ℤ) domain F."""
    left  = complex(-0.5, _SQRT3_2)
    right = complex( 0.5, _SQRT3_2)
    return [
        (left,  complex(-0.5, Y_MAX)),
        (right, complex( 0.5, Y_MAX)),
        (left,  right),
    ]

def bisector_params(z1, z2, eps=1e-9):
    """Return (center_x, radius) of the perpendicular bisector geodesic of z1, z2.
    Returns ('vertical', midpoint_x) if Im(z1) ≈ Im(z2)."""
    y1, y2 = z1.imag, z2.imag
    x1, x2 = z1.real, z2.real
    if abs(y2 - y1) < eps:
        return ('vertical', (x1 + x2) / 2)
    cx = (y2 * x1 - y1 * x2) / (y2 - y1)
    r = np.sqrt(cx**2 - (y2 * abs(z1)**2 - y1 * abs(z2)**2) / (y2 - y1))
    return (cx, r)

def sample_bisector(z1, z2, n=400):
    """Sample n points on the perpendicular bisector geodesic of z1 and z2 in ℌ."""
    params = bisector_params(z1, z2)
    if params[0] == 'vertical':
        x = params[1]
        ys = np.linspace(1e-6, Y_MAX, n)
        return (x + 1j * ys).astype(complex)
    cx, r = params
    thetas = np.linspace(1e-6, np.pi - 1e-6, n)
    return (cx + r * np.exp(1j * thetas)).astype(complex)

def _bisector_intersect(p1, p2):
    """Intersection in ℌ of two bisector geodesics (each from bisector_params).
    Returns a complex point or None if they don't intersect in ℌ."""
    v1 = (p1[0] == 'vertical')
    v2 = (p2[0] == 'vertical')
    if v1 and v2:
        return None
    if v1:
        xv = p1[1]; cx, r = p2
        disc = r**2 - (xv - cx)**2
        if disc <= 0: return None
        y = _math.sqrt(disc)
        return xv + 1j * y if y > 1e-10 else None
    if v2:
        xv = p2[1]; cx, r = p1
        disc = r**2 - (xv - cx)**2
        if disc <= 0: return None
        y = _math.sqrt(disc)
        return xv + 1j * y if y > 1e-10 else None
    cx1, r1 = p1; cx2, r2 = p2
    denom = 2.0 * (cx2 - cx1)
    if abs(denom) < 1e-12: return None
    x = (r1**2 - r2**2 - cx1**2 + cx2**2) / denom
    disc = r1**2 - (x - cx1)**2
    if disc <= 0: return None
    y = _math.sqrt(disc)
    return x + 1j * y if y > 1e-10 else None

def _bisector_param(p, z):
    """Arc parameter of point z on bisector p: angle θ ∈ (0,π) or height y."""
    if p[0] == 'vertical':
        return z.imag
    return np.angle(z - p[0])  # p[0] is cx for semicircles

def _sample_bisector_range(p, t_lo, t_hi, n=300):
    """Sample n points on bisector p between parameters t_lo and t_hi."""
    if p[0] == 'vertical':
        ys = np.linspace(t_lo, t_hi, n)
        return p[1] + 1j * ys
    cx, r = p
    thetas = np.linspace(t_lo, t_hi, n)
    return cx + r * np.exp(1j * thetas)

def _eval_bisector(p, t):
    """Evaluate bisector p at parameter t."""
    if p[0] == 'vertical':
        return p[1] + 1j * t
    cx, r = p
    return cx + r * np.exp(1j * t)

def dirichlet_boundary(z0, elements, eps=1e-8):
    """Return list of arc-point arrays forming the Dirichlet domain boundary of z0.

    Uses exact bisector intersection to find domain vertices, so adjacent arcs
    meet at the same point with no gap."""
    images = []
    for M in elements:
        w = mobius(M, z0)
        if w.imag > 1e-9 and abs(w - z0) > 1e-6:
            images.append(w)
    if not images:
        return []

    # Deduplicate orbit images
    unique = []
    for w in images:
        if not any(abs(w - v) < 1e-7 for v in unique):
            unique.append(w)
    images = unique

    bisectors = [bisector_params(z0, w) for w in images]

    boundary = []
    for i, (w, p) in enumerate(zip(images, bisectors)):
        if p[0] == 'vertical':
            t_lo_full, t_hi_full = 1e-6, float(Y_MAX)
        else:
            t_lo_full, t_hi_full = 1e-6, np.pi - 1e-6

        # Candidate split parameters: endpoints + intersections with every other bisector
        candidates = [t_lo_full, t_hi_full]
        for j, q in enumerate(bisectors):
            if j == i:
                continue
            pt = _bisector_intersect(p, q)
            if pt is None:
                continue
            t = _bisector_param(p, pt)
            if t_lo_full < t < t_hi_full:
                candidates.append(float(t))
        candidates = sorted(set(candidates))

        # Check each sub-interval's midpoint against the Dirichlet condition
        valid_intervals = []
        for k in range(len(candidates) - 1):
            t_mid = (candidates[k] + candidates[k + 1]) / 2.0
            z_mid = _eval_bisector(p, t_mid)
            d0 = hyp_dist(z_mid, z0)
            inside = all(
                hyp_dist(z_mid, v) >= d0 - eps
                for v in images if abs(v - w) > 1e-7
            )
            if inside:
                valid_intervals.append((candidates[k], candidates[k + 1]))

        if not valid_intervals:
            continue

        # Merge contiguous intervals and sample each
        merged = [list(valid_intervals[0])]
        for lo, hi in valid_intervals[1:]:
            if abs(lo - merged[-1][1]) < 1e-10:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])

        for t_lo, t_hi in merged:
            seg = _sample_bisector_range(p, t_lo, t_hi, n=300)
            boundary.append(seg)

    return boundary

# ── Quaternion algebra (Katok §5.2) ──────────────────────────────────────────

def quaternion_norm(x0, x1, x2, x3, a, b):
    """Reduced norm Nrd(x) = x0²−a·x1²−b·x2²+ab·x3²  (= det of ρ₁(x))."""
    return x0**2 - a*x1**2 - b*x2**2 + a*b*x3**2

def quaternion_embed(x0, x1, x2, x3, a, b):
    """Katok eq. 5.2.4: embed x = x0+x1·i+x2·j+x3·k into M(2,R).  Requires a > 0."""
    sa = np.sqrt(float(a))
    return np.array([[x0 + x1*sa,       x2 + x3*sa],
                     [b*(x2 - x3*sa),   x0 - x1*sa]], dtype=float)

def gen_quaternion(a, b, coord_bound=4):
    """Enumerate Γ(A,O) for A=(a,b/Q), standard integer order.
    Returns all ρ₁(x) with (x₀,x₁,x₂,x₃)∈[-R,R]⁴ and Nrd(x)=1, sign-normalised.
    Uses numpy broadcasting to find norm-1 coordinates without Python loops."""
    r = np.arange(-coord_bound, coord_bound + 1)
    X0, X1, X2, X3 = np.meshgrid(r, r, r, r, indexing='ij')
    norms = X0**2 - a*X1**2 - b*X2**2 + a*b*X3**2
    mask = (norms == 1)
    coords = np.stack([X0[mask], X1[mask], X2[mask], X3[mask]], axis=1)

    elements, seen = [], set()
    for x0, x1, x2, x3 in coords:
        M = quaternion_embed(int(x0), int(x1), int(x2), int(x3), a, b)
        flat = M.flatten()
        if flat[0] < -1e-9 or (abs(flat[0]) < 1e-9 and flat[1] < -1e-9):
            M = -M
        k = _mat_key(M)
        if k not in seen:
            seen.add(k)
            elements.append(M)
    return elements

def _is_prime(n):
    """Deterministic Miller-Rabin primality test, exact for n < 3_215_031_751."""
    n = abs(int(n))
    if n < 2: return False
    if n in (2, 3, 5, 7): return True
    if n % 2 == 0 or n % 3 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in (2, 3, 5, 7):
        if a >= n: continue
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else:
            return False
    return True

def _is_qnr_mod_p(a, p):
    """True iff a is a quadratic non-residue mod prime p > 2 (Euler's criterion)."""
    a_mod = a % p
    if a_mod == 0:
        return False
    return pow(int(a_mod), (p - 1) // 2, p) == p - 1

def check_division_algebra(a, b):
    """Return (is_division: bool, message: str) for A=(a,b/Q), a,b positive ints.

    Exact — no heuristics.  Only accepts inputs covered by Theorem 5.2.5:
      b prime, a not a perfect square, a QNR mod b  ↔  A is a division algebra.
    All other inputs are rejected with an explanatory message.
    """
    a, b = int(a), int(b)
    t = int(_math.isqrt(a))
    if t * t == a:
        return False, (f'a={a}={t}² is a perfect square → A≅M(2,ℚ), not a division algebra '
                       f'(Theorem 5.2.1)')
    if not _is_prime(b):
        return False, (f'b={b} is not prime.  Theorem 5.2.5 requires b to be prime.  '
                       f'Try a prime value of b.')
    if not _is_qnr_mod_p(a, b):
        return False, (f'a={a} is a quadratic residue mod b={b}.  '
                       f'Theorem 5.2.5 does not apply; choose a with a QNR mod b.  '
                       f'(Note: even for prime b, a QR mod b may still give a division '
                       f'algebra via other primes, but this cannot be confirmed within §5.2.)')
    return True, (f'Theorem 5.2.5: b={b} prime, a={a} is QNR mod {b} '
                  f'→ A is a division algebra; Γ\\ℌ is compact (Thm 5.4.1)')

# ── Group presets ─────────────────────────────────────────────────────────────

@dataclass
class GroupPreset:
    label: str
    group_desc: str
    fd_desc: str
    enum_func: Callable
    canonical_z0: complex
    use_std_fd: bool = False

_S   = np.array([[ 0., -1.], [1.,  0.]])
_T2  = np.array([[ 1.,  2.], [0.,  1.]])
_Ti2 = np.array([[ 1., -2.], [0.,  1.]])

GROUP_PRESETS: List[GroupPreset] = [
    GroupPreset(
        label='PSL(2,ℤ)',
        group_desc='Modular group: generated by S (z↦−1/z) and T (z↦z+1).',
        fd_desc='F = { z ∈ ℌ : |Re z| ≤ ½,  |z| ≥ 1 }  (geodesic triangle)',
        enum_func=gen_psl2z,
        canonical_z0=2j,
        use_std_fd=True,
    ),
    GroupPreset(
        label='⟨z↦2z⟩',
        group_desc='Cyclic hyperbolic group generated by z ↦ 2z.',
        fd_desc='Half-annulus  { 1/√2 ≤ |z| ≤ √2 }  in ℌ.',
        enum_func=lambda d: gen_dilation(2, d),
        canonical_z0=1j,
    ),
    GroupPreset(
        label='⟨z↦z+1⟩',
        group_desc='Cyclic parabolic group generated by z ↦ z+1.',
        fd_desc='Vertical strip  { |Re z| ≤ ½ }  in ℌ.',
        enum_func=lambda d: gen_translation(1, d),
        canonical_z0=0.5+1j,
    ),
    GroupPreset(
        label='⟨S, T²⟩',
        group_desc='Theta group: S (z↦−1/z) and T² (z↦z+2).  Index 3 in PSL(2,ℤ).',
        fd_desc='Geodesic polygon with 3× the hyperbolic area of the PSL(2,ℤ) domain.',
        enum_func=lambda d: gen_bfs([_S, _T2, _Ti2], d),
        canonical_z0=2j,
    ),
    GroupPreset(
        label='Γ(3,5/ℚ)',
        group_desc='Quaternion group A=(3,5/ℚ), standard integer order.  Norm: x₀²−3x₁²−5x₂²+15x₃²=1.',
        fd_desc='Compact FD (Thm 5.4.1): no cusps.  All non-identity elements hyperbolic.  3 is QNR mod 5.',
        enum_func=lambda d: gen_quaternion(3, 5, d + 2),
        canonical_z0=2j,
    ),
    GroupPreset(
        label='Γ(2,3/ℚ)',
        group_desc='Quaternion group A=(2,3/ℚ), standard integer order.  Norm: x₀²−2x₁²−3x₂²+6x₃²=1.',
        fd_desc='Compact FD (Thm 5.4.1): no cusps.  All non-identity elements hyperbolic.  2 is QNR mod 3.',
        enum_func=lambda d: gen_quaternion(2, 3, d + 2),
        canonical_z0=2j,
    ),
]
