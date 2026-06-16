import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
from collections import deque
from dataclasses import dataclass
from typing import Callable, List

Y_MAX = 20  # truncation height for vertical geodesic rays

# ── Math layer ────────────────────────────────────────────────────────────────

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
    """Canonical key for a PSL(2,R) element (identify M and −M). Works for int and float."""
    flat = np.round(M.flatten().astype(float), 8)
    if flat[0] < -1e-9 or (abs(flat[0]) < 1e-9 and flat[1] < -1e-9):
        flat = -flat
    return tuple(flat)

def gen_psl2z(depth):
    """BFS enumerate PSL(2,ℤ) elements up to given word-length depth.
    Returns list of 2×2 integer numpy arrays."""
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
    """Enumerate ⟨z ↦ λz⟩: returns [I, g, g², …, gᵈ, g⁻¹, …, g⁻ᵈ].
    Generator g = [[√λ, 0], [0, 1/√λ]]."""
    s = lam ** 0.5
    gp = np.array([[s,   0.], [0., 1./s]])
    gm = np.array([[1./s, 0.], [0., s  ]])
    result = [np.eye(2)]
    Mp, Mm = np.eye(2), np.eye(2)
    for _ in range(depth):
        Mp = Mp @ gp;  result.append(Mp.copy())
        Mm = Mm @ gm;  result.append(Mm.copy())
    return result

def gen_translation(t, depth):
    """Enumerate ⟨z ↦ z+t⟩: returns [I, T, T², …, Tᵈ, T⁻¹, …, T⁻ᵈ].
    Generator T = [[1, t], [0, 1]]."""
    Tp = np.array([[1., t], [0., 1.]])
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
    # solve c·z² + (d−a)·z − b = 0
    disc = complex((a - d)**2 + 4 * b * c)
    sq = np.sqrt(disc)
    z1 = ((a - d) + sq) / (2 * c)
    z2 = ((a - d) - sq) / (2 * c)
    # for elliptic, put the ℌ-interior point first
    if z1.imag < z2.imag:
        z1, z2 = z2, z1
    return [z1, z2]

_SQRT3_2 = np.sqrt(3) / 2

def std_fd_arcs():
    """Return the 3 boundary arc endpoint-pairs of the standard PSL(2,ℤ) domain F.
    Each entry is (z_start, z_end) for sample_geodesic_H."""
    left  = complex(-0.5, _SQRT3_2)
    right = complex( 0.5, _SQRT3_2)
    return [
        (left,  complex(-0.5, Y_MAX)),  # left vertical ray  x = −½
        (right, complex( 0.5, Y_MAX)),  # right vertical ray x = +½
        (left,  right),                 # unit-circle arc
    ]

def bisector_params(z1, z2, eps=1e-9):
    """Return (center_x, radius) of the perpendicular bisector geodesic of z1, z2.
    Returns ('vertical', midpoint_x) if Im(z1) ≈ Im(z2)."""
    y1, y2 = z1.imag, z2.imag
    x1, x2 = z1.real, z2.real
    if abs(y2 - y1) < eps:
        return ('vertical', (x1 + x2) / 2)
    cx = (y2 * x1 - y1 * x2) / (y2 - y1)
    # r² = cx² − (y2|z1|² − y1|z2|²)/(y2−y1)  (from expanding the bisector equation)
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

def dirichlet_boundary(z0, elements, eps=1e-8):
    """Return list of arc-point arrays forming the Dirichlet domain boundary of z0.
    Sieve: for each orbit image w=γ(z0), keep bisector points closest to z0."""
    images = []
    for M in elements:
        w = mobius(M, z0)
        # skip identity and fixed points (w coincides with z0)
        if w.imag > 1e-9 and abs(w - z0) > 1e-6:
            images.append(w)

    if not images:
        return []

    boundary = []
    for w in images:
        pts = sample_bisector(z0, w, n=500)
        d0 = np.array([hyp_dist(p, z0) for p in pts])
        valid_mask = np.ones(len(pts), dtype=bool)
        for v in images:
            if abs(v - w) < 1e-9:
                continue
            dv = np.array([hyp_dist(p, v) for p in pts])
            valid_mask &= (d0 <= dv + eps)
        seg = pts[valid_mask]
        if len(seg) > 1:
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
    Returns all ρ₁(x) with (x₀,x₁,x₂,x₃)∈[-R,R]⁴ and Nrd(x)=1.
    Sign-normalised the same way as _mat_key so ±M map to the same element."""
    elements, seen = [], set()
    for x0 in range(-coord_bound, coord_bound + 1):
        for x1 in range(-coord_bound, coord_bound + 1):
            for x2 in range(-coord_bound, coord_bound + 1):
                for x3 in range(-coord_bound, coord_bound + 1):
                    if quaternion_norm(x0, x1, x2, x3, a, b) == 1:
                        M = quaternion_embed(x0, x1, x2, x3, a, b)
                        # Normalise sign: prefer positive leading entry (PSL convention)
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
    # write n-1 = 2^r * d
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
    import math
    a, b = int(a), int(b)
    # (1) a must not be a perfect square (Theorem 5.2.1)
    t = int(math.isqrt(a))
    if t * t == a:
        return False, (f'a={a}={t}² is a perfect square → A≅M(2,ℚ), not a division algebra '
                       f'(Theorem 5.2.1)')
    # (2) b must be prime — Theorem 5.2.5 requires it
    if not _is_prime(b):
        return False, (f'b={b} is not prime.  Theorem 5.2.5 requires b to be prime.  '
                       f'Try a prime value of b.')
    # (3) a must be a QNR mod b — Theorem 5.2.5 requires it
    if not _is_qnr_mod_p(a, b):
        return False, (f'a={a} is a quadratic residue mod b={b}.  '
                       f'Theorem 5.2.5 does not apply; choose a with a QNR mod b.  '
                       f'(Note: even for prime b, a QR mod b may still give a division '
                       f'algebra via other primes, but this cannot be confirmed within §5.2.)')
    # (4) All conditions met — exact conclusion
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
_T   = np.array([[ 1.,  1.], [0.,  1.]])
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

# ── App layer ─────────────────────────────────────────────────────────────────

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
TILE_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
               '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990']


TYPE_COLORS = {
    'identity':   '#888888',
    'elliptic':   '#e41a1c',
    'parabolic':  '#377eb8',
    'hyperbolic': '#4daf4a',
}
TYPE_SHADES = {
    'elliptic':   ['#e41a1c', '#ff6666', '#ff9999', '#ffcccc'],
    'parabolic':  ['#377eb8', '#66aaff', '#99ccff', '#cce5ff'],
    'hyperbolic': ['#4daf4a', '#88cc66', '#aaddaa', '#cceecc'],
    'identity':   ['#888888'],
}


class HyperbolicVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(13, 10))
        self.fig.suptitle('Hyperbolic Geometry Visualizer', fontsize=13)

        # Panels
        self.ax_H = self.fig.add_axes([0.05, 0.40, 0.42, 0.55])
        self.ax_D = self.fig.add_axes([0.53, 0.40, 0.42, 0.55])
        self._setup_H()
        self._setup_D()

        # Mode radio
        ax_mode = self.fig.add_axes([0.05, 0.26, 0.15, 0.12])
        ax_mode.set_facecolor('0.95')
        self.radio_mode = RadioButtons(ax_mode,
                                       ('Geodesics', 'Fund. Domain', 'Dirichlet'),
                                       activecolor='steelblue')
        self.radio_mode.on_clicked(self._on_mode_change)
        self.fig.text(0.05, 0.385, 'Mode', fontsize=9, color='0.4')

        # Group radio (6 presets — taller box)
        ax_grp = self.fig.add_axes([0.22, 0.23, 0.25, 0.16])
        ax_grp.set_facecolor('0.95')
        self.radio_group = RadioButtons(ax_grp,
                                        [g.label for g in GROUP_PRESETS],
                                        activecolor='darkorange')
        self.radio_group.on_clicked(self._on_group_change)
        self.fig.text(0.22, 0.395, 'Group', fontsize=9, color='0.4')

        # Custom quaternion input row
        self.fig.text(0.50, 0.375, 'Custom Γ(a,b/ℚ)', fontsize=9, color='0.4')
        ax_ta = self.fig.add_axes([0.535, 0.325, 0.055, 0.04])
        self.tb_a = TextBox(ax_ta, 'a = ', initial='3')
        ax_tb = self.fig.add_axes([0.655, 0.325, 0.055, 0.04])
        self.tb_b = TextBox(ax_tb, 'b = ', initial='5')
        ax_gen = self.fig.add_axes([0.755, 0.325, 0.14, 0.04])
        self.btn_gen = Button(ax_gen, 'Generate Γ(a,b)')
        self.btn_gen.on_clicked(self._on_generate_quaternion)

        # Clear button
        ax_clear = self.fig.add_axes([0.50, 0.265, 0.07, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self._on_clear)

        # Depth controls (Fund. Domain only)
        self._ax_depth_lbl = self.fig.add_axes([0.59, 0.265, 0.07, 0.04])
        self._ax_depth_lbl.axis('off')
        self._depth_label = self._ax_depth_lbl.text(
            0.5, 0.5, 'Depth: 2', ha='center', va='center', fontsize=10)
        ax_minus = self.fig.add_axes([0.67, 0.265, 0.035, 0.04])
        ax_plus  = self.fig.add_axes([0.71, 0.265, 0.035, 0.04])
        self.btn_minus = Button(ax_minus, '−')
        self.btn_plus  = Button(ax_plus,  '+')
        self.btn_minus.on_clicked(self._on_depth_minus)
        self.btn_plus.on_clicked(self._on_depth_plus)

        # Show fixed points toggle (Fund. Domain only)
        self._ax_fp = self.fig.add_axes([0.755, 0.265, 0.14, 0.04])
        self.btn_fp = Button(self._ax_fp, 'Fixed pts: OFF')
        self.btn_fp.on_clicked(self._on_toggle_fp)

        # Description panel
        self.ax_desc = self.fig.add_axes([0.05, 0.01, 0.90, 0.22])
        self.ax_desc.axis('off')
        self._desc = self.ax_desc.text(
            0.0, 1.0, '', va='top', ha='left', fontsize=9,
            family='monospace', transform=self.ax_desc.transAxes,
            linespacing=1.5)

        # State
        self.mode = 'Geodesics'
        self.current_group = GROUP_PRESETS[0]
        self.depth = 2
        self.show_fixed_pts = False
        self.points_buffer = []
        self.color_idx = 0
        self.geodesic_artists = []
        self.marker_artists = []
        self.domain_artists = []
        self.fp_artists = []
        self.dirichlet_z0 = None

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._set_fd_controls_visible(False)
        self._update_description()

    # ── panel setup ──────────────────────────────────────────────────────────

    def _setup_H(self):
        ax = self.ax_H
        ax.set_title('Upper Half-Plane ℌ')
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, 4)
        ax.axhline(0, color='k', lw=1)
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def _setup_D(self):
        ax = self.ax_D
        ax.set_title('Poincaré Disk 𝔻')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='k', lw=2))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Re(w)')
        ax.set_ylabel('Im(w)')

    # ── mode / group / depth controls ─────────────────────────────────────────

    def _on_mode_change(self, label):
        self.mode = label
        self._clear_domain_artists()
        self._clear_fp_artists()
        self.points_buffer = []
        self.dirichlet_z0 = None
        fd = (label == 'Fund. Domain')
        self._set_fd_controls_visible(fd)
        if fd:
            self._draw_tessellation()
        self._update_description()
        self.fig.canvas.draw_idle()

    def _on_group_change(self, label):
        self.current_group = next(g for g in GROUP_PRESETS if g.label == label)
        self._clear_domain_artists()
        self._clear_fp_artists()
        self.dirichlet_z0 = None
        if self.mode == 'Fund. Domain':
            self._draw_tessellation()
        elif self.mode == 'Dirichlet' and self.dirichlet_z0 is not None:
            self._handle_dirichlet_click(self.dirichlet_z0)
        self._update_description()
        self.fig.canvas.draw_idle()

    def _set_fd_controls_visible(self, v):
        for ax in [self._ax_depth_lbl, self.btn_minus.ax,
                   self.btn_plus.ax, self._ax_fp]:
            ax.set_visible(v)

    def _on_depth_minus(self, _):
        if self.depth > 0:
            self.depth -= 1
            self._depth_label.set_text(f'Depth: {self.depth}')
            self._clear_domain_artists()
            self._clear_fp_artists()
            self._draw_tessellation()
            self._update_description()
            self.fig.canvas.draw_idle()

    def _on_depth_plus(self, _):
        self.depth += 1
        self._depth_label.set_text(f'Depth: {self.depth}')
        self._clear_domain_artists()
        self._clear_fp_artists()
        self._draw_tessellation()
        self._update_description()
        self.fig.canvas.draw_idle()

    def _on_toggle_fp(self, _):
        self.show_fixed_pts = not self.show_fixed_pts
        self.btn_fp.label.set_text(
            f'Fixed pts: {"ON" if self.show_fixed_pts else "OFF"}')
        self._clear_fp_artists()
        if self.show_fixed_pts and self.mode == 'Fund. Domain':
            self._draw_fixed_points()
        self.fig.canvas.draw_idle()

    # ── click dispatch ────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes == self.ax_H:
            z = complex(event.xdata, event.ydata)
            if z.imag <= 0:
                return
        elif event.inaxes == self.ax_D:
            w = complex(event.xdata, event.ydata)
            if abs(w) >= 1:
                return
            z = cayley_inv(w)
            if z.imag <= 0:
                return
        else:
            return

        if self.mode == 'Geodesics':
            self._handle_geodesic_click(z)
        elif self.mode == 'Dirichlet':
            self._handle_dirichlet_click(z)

    # ── geodesic mode ─────────────────────────────────────────────────────────

    def _handle_geodesic_click(self, z):
        color = COLORS[self.color_idx % len(COLORS)]
        w = cayley(z)
        m_H, = self.ax_H.plot(z.real, z.imag, 'o', color=color, ms=6, zorder=5)
        m_D, = self.ax_D.plot(w.real, w.imag, 'o', color=color, ms=6, zorder=5)
        self.marker_artists.extend([m_H, m_D])
        self.points_buffer.append(z)
        self.fig.canvas.draw_idle()

        if len(self.points_buffer) == 2:
            z1, z2 = self.points_buffer
            self._draw_geodesic(z1, z2, color)
            self.points_buffer = []
            self.color_idx += 1
            self._update_description(z1=z1, z2=z2)

    def _draw_geodesic(self, z1, z2, color):
        pts_H = sample_geodesic_H(z1, z2)
        pts_D = cayley(pts_H)
        l_H, = self.ax_H.plot(pts_H.real, pts_H.imag, color=color, lw=2)
        l_D, = self.ax_D.plot(pts_D.real, pts_D.imag, color=color, lw=2)
        self.geodesic_artists.append((l_H, l_D))
        self.fig.canvas.draw_idle()

    # ── fundamental domain mode ───────────────────────────────────────────────

    def _draw_tessellation(self):
        g = self.current_group
        elements = g.enum_func(self.depth)

        if g.use_std_fd:
            self._draw_tessellation_std(elements)
        else:
            self._draw_tessellation_dirichlet(g, elements)

        if self.show_fixed_pts:
            self._draw_fixed_points()

    def _tile_color(self, M, shade_idx):
        typ = element_type(M) if not np.allclose(M, np.eye(2)) else 'identity'
        shades = TYPE_SHADES.get(typ, TYPE_SHADES['identity'])
        return shades[shade_idx % len(shades)]

    def _draw_tessellation_std(self, elements):
        arcs = std_fd_arcs()
        for idx, M in enumerate(elements):
            color = self._tile_color(M, idx)
            for (a, b) in arcs:
                try:
                    ga, gb = mobius(M, a), mobius(M, b)
                    if ga.imag <= 1e-9 or gb.imag <= 1e-9:
                        continue
                    pts_H = sample_geodesic_H(ga, gb)
                    pts_D = cayley(pts_H)
                    l_H, = self.ax_H.plot(pts_H.real, pts_H.imag,
                                          color=color, lw=2.2)
                    l_D, = self.ax_D.plot(pts_D.real, pts_D.imag,
                                          color=color, lw=2.2)
                    self.domain_artists.extend([l_H, l_D])
                except Exception:
                    continue

    def _draw_tessellation_dirichlet(self, g, elements):
        fd_arcs = dirichlet_boundary(g.canonical_z0, elements)
        for idx, M in enumerate(elements):
            color = self._tile_color(M, idx)
            for arc in fd_arcs:
                try:
                    trf = mobius(M, arc)
                    seg = trf[trf.imag > 1e-9]
                    if len(seg) < 2:
                        continue
                    seg_D = cayley(seg)
                    l_H, = self.ax_H.plot(seg.real, seg.imag,
                                          color=color, lw=2.2)
                    l_D, = self.ax_D.plot(seg_D.real, seg_D.imag,
                                          color=color, lw=2.2)
                    self.domain_artists.extend([l_H, l_D])
                except Exception:
                    continue

    # ── fixed-point markers ───────────────────────────────────────────────────

    def _draw_fixed_points(self):
        elements = self.current_group.enum_func(self.depth)
        I = np.eye(2)
        for M in elements:
            if np.allclose(M, I) or np.allclose(M, -I):
                continue
            typ = element_type(M)
            fps = fixed_points(M)
            color = TYPE_COLORS[typ]
            for fp in fps:
                if fp == np.inf:
                    continue
                if typ == 'elliptic' and fp.imag > 1e-9:
                    # interior point in ℌ
                    z = fp
                    w = cayley(z)
                    a_H, = self.ax_H.plot(z.real, z.imag, '*',
                                          color=color, ms=8, zorder=7)
                    a_D, = self.ax_D.plot(w.real, w.imag, '*',
                                          color=color, ms=8, zorder=7)
                    self.fp_artists.extend([a_H, a_D])
                elif typ in ('parabolic', 'hyperbolic') and abs(fp.imag) < 1e-6:
                    x = fp.real
                    xlim_H = self.ax_H.get_xlim()
                    if xlim_H[0] <= x <= xlim_H[1]:
                        marker = '^' if typ == 'parabolic' else 'o'
                        a_H, = self.ax_H.plot(x, 0, marker,
                                              color=color, ms=7,
                                              clip_on=False, zorder=7)
                        self.fp_artists.append(a_H)
                    # map to disk boundary
                    w = cayley(complex(x, 1e-9))
                    a_D, = self.ax_D.plot(w.real, w.imag, marker,
                                          color=color, ms=7, zorder=7)
                    self.fp_artists.append(a_D)

    def _clear_fp_artists(self):
        for a in self.fp_artists:
            try:
                a.remove()
            except Exception:
                pass
        self.fp_artists = []

    # ── Dirichlet mode ────────────────────────────────────────────────────────

    def _handle_dirichlet_click(self, z0):
        self._clear_domain_artists()
        self.dirichlet_z0 = z0
        w0 = cayley(z0)
        m_H, = self.ax_H.plot(z0.real, z0.imag, '*', color='red', ms=10, zorder=6)
        m_D, = self.ax_D.plot(w0.real, w0.imag, '*', color='red', ms=10, zorder=6)
        self.domain_artists.extend([m_H, m_D])

        elements = self.current_group.enum_func(self.depth + 2)
        arcs = dirichlet_boundary(z0, elements)
        for seg in arcs:
            seg_D = cayley(seg)
            a_H, = self.ax_H.plot(seg.real, seg.imag,
                                  color='purple', lw=1.8, zorder=4)
            a_D, = self.ax_D.plot(seg_D.real, seg_D.imag,
                                  color='purple', lw=1.8, zorder=4)
            self.domain_artists.extend([a_H, a_D])

        self._update_description(z0=z0, n_arcs=len(arcs))
        self.fig.canvas.draw_idle()

    # ── custom quaternion input ───────────────────────────────────────────────

    def _on_generate_quaternion(self, _):
        try:
            a = int(self.tb_a.text.strip())
            b = int(self.tb_b.text.strip())
            if a <= 0:
                raise ValueError('a must be a positive integer')
            if b == 0:
                raise ValueError('b must be non-zero')
        except ValueError as exc:
            self._desc.set_text(f'Custom quaternion: invalid input — {exc}')
            self.fig.canvas.draw_idle()
            return

        is_div, div_msg = check_division_algebra(a, b)
        if not is_div:
            self._desc.set_text(f'Γ({a},{b}/ℚ) — cannot generate:\n{div_msg}')
            self.fig.canvas.draw_idle()
            return

        norm_eq = f'x₀²−{a}x₁²−{b}x₂²+{a*b}x₃²=1'
        custom = GroupPreset(
            label=f'Γ({a},{b}/ℚ)',
            group_desc=f'A=({a},{b}/ℚ), standard order.  Norm: {norm_eq}.',
            fd_desc='Compact FD: no cusps, all elements hyperbolic.',
            enum_func=lambda d, _a=a, _b=b: gen_quaternion(_a, _b, d + 2),
            canonical_z0=2j,
        )
        self.current_group = custom
        self._clear_domain_artists()
        self._clear_fp_artists()
        self.dirichlet_z0 = None

        if self.mode == 'Fund. Domain':
            self._draw_tessellation()

        elements = custom.enum_func(self.depth)
        counts = self._element_type_counts(elements)
        self._desc.set_text(
            f'Custom  Γ({a},{b}/ℚ)  │  norm eq: {norm_eq}\n'
            f'{div_msg}\n'
            f'Depth {self.depth}: {len(elements)} elements at coord_bound={self.depth+2}  '
            f'(elliptic: {counts["elliptic"]},  parabolic: {counts["parabolic"]},  '
            f'hyperbolic: {counts["hyperbolic"]})'
        )
        self.fig.canvas.draw_idle()

    # ── clear ─────────────────────────────────────────────────────────────────

    def _clear_domain_artists(self):
        for a in self.domain_artists:
            try:
                a.remove()
            except Exception:
                pass
        self.domain_artists = []

    def _on_clear(self, _):
        for l_H, l_D in self.geodesic_artists:
            l_H.remove()
            l_D.remove()
        for m in self.marker_artists:
            m.remove()
        self._clear_domain_artists()
        self._clear_fp_artists()
        self.geodesic_artists = []
        self.marker_artists = []
        self.points_buffer = []
        self.color_idx = 0
        self.dirichlet_z0 = None
        if self.mode == 'Fund. Domain':
            self._draw_tessellation()
        self._update_description()
        self.fig.canvas.draw_idle()

    # ── text description ──────────────────────────────────────────────────────

    def _element_type_counts(self, elements):
        counts = {'elliptic': 0, 'parabolic': 0, 'hyperbolic': 0}
        I = np.eye(2)
        for M in elements:
            if np.allclose(M, I) or np.allclose(M, -I):
                continue
            counts[element_type(M)] += 1
        return counts

    def _update_description(self, z1=None, z2=None, z0=None, n_arcs=None):
        g = self.current_group

        if self.mode == 'Geodesics':
            if z1 is not None and z2 is not None:
                d = hyp_dist(z1, z2)
                if geodesic_type(z1, z2) == 'vertical':
                    geo_info = f'Type: vertical line  |  x = {z1.real:.3f}'
                else:
                    c, r = geodesic_H_params(z1, z2)
                    geo_info = (f'Type: semicircle  |  '
                                f'center c = {c:.3f},  radius r = {r:.3f}')
                text = (
                    f'Geodesic through  z₁ = {z1.real:.3f}+{z1.imag:.3f}i'
                    f',  z₂ = {z2.real:.3f}+{z2.imag:.3f}i\n'
                    f'{geo_info}\n'
                    f'Hyperbolic distance:  ρ(z₁, z₂) = {d:.4f}'
                )
            else:
                text = (
                    'Click two points in either panel to draw the hyperbolic geodesic through them.\n'
                    'Points in the disk are mapped to ℌ via  f⁻¹(w) = i(1+w)/(1−w).'
                )

        elif self.mode == 'Fund. Domain':
            elements = g.enum_func(self.depth)
            counts = self._element_type_counts(elements)
            text = (
                f'Group: {g.group_desc}\n'
                f'Fundamental domain: {g.fd_desc}\n'
                f'Depth {self.depth}: {len(elements)} tiles γ(FD),  γ ∈ {g.label}.\n'
                f'Element types — '
                f'elliptic: {counts["elliptic"]}  '
                f'(|tr|<2, rotation in ℌ)  │  '
                f'parabolic: {counts["parabolic"]}  '
                f'(|tr|=2, fixes ∞ or real pt)  │  '
                f'hyperbolic: {counts["hyperbolic"]}  '
                f'(|tr|>2, translates along axis)'
            )

        else:  # Dirichlet
            if z0 is not None and n_arcs is not None:
                elements = g.enum_func(self.depth + 2)
                counts = self._element_type_counts(elements)
                text = (
                    f'Group: {g.group_desc}\n'
                    f'Dirichlet domain centred at  z₀ = {z0.real:.3f}+{z0.imag:.3f}i\n'
                    f'D(z₀) = {{ z ∈ ℌ : ρ(z,z₀) ≤ ρ(z, γ(z₀))  for all γ ∈ {g.label} }}\n'
                    f'Boundary arcs: {n_arcs}  │  '
                    f'Element types — '
                    f'elliptic: {counts["elliptic"]}  '
                    f'parabolic: {counts["parabolic"]}  '
                    f'hyperbolic: {counts["hyperbolic"]}'
                )
            else:
                text = (
                    f'Group: {g.group_desc}\n'
                    'Click a point z₀ in either panel to compute its Dirichlet domain.\n'
                    'D(z₀) = { z ∈ ℌ : ρ(z,z₀) ≤ ρ(z, γ(z₀))  for all γ ∈ Γ }\n'
                    'Boundary: hyperbolic perpendicular bisectors of segments z₀ → γ(z₀).\n'
                    'Three element types — '
                    'elliptic (|tr|<2): rotation around fixed pt in ℌ  │  '
                    'parabolic (|tr|=2): fixes one boundary pt  │  '
                    'hyperbolic (|tr|>2): translates along geodesic axis'
                )

        self._desc.set_text(text)


if __name__ == '__main__':
    viz = HyperbolicVisualizer()
    plt.show()
