import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

from fuchsian_math import (
    cayley, cayley_inv,
    geodesic_type, geodesic_H_params, sample_geodesic_H,
    mobius, hyp_dist,
    element_type, fixed_points,
    std_fd_arcs, sample_bisector, dirichlet_boundary,
    gen_quaternion, check_division_algebra,
    GroupPreset, GROUP_PRESETS,
)

# ── Colour palette ────────────────────────────────────────────────────────────

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

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

        # Group radio
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

        # Depth controls
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

        # Fixed points toggle
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

    # ── mode / group / depth controls ────────────────────────────────────────

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
                    l_H, = self.ax_H.plot(pts_H.real, pts_H.imag, color=color, lw=2.2)
                    l_D, = self.ax_D.plot(pts_D.real, pts_D.imag, color=color, lw=2.2)
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
                    l_H, = self.ax_H.plot(seg.real, seg.imag, color=color, lw=2.2)
                    l_D, = self.ax_D.plot(seg_D.real, seg_D.imag, color=color, lw=2.2)
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
                    z = fp
                    w = cayley(z)
                    a_H, = self.ax_H.plot(z.real, z.imag, '*', color=color, ms=8, zorder=7)
                    a_D, = self.ax_D.plot(w.real, w.imag, '*', color=color, ms=8, zorder=7)
                    self.fp_artists.extend([a_H, a_D])
                elif typ in ('parabolic', 'hyperbolic') and abs(fp.imag) < 1e-6:
                    x = fp.real
                    xlim_H = self.ax_H.get_xlim()
                    if xlim_H[0] <= x <= xlim_H[1]:
                        marker = '^' if typ == 'parabolic' else 'o'
                        a_H, = self.ax_H.plot(x, 0, marker, color=color, ms=7,
                                              clip_on=False, zorder=7)
                        self.fp_artists.append(a_H)
                    w = cayley(complex(x, 1e-9))
                    a_D, = self.ax_D.plot(w.real, w.imag, marker, color=color, ms=7, zorder=7)
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
            a_H, = self.ax_H.plot(seg.real, seg.imag, color='purple', lw=1.8, zorder=4)
            a_D, = self.ax_D.plot(seg_D.real, seg_D.imag, color='purple', lw=1.8, zorder=4)
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
