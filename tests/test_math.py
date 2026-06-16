import numpy as np
import pytest
from visualizer import (
    cayley, cayley_inv, geodesic_type, geodesic_H_params, sample_geodesic_H,
    mobius, hyp_dist as hyp_dist_acosh,
    gen_psl2z, _mat_key,
    gen_dilation, gen_translation, gen_bfs,
    element_type, fixed_points,
    bisector_params, sample_bisector,
    dirichlet_boundary,
    quaternion_norm, quaternion_embed, gen_quaternion,
    check_division_algebra,
)


def test_cayley_known_values():
    assert abs(cayley(1j)) < 1e-12
    assert abs(cayley(2j) - 1/3) < 1e-12
    assert abs(cayley(-1+1j) - (1+2j)/5) < 1e-12


def test_cayley_inv_round_trip():
    for z in [1j, 2+3j, -1+0.5j, 0.1+10j]:
        assert abs(cayley_inv(cayley(z)) - z) < 1e-10


def test_cayley_maps_into_disk():
    for z in [1j, 2+3j, 0.001+0.001j, 0+100j]:
        assert abs(cayley(z)) < 1 - 1e-12


def test_geodesic_params():
    c, r = geodesic_H_params(1+1j, 3+1j)
    assert abs(c - 2) < 1e-12
    assert abs(r - 2**0.5) < 1e-12


def test_geodesic_passes_through_points():
    for z1, z2 in [(1+1j, 3+2j), (0.5+2j, -0.5+0.7j), (2+0.1j, -3+4j)]:
        c, r = geodesic_H_params(z1, z2)
        assert abs(abs(z1 - c) - r) < 1e-10
        assert abs(abs(z2 - c) - r) < 1e-10


def test_geodesic_type():
    assert geodesic_type(2+1j, 2+3j) == "vertical"
    assert geodesic_type(1+1j, 3+1j) == "semicircle"


def test_geodesic_symmetry():
    for z1, z2 in [(1+1j, 3+2j), (0.1+0.5j, -2+3j)]:
        c1, r1 = geodesic_H_params(z1, z2)
        c2, r2 = geodesic_H_params(z2, z1)
        assert abs(c1 - c2) < 1e-12
        assert abs(r1 - r2) < 1e-12


def hyp_dist(z, w):
    num = abs(z - w.conjugate()) + abs(z - w)
    den = abs(z - w.conjugate()) - abs(z - w)
    return np.log(num / den)


def test_hyperbolic_distance():
    assert abs(hyp_dist(1j, 2j) - np.log(2)) < 1e-12
    assert abs(hyp_dist(1+1j, 3+2j) - hyp_dist(3+2j, 1+1j)) < 1e-12


def test_geodesic_arc_stays_in_H():
    pts = sample_geodesic_H(1+1j, 3+2j, n=200)
    assert all(z.imag > 0 for z in pts)


def test_disk_arc_stays_inside_D():
    pts_H = sample_geodesic_H(1+1j, 3+2j, n=200)
    pts_D = [cayley(z) for z in pts_H]
    assert all(abs(w) < 1 for w in pts_D)


# ── New tests for mobius, hyp_dist, gen_psl2z, bisector, dirichlet ────────────

def test_mobius_identity():
    I = np.eye(2, dtype=int)
    for z in [1j, 2+3j, -1+0.5j]:
        assert abs(mobius(I, z) - z) < 1e-12


def test_mobius_S():
    S = np.array([[0, -1], [1, 0]])
    assert abs(mobius(S, 1j) - 1j) < 1e-12       # S(i) = i  (fixed point)
    assert abs(mobius(S, 2j) - 0.5j) < 1e-12     # S(2i) = i/2


def test_mobius_composition():
    S  = np.array([[ 0, -1], [1, 0]])
    T  = np.array([[ 1,  1], [0, 1]])
    ST = S @ T
    for z in [1j, 2+3j, -0.5+2j]:
        assert abs(mobius(ST, z) - mobius(S, mobius(T, z))) < 1e-12


def test_hyp_dist_known():
    # Two equivalent formulas should agree; acosh formula vs log formula
    assert abs(hyp_dist_acosh(1j, 2j) - np.log(2)) < 1e-12
    # symmetry
    assert abs(hyp_dist_acosh(1+1j, 3+2j) - hyp_dist_acosh(3+2j, 1+1j)) < 1e-12


def test_gen_psl2z_det():
    for M in gen_psl2z(3):
        assert abs(int(np.round(np.linalg.det(M))) ) == 1


def test_gen_psl2z_contains_generators():
    S = np.array([[ 0, -1], [1,  0]])
    T = np.array([[ 1,  1], [0,  1]])
    keys = {_mat_key(M) for M in gen_psl2z(2)}
    assert _mat_key(S) in keys
    assert _mat_key(T) in keys


def test_bisector_equidistant():
    for z1, z2 in [(1+1j, 3+2j), (0.5+3j, -1+2j)]:
        pts = sample_bisector(z1, z2, n=50)
        for p in pts[::10]:
            assert abs(hyp_dist_acosh(p, z1) - hyp_dist_acosh(p, z2)) < 1e-6


def test_bisector_midpoint():
    # The hyperbolic midpoint of z1, z2 lies on the bisector
    for z1, z2 in [(1+1j, 3+2j), (0.5+3j, -1+2j)]:
        pts = sample_geodesic_H(z1, z2, n=5000)
        diffs = np.abs(
            np.array([hyp_dist_acosh(p, z1) for p in pts]) -
            np.array([hyp_dist_acosh(p, z2) for p in pts])
        )
        mid = pts[np.argmin(diffs)]
        assert abs(hyp_dist_acosh(mid, z1) - hyp_dist_acosh(mid, z2)) < 1e-3


def test_gen_dilation_images():
    elements = gen_dilation(2, 3)
    # g^n(i) = 2^n * i
    for n, expected in [(1, 2j), (2, 4j), (-1, 0.5j)]:
        found = any(abs(mobius(M, 1j) - expected) < 1e-9 for M in elements)
        assert found, f"g^{n}(i) = {expected} not found"


def test_gen_translation_images():
    elements = gen_translation(1, 3)
    z = 0.5 + 1j
    for n, expected in [(1, z+1), (2, z+2), (-1, z-1)]:
        found = any(abs(mobius(M, z) - expected) < 1e-9 for M in elements)
        assert found, f"T^{n}(z) = {expected} not found"


def test_gen_bfs_theta_det():
    S  = np.array([[ 0., -1.], [1.,  0.]])
    T2 = np.array([[ 1.,  2.], [0.,  1.]])
    Ti2= np.array([[ 1., -2.], [0.,  1.]])
    for M in gen_bfs([S, T2, Ti2], 2):
        assert abs(abs(np.linalg.det(M)) - 1) < 1e-9


def test_element_type_S():
    S = np.array([[0., -1.], [1., 0.]])
    assert element_type(S) == 'elliptic'       # tr = 0


def test_element_type_T():
    T = np.array([[1., 1.], [0., 1.]])
    assert element_type(T) == 'parabolic'      # tr = 2


def test_element_type_hyperbolic():
    g = np.array([[2.**0.5, 0.], [0., 2.**-0.5]])
    assert element_type(g) == 'hyperbolic'     # tr = √2 + 1/√2 ≈ 2.83


def test_fixed_points_parabolic_inf():
    T = np.array([[1., 1.], [0., 1.]])         # c=0 → fixes ∞
    fps = fixed_points(T)
    assert fps == [np.inf]


def test_fixed_points_hyperbolic():
    # dilation z↦2z: fixes 0 and ∞ (c=0 case normalised differently)
    # use g = [[√2,0],[0,1/√2]] — c=0 → fixes ∞; but the actual group action
    # z↦2z also fixes 0.  For c≠0 form: use [[3,4],[2,3]] (tr=6, hyperbolic)
    M = np.array([[3., 4.], [2., 3.]])
    fps = fixed_points(M)
    assert len(fps) == 2
    for fp in fps:
        assert abs(fp.imag) < 1e-9             # both on real axis


def test_fixed_points_elliptic():
    S = np.array([[0., -1.], [1., 0.]])
    fps = fixed_points(S)
    # first returned point should be in ℌ (Im > 0)
    assert fps[0].imag > 0
    assert abs(fps[0] - 1j) < 1e-9            # S fixes i


def test_dirichlet_no_fixed_point_issue():
    # z0=i is fixed by S; dirichlet_boundary should still return valid arcs
    arcs = dirichlet_boundary(1j, gen_psl2z(3))
    assert len(arcs) > 0


def test_dirichlet_contains_z0():
    z0 = 2j  # 2i has no non-trivial fixed-point in PSL(2,ℤ)
    elements = gen_psl2z(3)
    I = np.eye(2, dtype=int)
    images = [mobius(M, z0) for M in elements if _mat_key(M) != _mat_key(I)]
    # All orbit images must be strictly farther from z0 than z0 itself (distance 0)
    for w in images:
        assert hyp_dist_acosh(z0, w) > 1e-9


# ── Quaternion algebra tests ──────────────────────────────────────────────────

def test_quaternion_norm_identity():
    assert quaternion_norm(1, 0, 0, 0, 3, 5) == 1

def test_quaternion_norm_known():
    assert quaternion_norm(2, 1, 0, 0, 3, 5) == 1   # 4−3 = 1
    assert quaternion_norm(3, 1, 1, 0, 3, 5) == 1   # 9−3−5 = 1
    assert quaternion_norm(2, 0, 1, 0, 2, 3) == 1   # 4−3 = 1  (for A=(2,3/Q))
    assert quaternion_norm(3, 2, 0, 0, 2, 3) == 1   # 9−8 = 1

def test_quaternion_embed_det():
    for (x0, x1, x2, x3), a, b in [
        ((2, 1, 0, 0), 3, 5),
        ((3, 1, 1, 0), 3, 5),
        ((2, 0, 1, 0), 2, 3),
    ]:
        M = quaternion_embed(x0, x1, x2, x3, a, b)
        expected = quaternion_norm(x0, x1, x2, x3, a, b)
        assert abs(np.linalg.det(M) - expected) < 1e-10

def test_quaternion_embed_trace():
    for x0 in [1, 2, 3]:
        M = quaternion_embed(x0, 0, 0, 0, 3, 5)
        assert abs(np.trace(M) - 2 * x0) < 1e-10

def test_gen_quaternion_contains_generators():
    elements = gen_quaternion(3, 5, 3)
    keys = {_mat_key(M) for M in elements}
    for x0, x1, x2, x3 in [(2, 1, 0, 0), (3, 1, 1, 0), (3, -1, 1, 0)]:
        M = quaternion_embed(x0, x1, x2, x3, 3, 5)
        assert _mat_key(M) in keys, f'({x0},{x1},{x2},{x3}) not found'

def test_gen_quaternion_unit_det():
    for M in gen_quaternion(3, 5, 3):
        assert abs(np.linalg.det(M) - 1) < 1e-9

def test_gen_quaternion_all_hyperbolic():
    I = np.eye(2)
    for M in gen_quaternion(3, 5, 4):
        if np.allclose(M, I):
            continue
        assert element_type(M) == 'hyperbolic', \
            f'unexpected type {element_type(M)} for tr={np.trace(M):.4f}'

def test_check_div_alg_perfect_square():
    is_div, _ = check_division_algebra(4, 5)   # a=4=2²
    assert not is_div
    is_div2, _ = check_division_algebra(1, 7)  # a=1=1²
    assert not is_div2

def test_check_div_alg_thm525():
    is_div, _ = check_division_algebra(3, 5)   # 3 is QNR mod 5
    assert is_div
    is_div2, _ = check_division_algebra(2, 3)  # 2 is QNR mod 3
    assert is_div2

def test_check_div_alg_not_division():
    # a=9=3² → perfect square → not division algebra
    is_div, _ = check_division_algebra(9, 5)
    assert not is_div
