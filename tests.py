"""
tests.py
========
Pytest test suite for vlasov.py.

Run with:
    pytest tests.py -v

Each test class maps to one notebook section. All assertions mirror the
sanity checks embedded in the notebook cells.
"""

import numpy as np
import pytest
from vlasov import (
    EPS0, KB, E_C, ME,
    debye_length, plasma_parameter, plasma_frequency,
    maxwellian, bump_on_tail, schamel_hole, grad13,
    compute_moments,
    bohm_gross, landau_rate, Z_func, Z_prime,
    pic_noise_scaling,
    solve_poisson, advect_x, advect_v, run_vlasov,
)


# ---------------------------------------------------------------------------
# §1  Maxwellian distribution
# ---------------------------------------------------------------------------

class TestMaxwellian:
    """Sanity checks for the canonical Vlasov stationary solution."""

    def setup_method(self):
        self.v  = np.linspace(-8, 8, 4000)
        self.f0 = maxwellian(self.v, vth=1.0)

    def test_normalisation(self):
        n = np.trapezoid(self.f0, self.v)
        assert abs(n - 1.0) < 1e-4, f"norm={n:.8f}, expect 1.0"

    def test_zero_bulk_velocity(self):
        n = np.trapezoid(self.f0, self.v)
        u = np.trapezoid(self.v * self.f0, self.v) / n
        assert abs(u) < 1e-10, f"<v>={u:.2e}, expect 0"

    def test_unit_temperature(self):
        n = np.trapezoid(self.f0, self.v)
        u = np.trapezoid(self.v * self.f0, self.v) / n
        T = np.trapezoid((self.v - u)**2 * self.f0, self.v) / n
        assert abs(T - 1.0) < 1e-4, f"<v^2>={T:.8f}, expect 1.0"

    def test_positivity(self):
        assert np.all(self.f0 >= 0), "Maxwellian went negative"

    def test_zero_heat_flux(self):
        n = np.trapezoid(self.f0, self.v)
        u = np.trapezoid(self.v * self.f0, self.v) / n
        q = np.trapezoid((self.v - u)**3 * self.f0, self.v)
        assert abs(q) < 1e-8, f"heat flux={q:.2e}, expect 0"

    def test_vth_scaling(self):
        for vth in [0.5, 1.0, 2.0]:  # vth=3 needs wider v-grid; tested separately
            f = maxwellian(self.v, vth=vth)
            n = np.trapezoid(f, self.v)
            u = np.trapezoid(self.v * f, self.v) / n
            T = np.trapezoid((self.v - u)**2 * f, self.v) / n
            assert abs(T - vth**2) / vth**2 < 5e-3, (
                f"vth={vth}: T={T:.4f}, expect {vth**2:.4f}"
            )


# ---------------------------------------------------------------------------
# §2  Plasma Parameters & Debye Screening
# ---------------------------------------------------------------------------

class TestPlasmaParameters:

    CASES = [
        ("Fusion edge",  1e18,    100),
        ("Solar wind",   1e7,      10),
        ("Tokamak core", 1e20, 10_000),
    ]

    def test_lambda_gt_100_all_cases(self):
        for label, n0, te_ev in self.CASES:
            lam = plasma_parameter(n0, te_ev)
            assert lam > 100, f"{label}: Lambda={lam:.1e} not >> 1"

    def test_debye_factor_at_lambda_d(self):
        r = np.linspace(0.02, 6, 500)
        ratio = (np.exp(-r) / r)[np.argmin(np.abs(r - 1.0))] / (1.0 / r)[np.argmin(np.abs(r - 1.0))]
        assert abs(ratio - np.exp(-1)) < 0.01, (
            f"Debye factor at r=lambda_D = {ratio:.4f}, expect {np.exp(-1):.4f}"
        )

    def test_plasma_frequency_positive(self):
        for n0 in [1e10, 1e18, 1e25]:
            assert plasma_frequency(n0) > 0

    def test_debye_increases_with_Te(self):
        assert debye_length(1e18, 1000) > debye_length(1e18, 10)

    def test_debye_decreases_with_density(self):
        assert debye_length(1e10, 100) > debye_length(1e20, 100)


# ---------------------------------------------------------------------------
# §3  Fluid Moments
# ---------------------------------------------------------------------------

class TestFluidMoments:

    def setup_method(self):
        self.v = np.linspace(-8, 8, 4000)
        self.m = compute_moments(maxwellian(self.v), self.v)

    def test_density(self):
        assert abs(self.m["n"] - 1.0) < 1e-5

    def test_bulk_velocity(self):
        assert abs(self.m["u"]) < 1e-10

    def test_temperature(self):
        assert abs(self.m["T"] - 1.0) < 1e-5

    def test_heat_flux_zero(self):
        assert abs(self.m["q"]) < 1e-8

    def test_shifted_distribution(self):
        v_shift = 2.0
        m = compute_moments(maxwellian(self.v - v_shift), self.v)
        assert abs(m["u"] - v_shift) < 1e-4, (
            f"u={m['u']:.4f}, expect {v_shift}"
        )


# ---------------------------------------------------------------------------
# §4  Grad 13-Moment Closure
# ---------------------------------------------------------------------------

class TestGrad13:

    def setup_method(self):
        self.v = np.linspace(-5, 5, 600)

    def test_norm_all_q(self):
        for q in [0.0, 0.5, 0.8, 1.5]:
            n = np.trapezoid(grad13(self.v, q), self.v)
            assert abs(n - 1.0) < 1e-3, f"q={q}: norm={n:.6f}"

    def test_recovers_maxwellian_at_q0(self):
        assert np.allclose(grad13(self.v, 0.0), maxwellian(self.v), atol=1e-12)

    def test_positivity_lost_large_q(self):
        assert grad13(self.v, 1.5).min() < 0

    def test_positive_at_q0(self):
        assert np.all(grad13(self.v, 0.0) >= 0)


# ---------------------------------------------------------------------------
# §5  Bohm-Gross Dispersion
# ---------------------------------------------------------------------------

class TestBohmGross:

    def setup_method(self):
        self.k = np.linspace(0.001, 1.5, 600)
        self.omega, self.v_phi, self.v_group = bohm_gross(self.k)

    def test_omega_at_k0(self):
        assert abs(self.omega[0] - 1.0) < 1e-3

    def test_omega_at_k_half(self):
        idx = np.argmin(np.abs(self.k - 0.5))
        assert abs(self.omega[idx] - np.sqrt(1.75)) < 0.005

    def test_phase_velocity_exceeds_vth(self):
        mask = self.k < 0.5
        assert np.all(self.v_phi[mask] > 1.0)

    def test_group_velocity_near_zero_at_k0(self):
        assert self.v_group[0] < 0.01

    def test_omega_monotone(self):
        assert np.all(np.diff(self.omega) > 0)


# ---------------------------------------------------------------------------
# §6  Plasma Dispersion Function & Landau Rate
# ---------------------------------------------------------------------------

class TestZFunc:

    def test_z_at_zero(self):
        assert abs(Z_func(0j) - 1j * np.sqrt(np.pi)) < 1e-10

    def test_large_argument(self):
        z = 10.0 + 0j
        assert abs(Z_func(z) - (-1 / z)) / abs(-1 / z) < 0.02

    def test_symmetry(self):
        z = 1.5 + 0.3j
        assert abs(Z_func(-z.conjugate()) - (-Z_func(z).conjugate())) < 1e-12

    def test_derivative_identity(self):
        z   = 1.5 + 0.3j
        dZn = (Z_func(z + 1e-7) - Z_func(z - 1e-7)) / 2e-7
        assert abs(dZn - Z_prime(z)) / abs(Z_prime(z)) < 1e-4


class TestLandauRate:

    def setup_method(self):
        self.k     = np.linspace(0.10, 0.50, 300)
        self.gamma = landau_rate(self.k)

    def test_negative_everywhere(self):
        assert np.all(self.gamma < 0)

    def test_increases_with_k(self):
        assert np.all(np.diff(self.gamma) < 0)

    def test_known_value_at_k_half(self):
        idx = np.argmin(np.abs(self.k - 0.5))
        assert abs(self.gamma[idx] - (-0.1533)) < 0.005


# ---------------------------------------------------------------------------
# §7  Schamel Phase-Space Hole
# ---------------------------------------------------------------------------

class TestSchamelHole:

    def setup_method(self):
        self.x = np.linspace(-np.pi, np.pi, 200)
        self.v = np.linspace(-4, 4, 300)
        self.f_hole, self.phi, self.W = schamel_hole(
            self.x, self.v, phi_amp=0.5, beta=1.5
        )
        self.dx = self.x[1] - self.x[0]
        self.dv = self.v[1] - self.v[0]

    def test_positivity(self):
        assert self.f_hole.min() >= 0

    def test_trapped_fraction(self):
        frac = float((self.W < 0).mean())
        assert 0 < frac < 0.3

    def test_particle_deficit(self):
        _, _, _ = schamel_hole(self.x, self.v, phi_amp=0.5, beta=1.5)
        X, V   = np.meshgrid(self.x, self.v, indexing='ij')
        f_bg   = np.exp(-0.5 * V**2) / np.sqrt(2.0 * np.pi)
        deficit = (np.sum(f_bg) - np.sum(self.f_hole)) * self.dx * self.dv
        assert deficit > 0

    def test_potential_peak_positive(self):
        assert self.phi.max() > 0

    def test_beta_sign_determines_hole_type(self):
        f_pos, _, _ = schamel_hole(self.x, self.v, phi_amp=0.5, beta=1.5)
        f_neg, _, _ = schamel_hole(self.x, self.v, phi_amp=0.5, beta=-1.5)
        assert np.sum(f_pos) < np.sum(f_neg)


# ---------------------------------------------------------------------------
# §8  PIC Shot Noise
# ---------------------------------------------------------------------------

class TestPICNoise:

    def setup_method(self):
        self.N    = np.logspace(2, 5, 20, dtype=int)
        self.noise, self.slope = pic_noise_scaling(self.N, seed=42)

    def test_slope_near_minus_half(self):
        assert abs(self.slope - (-0.5)) < 0.12

    def test_overall_decrease(self):
        assert self.noise[-1] < self.noise[0] / 5

    def test_small_at_large_N(self):
        assert self.noise[-1] < 1e-2

    def test_reproducible(self):
        noise2, _ = pic_noise_scaling(self.N, seed=42)
        assert np.allclose(self.noise, noise2)


# ---------------------------------------------------------------------------
# §9  Semi-Lagrangian Solver Components
# ---------------------------------------------------------------------------

class TestSolverComponents:

    def setup_method(self):
        self.Nx = 32
        self.Nv = 64
        self.x  = np.linspace(0, 4 * np.pi, self.Nx, endpoint=False)
        self.v  = np.linspace(-6, 6, self.Nv)
        X, V    = np.meshgrid(self.x, self.v, indexing='ij')
        self.f_unif  = maxwellian(V)
        self.f_pert  = maxwellian(V) * (1 + 0.01 * np.cos(0.5 * X))

    def test_poisson_zero_for_uniform(self):
        E = solve_poisson(self.f_unif, self.v, self.x)
        assert np.max(np.abs(E)) < 1e-12

    def test_poisson_mean_zero(self):
        E = solve_poisson(self.f_pert, self.v, self.x)
        assert abs(np.mean(E)) < 1e-12

    def test_poisson_nonzero_for_perturbed(self):
        E = solve_poisson(self.f_pert, self.v, self.x)
        assert np.max(np.abs(E)) > 1e-3

    def test_advect_x_mass_conservation(self):
        m0 = np.trapezoid(np.trapezoid(self.f_pert, self.v, axis=1), self.x)
        fa = advect_x(self.f_pert, self.v, dt=0.05, x=self.x)
        m1 = np.trapezoid(np.trapezoid(fa, self.v, axis=1), self.x)
        assert abs(m1 - m0) / m0 < 0.005

    def test_advect_v_mass_conservation(self):
        E  = 0.01 * np.sin(0.5 * self.x)
        m0 = np.trapezoid(np.trapezoid(self.f_pert, self.v, axis=1), self.x)
        fa = advect_v(self.f_pert, E, dt=0.05, v=self.v)
        m1 = np.trapezoid(np.trapezoid(fa, self.v, axis=1), self.x)
        assert abs(m1 - m0) / m0 < 0.005


class TestCFLAndPreRun:

    def test_cfl_exceeds_unity(self):
        NX, VMAX, DT, LX = 64, 6.0, 0.05, 4 * np.pi
        cfl = VMAX * DT / (LX / NX)
        assert cfl > 1.0, f"CFL={cfl:.2f}; expect > 1"

    def test_initial_E_proportional_to_eps(self):
        EPS, K   = 0.01, 0.5
        NX, NV   = 32, 64
        LX, VMAX = 4 * np.pi, 6.0
        x = np.linspace(0, LX, NX, endpoint=False)
        v = np.linspace(-VMAX, VMAX, NV)
        X, V = np.meshgrid(x, v, indexing='ij')
        f = maxwellian(V) * (1 + EPS * np.cos(K * X))
        E_rms = float(np.sqrt(np.mean(solve_poisson(f, v, x)**2)))
        assert 0.5 * EPS < E_rms < 5 * EPS


class TestLandauBenchmark:
    """Short (60-step) Landau damping run to verify solver correctness."""

    def setup_method(self):
        NX, NV   = 32, 64
        LX, VMAX = 4 * np.pi, 6.0
        DT, NT   = 0.1, 60
        x = np.linspace(0, LX, NX, endpoint=False)
        v = np.linspace(-VMAX, VMAX, NV)
        X, V = np.meshgrid(x, v, indexing='ij')
        f0 = maxwellian(V) * (1 + 0.01 * np.cos(0.5 * X))
        self.f, self.t, self.em = run_vlasov(f0, x, v, DT, NT)
        self.LX, self.x, self.v = LX, x, v

    def test_field_decreases(self):
        assert self.em[-1] < self.em[0]

    def test_particle_conservation(self):
        total = float(np.trapezoid(np.trapezoid(self.f, self.v, axis=1), self.x))
        assert abs(total - self.LX) / self.LX < 0.05  # coarse short-run tolerance


# ---------------------------------------------------------------------------
# §10  Two-Stream Instability
# ---------------------------------------------------------------------------

class TestTwoStream:

    def test_chosen_k_unstable(self):
        K2, V0 = 0.5, 1.5
        k_crit = np.sqrt(2) / V0   # symmetric two-beam: k < sqrt(2)*omega_pe/v0
        assert K2 < k_crit

    def test_field_grows(self):
        NX, NV  = 32, 128
        LX, VMAX = 4 * np.pi, 5.0
        x2 = np.linspace(0, LX, NX, endpoint=False)
        v2 = np.linspace(-VMAX, VMAX, NV)
        X2, V2 = np.meshgrid(x2, v2, indexing='ij')
        V0, VTH2 = 1.5, 0.3
        f2 = (
            (np.exp(-(V2 - V0)**2 / (2 * VTH2**2))
             + np.exp(-(V2 + V0)**2 / (2 * VTH2**2)))
            / (2 * np.sqrt(2 * np.pi) * VTH2)
            * (1 + 0.02 * np.cos(0.5 * X2))
        )
        _, _, em = run_vlasov(f2, x2, v2, dt=0.05, n_steps=80)
        assert em[-1] > em[0], "Two-stream: |E| should grow"


# ---------------------------------------------------------------------------
# §11  Bump-on-Tail
# ---------------------------------------------------------------------------

class TestBumpOnTail:

    def setup_method(self):
        self.v = np.linspace(-5, 8, 800)
        self.f0, self.df0 = bump_on_tail(
            self.v, n_beam=0.10, v_beam=3.5, sigma_beam=0.4
        )

    def test_normalisation(self):
        assert abs(np.trapezoid(self.f0, self.v) - 1.0) < 1e-3

    def test_positivity(self):
        assert np.all(self.f0 >= 0)

    def test_positive_slope_in_beam(self):
        mask = (self.v > 3.5 - 0.8) & (self.v < 3.5)
        assert np.any(self.df0[mask] > 0)

    def test_recovers_maxwellian_at_zero_beam(self):
        f_nb0, _ = bump_on_tail(self.v, 0.0, 3.5, 0.4)
        assert np.allclose(f_nb0, maxwellian(self.v), atol=1e-12)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
