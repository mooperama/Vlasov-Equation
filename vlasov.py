"""
vlasov.py
=========
Physics and numerics module for collisionless plasma / Vlasov-Poisson simulations.

All functions operate in normalised units unless stated otherwise:
    length   : Debye length  lambda_D = 1
    velocity : thermal speed  v_th = 1
    time     : inverse plasma frequency  omega_pe^{-1} = 1

Public API
----------
Physical constants (SI)
    EPS0, KB, E_C, ME

Plasma parameters
    debye_length(n0, Te_eV)
    plasma_parameter(n0, Te_eV)
    plasma_frequency(n0)

Distribution functions
    maxwellian(v, vth=1.0)
    bump_on_tail(v, n_beam, v_beam, sigma_beam)
    schamel_hole(x, v, phi_amp, beta, vth=1.0)
    grad13(v, q_val, vth=1.0)

Fluid moments
    compute_moments(f, v)

Linear wave physics
    bohm_gross(k_lam)
    landau_rate(k_lam)
    Z_func(zeta)
    Z_prime(zeta)

PIC noise
    pic_noise_scaling(N_vals, n_bins=60, seed=42)

Semi-Lagrangian Vlasov-Poisson solver
    solve_poisson(f, v, x)
    advect_x(f, v, dt, x)
    advect_v(f, E, dt, v)
    run_vlasov(f0, x, v, dt, n_steps, record_every=1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
EPS0 = 8.854_187_817e-12   # F m^{-1}  permittivity of free space
KB   = 1.380_649e-23       # J K^{-1}  Boltzmann constant
E_C  = 1.602_176_634e-19   # C         elementary charge
ME   = 9.109_383_7015e-31  # kg        electron mass


# ---------------------------------------------------------------------------
# Plasma parameters  (SI inputs)
# ---------------------------------------------------------------------------

def debye_length(n0, Te_eV):
    """
    Debye length [m] for electron density n0 [m^{-3}] and
    electron temperature Te_eV [eV].
    """
    Te = Te_eV * E_C
    return np.sqrt(EPS0 * Te / (n0 * E_C**2))


def plasma_parameter(n0, Te_eV):
    """
    Plasma parameter Lambda = n0 * lambda_D^3.
    Must satisfy Lambda >> 1 for the Vlasov mean-field limit to be valid.
    """
    lam_d = debye_length(n0, Te_eV)
    return n0 * lam_d**3


def plasma_frequency(n0):
    """Electron plasma frequency omega_pe [rad s^{-1}] for density n0 [m^{-3}]."""
    return np.sqrt(n0 * E_C**2 / (EPS0 * ME))


# ---------------------------------------------------------------------------
# Distribution functions  (normalised units)
# ---------------------------------------------------------------------------

def maxwellian(v, vth=1.0):
    """
    1-D Maxwellian distribution normalised to unit density.

    Parameters
    ----------
    v   : array_like  velocity grid
    vth : float       thermal speed (default 1.0)

    Returns
    -------
    f0 : ndarray same shape as v
    """
    v = np.asarray(v, dtype=float)
    return np.exp(-v**2 / (2.0 * vth**2)) / (np.sqrt(2.0 * np.pi) * vth)


def bump_on_tail(v, n_beam, v_beam, sigma_beam):
    """
    Bump-on-tail distribution: background Maxwellian + warm beam.

        f = (1 - n_beam) * M(v, vth=1) + n_beam * M(v - v_beam, sigma_beam)

    Parameters
    ----------
    v          : array_like  velocity grid
    n_beam     : float       beam fraction  0 < n_beam < 1
    v_beam     : float       beam drift velocity
    sigma_beam : float       beam thermal spread

    Returns
    -------
    f0  : ndarray  distribution function
    df0 : ndarray  velocity derivative  df0/dv
    """
    v      = np.asarray(v, dtype=float)
    f_bg   = (1.0 - n_beam) * maxwellian(v, vth=1.0)
    f_beam = n_beam * maxwellian(v - v_beam, vth=sigma_beam)
    f0     = f_bg + f_beam
    df0    = np.gradient(f0, v)
    return f0, df0


def schamel_hole(x, v, phi_amp, beta, vth=1.0):
    """
    Schamel phase-space electron hole on a 2-D (x, v) grid.

    The potential hump is phi(x) = phi_amp * exp(-x^2 / 0.6).
    Trapped particles (W < 0, W = v^2/2 - phi) follow a modified Maxwellian
    parameterised by trapping parameter beta.

    Parameters
    ----------
    x       : 1-D array  spatial grid
    v       : 1-D array  velocity grid
    phi_amp : float      potential amplitude [kBT/e]
    beta    : float      trapping parameter (>0 -> electron hole, depletion)
    vth     : float      thermal velocity (default 1.0)

    Returns
    -------
    f_hole : ndarray (Nx, Nv)
    phi    : ndarray (Nx, Nv)  potential field
    W      : ndarray (Nx, Nv)  particle energy in wave frame
    """
    X, V   = np.meshgrid(x, v, indexing='ij')
    phi    = phi_amp * np.exp(-X**2 / 0.6)
    W      = 0.5 * V**2 - phi
    norm   = np.sqrt(2.0 * np.pi) * vth
    f_free = np.exp(-0.5 * V**2 / vth**2) / norm
    f_trap = np.exp(-beta * 0.5 * V**2 / vth**2) / norm
    f_hole = np.where(W < 0, f_trap, f_free)
    return f_hole, phi, W


def grad13(v, q_val, vth=1.0):
    """
    Grad 13-moment distribution function (1-D Hermite projection).

        f_13 = f_M * [1 + (q/3) * H3(v/vth) / sqrt(2*pi)]

    where H3(c) = c^3 - 3c is the third physicists' Hermite polynomial.

    Parameters
    ----------
    v     : array_like  velocity grid
    q_val : float       normalised parallel heat flux
    vth   : float       thermal speed (default 1.0)

    Returns
    -------
    f13 : ndarray  (may go negative at large |v| when |q_val| is large)
    """
    v  = np.asarray(v, dtype=float)
    fM = maxwellian(v, vth=vth)
    c  = v / vth
    H3 = c**3 - 3.0 * c
    return fM * (1.0 + (q_val / 3.0) * H3 / np.sqrt(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Fluid moments
# ---------------------------------------------------------------------------

def compute_moments(f, v):
    """
    Compute the first four velocity moments of a 1-D distribution.

    Parameters
    ----------
    f : 1-D array  f(v)
    v : 1-D array  velocity grid

    Returns
    -------
    dict with keys:
        n  density (0th moment)
        u  bulk velocity (1st moment / n)
        T  temperature  (2nd central moment)
        q  heat flux    (3rd central moment)
    """
    n = float(np.trapezoid(f, v))
    u = float(np.trapezoid(v * f, v)) / n
    T = float(np.trapezoid((v - u)**2 * f, v)) / n
    q = float(np.trapezoid((v - u)**3 * f, v))
    return {"n": n, "u": u, "T": T, "q": q}


# ---------------------------------------------------------------------------
# Linear wave physics
# ---------------------------------------------------------------------------

def bohm_gross(k_lam):
    """
    Bohm-Gross (Langmuir wave) dispersion in normalised units.

        omega^2 = omega_pe^2 + 3 * k^2 * vth^2
     => omega / omega_pe = sqrt(1 + 3*(k*lambda_D)^2)

    Parameters
    ----------
    k_lam : array_like  k * lambda_D

    Returns
    -------
    omega   : ndarray  omega / omega_pe
    v_phi   : ndarray  phase velocity v_phi / vth
    v_group : ndarray  group velocity v_g / vth
    """
    k_lam   = np.asarray(k_lam, dtype=float)
    omega   = np.sqrt(1.0 + 3.0 * k_lam**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_phi = np.where(k_lam > 0, omega / k_lam, np.inf)
    v_group = 3.0 * k_lam / omega
    return omega, v_phi, v_group


def landau_rate(k_lam):
    """
    Analytic Landau damping rate for a Maxwellian plasma (long-wavelength limit).

        gamma / omega_pe = -sqrt(pi/8) / (k*lambda_D)^3
                           * exp(-1/(2*(k*lambda_D)^2) - 3/2)

    Parameters
    ----------
    k_lam : array_like  k * lambda_D  (valid for k*lambda_D in [0.1, 0.5])

    Returns
    -------
    gamma : ndarray  gamma / omega_pe  (negative = damping)
    """
    k_lam = np.asarray(k_lam, dtype=float)
    return (-np.sqrt(np.pi / 8.0) / k_lam**3
            * np.exp(-1.0 / (2.0 * k_lam**2) - 1.5))


def Z_func(zeta):
    """
    Plasma dispersion function Z(zeta) = i*sqrt(pi)*w(zeta).

    w(z) is the Faddeeva function (scipy.special.wofz).

    Parameters
    ----------
    zeta : complex scalar or array

    Returns
    -------
    Z : complex ndarray
    """
    return 1j * np.sqrt(np.pi) * wofz(np.asarray(zeta, dtype=complex))


def Z_prime(zeta):
    """
    Derivative of the plasma dispersion function.

    Identity: Z'(zeta) = -2 * (1 + zeta * Z(zeta))

    Parameters
    ----------
    zeta : complex scalar or array

    Returns
    -------
    dZ : complex ndarray
    """
    zeta = np.asarray(zeta, dtype=complex)
    return -2.0 * (1.0 + zeta * Z_func(zeta))


# ---------------------------------------------------------------------------
# PIC shot-noise benchmark
# ---------------------------------------------------------------------------

def pic_noise_scaling(N_vals, n_bins=60, seed=42):
    """
    Measure PIC shot noise (RMS deviation from exact Maxwellian) vs N.

    Parameters
    ----------
    N_vals : array_like  particle counts to benchmark
    n_bins : int         histogram bins (default 60)
    seed   : int         RNG seed for reproducibility (default 42)

    Returns
    -------
    noise : 1-D ndarray  RMS noise for each N
    slope : float        log-log slope (theoretical value = -0.5)
    """
    rng   = np.random.default_rng(seed)
    noise = []
    for N in np.asarray(N_vals, dtype=int):
        pts  = rng.standard_normal(int(N))
        h, edges = np.histogram(pts, bins=n_bins, density=True)
        vc   = 0.5 * (edges[:-1] + edges[1:])
        f_ex = maxwellian(vc, vth=1.0)
        noise.append(float(np.std(h - f_ex)))
    noise = np.array(noise)
    slope = float(np.polyfit(np.log10(N_vals), np.log10(noise), 1)[0])
    return noise, slope


# ---------------------------------------------------------------------------
# Semi-Lagrangian Vlasov-Poisson solver
# ---------------------------------------------------------------------------

def solve_poisson(f, v, x):
    """
    Spectral Poisson solver for a 1-D periodic domain.

    Solves  d^2 phi/dx^2 = -(n - 1)  with n = integral(f, v).
    Returns E(x) = -d phi/dx.

    Parameters
    ----------
    f : ndarray (Nx, Nv)  distribution function
    v : 1-D array          velocity grid
    x : 1-D array          spatial grid (uniform, periodic)

    Returns
    -------
    E : 1-D array (Nx,)  electric field
    """
    rho     = np.trapezoid(f, v, axis=1)
    kx      = np.fft.rfftfreq(len(x), d=x[1] - x[0]) * 2.0 * np.pi
    rho_hat = np.fft.rfft(rho - 1.0)
    kx[0]   = 1.0               # avoid DC singularity
    phi_hat = rho_hat / kx**2
    kx[0]   = 0.0
    phi_hat[0] = 0.0            # zero mean potential
    return np.fft.irfft(-1j * kx * phi_hat, n=len(x))


def advect_x(f, v, dt, x):
    """
    Semi-Lagrangian x-advection: shift each velocity slice by v*dt (periodic).

    Uses cubic-spline interpolation with periodic boundary closure.

    Parameters
    ----------
    f  : ndarray (Nx, Nv)
    v  : 1-D array  velocity grid
    dt : float      time step
    x  : 1-D array  spatial grid (uniform, endpoint=False)

    Returns
    -------
    f_new : ndarray (Nx, Nv)
    """
    lx_dom = x[-1] - x[0] + (x[1] - x[0])
    f_new  = np.empty_like(f)
    xe     = np.append(x, x[-1] + (x[1] - x[0]))   # periodic extension
    for j in range(f.shape[1]):
        x_orig      = (x - v[j] * dt) % lx_dom
        fe          = np.append(f[:, j], f[0, j])
        f_new[:, j] = CubicSpline(xe, fe)(x_orig)
    return f_new


def advect_v(f, E, dt, v):
    """
    Semi-Lagrangian v-advection: shift each spatial slice by E*dt.

    Uses natural cubic-spline interpolation; sets f=0 outside velocity domain.

    Parameters
    ----------
    f  : ndarray (Nx, Nv)
    E  : 1-D array (Nx,)  electric field at each grid point
    dt : float            time step
    v  : 1-D array        velocity grid

    Returns
    -------
    f_new : ndarray (Nx, Nv)
    """
    f_new = np.empty_like(f)
    for i in range(f.shape[0]):
        v_orig      = v - E[i] * dt
        vals        = CubicSpline(v, f[i, :], extrapolate=False)(v_orig)
        f_new[i, :] = np.where(np.isnan(vals), 0.0, vals)
    return f_new


def run_vlasov(f0, x, v, dt, n_steps, record_every=1):
    """
    Run the Strang (Cheng-Knorr) splitting Vlasov-Poisson integrator.

    Each time step follows the sequence:
        advect_x (dt/2)  ->  solve_poisson  ->  advect_v (dt)
        ->  solve_poisson  ->  advect_x (dt/2)

    Parameters
    ----------
    f0           : ndarray (Nx, Nv)  initial distribution
    x            : 1-D array         spatial grid
    v            : 1-D array         velocity grid
    dt           : float             time step
    n_steps      : int               number of time steps
    record_every : int               sample interval for diagnostics (default 1)

    Returns
    -------
    f        : ndarray (Nx, Nv)  final distribution
    t_hist   : 1-D ndarray       recorded times
    emax_hist: 1-D ndarray       recorded max |E| values
    """
    f         = f0.copy()
    t_hist    = []
    emax_hist = []

    for n in range(n_steps):
        E = solve_poisson(f, v, x)
        if n % record_every == 0:
            t_hist.append(n * dt)
            emax_hist.append(float(np.max(np.abs(E))))

        f = advect_x(f, v, dt / 2.0, x)
        E = solve_poisson(f, v, x)
        f = advect_v(f, E, dt, v)
        f = advect_x(f, v, dt / 2.0, x)

    return f, np.array(t_hist), np.array(emax_hist)

def plot_phase_space(f, x, v, title="Electron Phase Space", v_frame=0.0):
    """
    Plots the electron phase space distribution.
    
    Parameters
    ----------
    f       : ndarray (Nx, Nv)  distribution function
    x       : 1-D array         spatial grid
    v       : 1-D array         velocity grid
    v_frame : float             velocity of the observation frame (default 0.0). 
                                Set to EH speed to see the co-moving frame.
    """
    X, V = np.meshgrid(x, v - v_frame, indexing='ij')
    
    plt.figure(figsize=(10, 5))
    # Transpose f for plotting so x is horizontal and v is vertical
    cmap_plot = plt.pcolormesh(X, V, f, shading='auto', cmap='jet')
    plt.colorbar(cmap_plot, label='Distribution Function')
    plt.title(title)
    plt.xlabel("Spatial direction (x)")
    plt.ylabel("Velocity ($v - v_{frame}$)")
    plt.tight_layout()
    plt.show()

def plot_potential_and_field(phi, E, x, title="Snapshot"):
    """
    Plots the electrostatic potential and electric field profiles.
    
    Parameters
    ----------
    phi : 1-D array (Nx,)  electrostatic potential
    E   : 1-D array (Nx,)  electric field
    x   : 1-D array        spatial grid
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(x, phi, 'k-', linewidth=2)
    ax1.set_ylabel(r"Electric Potential ($\phi$)")
    ax1.set_title(title)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(x, E, 'k-', linewidth=2)
    ax2.set_ylabel(r"Electric Field ($E_x$)")
    ax2.set_xlabel("Spatial direction (x)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Moving Electron Hole Initializations 
# ---------------------------------------------------------------------------

def create_moving_hole(x, v, phi_amp, L, M, cs, x0, beta=-0.5, vth=428.5, Te_ratio=100.0):
    """
    Creates a generalized moving electron hole based on the paper's exact energy parameters.
    """
    v_EH = M * cs  
    X, V = np.meshgrid(x, v, indexing='ij')
    
    # 1. Potential Profile
    phi = phi_amp * (1.0 / np.cosh((X - x0) / L))**2
    
    # 2. Shifted velocities and energies (normalized to electron temperature)
    V_prime = V - v_EH
    eps_K_prime = 0.5 * (V_prime / vth)**2
    eps_phi = phi / Te_ratio  
    
    # 3. Free Particles (Shifted back to lab frame for the unperturbed Maxwellian)
    # Using energy conservation: v_sh_prime = sign(V') * sqrt(2 * (eps_K' - eps_phi)) * vth
    v_sh_prime = np.sign(V_prime) * vth * np.sqrt(np.maximum(eps_K_prime - eps_phi, 0.0) * 2.0)
    v_sh_lab = v_sh_prime + v_EH
    
    norm = np.sqrt(2.0 * np.pi) * vth
    f_free = np.exp(-0.5 * (v_sh_lab / vth)**2) / norm
    
    # 4. Trapped Particles (Continuous at the separatrix eps_K' == eps_phi)
    # At separatrix, v_sh_prime = 0, so v_sh_lab = v_EH.
    f_base = np.exp(-0.5 * (v_EH / vth)**2) / norm
    f_trap = f_base * np.exp(-beta * eps_K_prime)
    
    # 5. Composite Phase Space
    f_hole = np.where(eps_K_prime < eps_phi, f_trap, f_free)
    return f_hole, phi

def setup_head_on_collision(x, v, cs=17.43, vth=428.5):
    """
    Sets up the initial distribution for a head-on collision between EH1 and EH2.
    """
    # Create EH1 moving right
    f1, phi1 = create_moving_hole(x, v, phi_amp=19.0, L=22.5, M=40, cs=cs, x0=350, vth=vth)
    
    # Create EH2 moving left
    f2, phi2 = create_moving_hole(x, v, phi_amp=9.5, L=22.5, M=-40, cs=cs, x0=800, vth=vth)
    
    # Background Maxwellian distribution scaled by correct vth
    X, V = np.meshgrid(x, v, indexing='ij')
    f_bg = np.exp(-0.5 * (V / vth)**2) / (np.sqrt(2.0 * np.pi) * vth)
    
    # Superimpose
    f_total = f1 + f2 - f_bg
    phi_total = phi1 + phi2 
    
    return f_total, phi_total


def setup_overtaking_collision(x, v, cs=17.43, vth=428.5):
    """
    Sets up the initial distribution for an overtaking collision between EH1 and EH3.
    """
    # Create EH1 moving right (faster)
    f1, phi1 = create_moving_hole(x, v, phi_amp=19.0, L=22.5, M=40, cs=cs, x0=250, vth=vth)
    
    # Create EH3 moving right (slower)
    f3, phi3 = create_moving_hole(x, v, phi_amp=19.0, L=22.5, M=30, cs=cs, x0=450, vth=vth)
    
    # Background Maxwellian distribution scaled by correct vth
    X, V = np.meshgrid(x, v, indexing='ij')
    f_bg = np.exp(-0.5 * (V / vth)**2) / (np.sqrt(2.0 * np.pi) * vth)
    
    f_total = f1 + f3 - f_bg
    phi_total = phi1 + phi3
    
    return f_total, phi_total

def _visualizer_maxwellian(v, amplitude, drift, thermal_spread):
    """Internal helper to calculate a 1D Maxwellian for the visualizer."""
    return (amplitude / (np.sqrt(2 * np.pi) * thermal_spread)) * \
           np.exp(-0.5 * ((v - drift) / thermal_spread)**2)

def plot_instability_grid(mode='bump'):
    """
    Plots a 2x2 grid showing the evolution of the velocity distribution
    and the unstable positive-slope regions for different drift velocities.
    """
    v = np.linspace(-8, 8, 1000)
    
    # Setup a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    mode_title = "Bump-on-Tail" if mode == 'bump' else "Two-Stream"
    fig.suptitle(f'Evolution of {mode_title} Instability', fontsize=16, y=0.95)
    
    # Define the parameter sweep for Drift Velocity (u_b)
    if mode == 'bump':
        drift_vals = [2.0, 3.0, 4.0, 5.0]
        density, spread = 0.15, 0.5
    else: 
        # Two-stream requires slightly lower drift velocities to see the merge
        drift_vals = [1.0, 2.0, 3.0, 4.0]
        density, spread = 0.5, 0.5 # Density is fixed 50/50 for two-stream

    # Flatten the 2x2 axes array for easy iteration
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        drift = drift_vals[i]
        
        # Calculate f(v)
        if mode == 'bump':
            f = _visualizer_maxwellian(v, 1.0 - density, 0.0, 1.0) + \
                _visualizer_maxwellian(v, density, drift, spread)
        else:
            f = _visualizer_maxwellian(v, 0.5, drift, spread) + \
                _visualizer_maxwellian(v, 0.5, -drift, spread)
            
        # Calculate derivative and find unstable regions
        dfdv = np.gradient(f, v)
        if mode == 'bump':
            unstable_mask = (v > 0.2) & (dfdv > 0.001)
        else:
            unstable_mask = ((v > 0.2) & (dfdv > 0.001)) | ((v < -0.2) & (dfdv < -0.001))
            
        # Plotting
        ax.plot(v, f, color='#0056b3', linewidth=2)
        ax.fill_between(v, f, 0, where=unstable_mask, color='red', alpha=0.3, 
                        label='Unstable ($v \\cdot \\partial f/\\partial v > 0$)')
        
        # Formatting
        ax.set_title(f'Drift Velocity $u_b$ = {drift}', fontsize=12)
        ax.set_xlim(-8, 8)
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3)
        
        # Only add labels to the outer edges to keep it clean
        if i >= 2:
            ax.set_xlabel('Velocity ($v / v_{th}$)', fontsize=11)
        if i % 2 == 0:
            ax.set_ylabel('Distribution $f(v)$', fontsize=11)
            
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout to fit the suptitle
    plt.show()