#!/usr/bin/env python3
# Minimal reproducible script to simulate 1D fast diffusion (β=1/2) with a finite-volume implicit scheme,
# then compute and plot log E_d and log F_d for α ∈ {0.5, 1, 2, 6} as in Fig. 6 (d=1).
#
# Usage:
#   python fast_diffusion_entropy_demo.py
#
# Dependencies: numpy, matplotlib
#
# Notes:
# - Scheme: fully implicit backward Euler + finite volumes (uniform mesh), Neumann BCs
# - Discrete entropies per Chainais–Jüngel–Schuchnigg (2015): Eq. (8) and (9)
# - Parameters: Ω=(0,1), N=50 cells (Δx=0.02), Δt=2e-4, T=0.1, β=1/2
# - Initial datum (fast diffusion case in the paper): u0(x) = C * ((x0 - x)(x - x1))_+^2 with x0=0.3, x1=0.7, C=3000
#
# The code is intentionally compact but numerically robust enough for a demo:
# - Newton method with damping
# - Regularization near 0 for u^{β-1}
# - Mass is approximately conserved by the scheme (FV + Neumann)
#
# You can adapt N, dt, T, alphas below if needed.
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
beta = 0.5
x0, x1 = 0.3, 0.7
C = 3000.0
T = 0.1
dt = 2e-4
N = 50  # number of control volumes
L = 1.0
dx = L / N
M = int(T / dt)

# Alphas to reproduce figure
alphas = [0.5, 1.0, 2.0, 6.0]

# Spatial grid: cell centers
xc = (np.arange(N) + 0.5) * dx

# Initial data: truncated polynomial (._+ means positive part)
def u0_fun(x):
    z = (x0 - x) * (x - x1)
    z[z < 0] = 0.0
    return C * z**2

u = u0_fun(xc)

# Storage for entropies over time
times = np.arange(M + 1) * dt
E_hist = {a: np.zeros(M + 1) for a in alphas}
F_hist = {a: np.zeros(M + 1) for a in alphas}

def compute_Ed(u, alpha):
    # E_d = 1/(α+1) * ( ∑ m(K) u_K^{α+1} - (∑ m(K) u_K)^{α+1} )
    s1 = np.sum(dx * u**(alpha + 1))
    s0 = np.sum(dx * u)
    return (s1 - s0**(alpha + 1)) / (alpha + 1)

def compute_Fd(u, alpha):
    # F_d = 1/2 * | (u^{α/2}) |_{1,2,T}^2, with |.| seminorm using 1D FV: sum (m(σ)/dσ) (jump)^2 = (1/dx) ∑ (Δf)^2
    f = np.maximum(u, 0.0)**(alpha / 2.0)
    jumps = f[1:] - f[:-1]
    seminorm_sq = (1.0 / dx) * np.sum(jumps**2)
    return 0.5 * seminorm_sq

# Record initial entropies
for a in alphas:
    E_hist[a][0] = compute_Ed(u, a)
    F_hist[a][0] = compute_Fd(u, a)

# Helper: residual and Jacobian for Newton (implicit FV scheme)
def residual_and_jacobian(u_new, u_old, dt, dx, beta, eps=1e-12):
    N = u_new.size
    D = dt / (dx * dx)
    r = np.zeros(N)
    # Tridiagonal Jacobian
    main = np.ones(N)
    lower = np.zeros(N - 1)
    upper = np.zeros(N - 1)
    # Precompute powers safely
    u_eps = np.maximum(u_new, 0.0) + eps
    pow_beta = u_eps**beta
    d_beta = beta * u_eps**(beta - 1.0)
    # i = 0 (left boundary, one neighbor due to Neumann)
    r[0] = u_new[0] - u_old[0] + D * (pow_beta[0] - pow_beta[1])
    main[0] = 1.0 + D * d_beta[0]
    upper[0] = -D * d_beta[1]
    # interior i
    for i in range(1, N - 1):
        r[i] = u_new[i] - u_old[i] + D * (2.0 * pow_beta[i] - pow_beta[i - 1] - pow_beta[i + 1])
        main[i] = 1.0 + 2.0 * D * d_beta[i]
        lower[i - 1] = -D * d_beta[i - 1]
        upper[i] = -D * d_beta[i + 1]
    # i = N-1 (right boundary)
    r[N - 1] = u_new[N - 1] - u_old[N - 1] + D * (pow_beta[N - 1] - pow_beta[N - 2])
    main[N - 1] = 1.0 + D * d_beta[N - 1]
    lower[N - 2] = -D * d_beta[N - 2]
    return r, main, lower, upper

def solve_tridiag(main, lower, upper, rhs):
    # Thomas algorithm
    n = main.size
    c = upper.copy()
    d = rhs.copy()
    a = lower.copy()
    b = main.copy()
    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x

# Time stepping with Newton + damping
for k in range(1, M + 1):
    u_old = u.copy()
    u_new = u.copy()
    # Newton iterations
    for it in range(50):
        r, main, lower, upper = residual_and_jacobian(u_new, u_old, dt, dx, beta)
        norm_r = np.linalg.norm(r, ord=np.inf)
        if norm_r < 1e-10:
            break
        # Solve J * delta = -r
        delta = solve_tridiag(main, lower, upper, -r)
        # Damping / line search
        lam = 1.0
        for _ in range(10):
            trial = np.maximum(u_new + lam * delta, 0.0)  # enforce nonnegativity
            r_trial, *_ = residual_and_jacobian(trial, u_old, dt, dx, beta)
            if np.linalg.norm(r_trial, ord=np.inf) < norm_r:
                u_new = trial
                break
            lam *= 0.5
        else:
            # If no improvement, accept small step
            u_new = np.maximum(u_new + 0.1 * delta, 0.0)
    u = u_new
    # Record entropies
    for a in alphas:
        E_hist[a][k] = compute_Ed(u, a)
        F_hist[a][k] = compute_Fd(u, a)

# Plot: log(E_d) vs time
plt.figure(figsize=(7, 4.5))
for a in alphas:
    # Avoid log(0) by clipping to tiny positive
    y = np.maximum(E_hist[a], 1e-300)
    plt.plot(times, np.log(y), label=f"α = {a:g}")
plt.xlabel("time")
plt.ylabel("log(E_d_α[u](t))")
plt.legend()
plt.title("Fast diffusion (β=1/2, d=1): log E_d vs time")
plt.tight_layout()
plt.show()

# Plot: log(F_d) vs time
plt.figure(figsize=(7, 4.5))
for a in alphas:
    y = np.maximum(F_hist[a], 1e-300)
    plt.plot(times, np.log(y), label=f"α = {a:g}")
plt.xlabel("time")
plt.ylabel("log(F_d_α[u](t))")
plt.legend()
plt.title("Fast diffusion (β=1/2, d=1): log F_d vs time")
plt.tight_layout()
plt.show()
