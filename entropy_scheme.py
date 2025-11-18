"""
Finite-volume entropy-dissipative scheme (1D) for u_t = \Delta(u^beta)
Implements implicit Euler in time and a Newton solver for the nonlinear system.
Computes discrete entropies E_alpha and F_alpha and plots their decay.

Usage: run the script with Python 3 (needs numpy and matplotlib).
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------ Numerical helpers ------------------------

def build_A_1d(N, dx, bc='periodic'):
    """Build matrix A such that (A v)_i = sum_sigma tau_sigma (v_i - v_j)
    For uniform 1D finite volumes with transmissibility tau = 1/dx.
    Returns dense NxN array (sufficient for moderate N).
    bc: 'periodic' or 'neumann'
    """
    tau = 1.0 / dx
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        # left neighbor
        if i - 1 >= 0:
            j = i - 1
            A[i, i] += tau
            A[i, j] -= tau
        else:
            if bc == 'periodic':
                j = N - 1
                A[i, i] += tau
                A[i, j] -= tau
            # Neumann: no flux -> nothing added
        # right neighbor
        if i + 1 < N:
            j = i + 1
            A[i, i] += tau
            A[i, j] -= tau
        else:
            if bc == 'periodic':
                j = 0
                A[i, i] += tau
                A[i, j] -= tau
            # Neumann: no flux
    return A


def compute_E_alpha(u, dx, alpha):
    mass = np.sum(dx * u)
    term = np.sum(dx * (u ** (alpha + 1)))
    return (1.0 / (alpha + 1.0)) * (term - mass ** (alpha + 1.0))


def compute_F_alpha(u, dx, alpha, bc='periodic'):
    # F_alpha = 0.5 * sum_sigma tau_sigma (u_K^{alpha/2} - u_L^{alpha/2})^2
    N = u.size
    dx_local = dx
    tau = 1.0 / dx_local
    s = 0.0
    for i in range(N - 1):
        s += tau * (u[i] ** (alpha / 2.0) - u[i + 1] ** (alpha / 2.0)) ** 2
    if bc == 'periodic':
        s += tau * (u[-1] ** (alpha / 2.0) - u[0] ** (alpha / 2.0)) ** 2
    # Neumann: no boundary face at domain ends (so only internal faces counted)
    return 0.5 * s


# ------------------------ Time-stepping (Newton) ------------------------

def implicit_step_newton(u_old, A, dx, dt, beta, tol_newton=1e-8, max_iter=30):
    """Solve R(u) = M*(u - u_old)/dt + A*(u**beta) = 0 using Newton's method.
    M is diagonal with entries dx.
    Returns u_new.
    """
    N = u_old.size
    Mdiag = np.full(N, dx)

    # initial guess: previous time level
    u = u_old.copy()

    for it in range(max_iter):
        v = u ** beta
        R = Mdiag * (u - u_old) / dt + A.dot(v)
        res_norm = np.linalg.norm(R)
        if res_norm < tol_newton:
            # converged
            # print(f"Newton iter {it}, res {res_norm:.2e}")
            return u
        # Jacobian J = diag(M/dt) + A * diag(beta * u^{beta-1})
        diag_term = Mdiag / dt
        D = beta * (u ** (beta - 1.0))
        J = np.diag(diag_term) + A.dot(np.diag(D))
        # Solve for Newton step J * delta = -R
        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            raise RuntimeError("Linear solve in Newton failed (singular Jacobian)")
        # damping for robustness
        lam = 1.0
        u_new = u + lam * delta
        # enforce positivity
        u_new[u_new <= 0] = 1e-14
        u = u_new
    raise RuntimeError(f"Newton did not converge after {max_iter} iterations, last res {res_norm}")


# ------------------------ Main experiment ------------------------

def run_experiment(
    beta=2.0,
    alpha=1.0,
    L=1.0,
    Nx=100,
    T=0.5,
    dt=1e-3,
    bc='periodic',
    plot=True,
    verbose=True,
):
    dx = L / Nx
    x = (np.arange(Nx) + 0.5) * dx

    # initial condition: small gaussian + background for positivity
    u0 = 1.0 + 0.5 * np.exp(-((x - 0.5 * L) ** 2) / (2 * 0.03 ** 2))
    u = u0.copy()

    A = build_A_1d(Nx, dx, bc=bc)

    nt = int(np.ceil(T / dt))
    dt = T / nt

    times = [0.0]
    Elist = [compute_E_alpha(u, dx, alpha)]
    Flist = [compute_F_alpha(u, dx, alpha, bc=bc)]
    mass0 = np.sum(dx * u0)

    tstart = time.time()
    for k in range(nt):
        t = (k + 1) * dt
        u = implicit_step_newton(u, A, dx, dt, beta)
        if (k + 1) % max(1, nt // 50) == 0 or k == nt - 1:
            times.append(t)
            Elist.append(compute_E_alpha(u, dx, alpha))
            Flist.append(compute_F_alpha(u, dx, alpha, bc=bc))
        # quick mass check every few steps
        if (k + 1) % max(1, nt // 10) == 0 and verbose:
            mass = np.sum(dx * u)
            if abs(mass - mass0) > 1e-8:
                print(f"[step {k+1}] mass deviation = {mass - mass0:.2e}")
    tend = time.time()

    if verbose:
        print(f"Ran {nt} steps, final time {T:.4f}, elapsed {tend - tstart:.3f}s")
        print(f"Initial E = {Elist[0]:.5e}, final E = {Elist[-1]:.5e}")

    if plot:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, u0, label='u0')
        plt.plot(x, u, label='u(T)')
        plt.xlabel('x')
        plt.legend()
        plt.title(f'u at T, beta={beta}')

        plt.subplot(1, 2, 2)
        plt.loglog(times, np.maximum(Elist, 1e-20), '-o', label=f'E_alpha (alpha={alpha})')
        plt.loglog(times, np.maximum(Flist, 1e-20), '-s', label=f'F_alpha (alpha={alpha})')
        plt.xlabel('t')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return x, u, times, Elist, Flist


# ------------------------ Run sample if executed directly ------------------------
if __name__ == '__main__':
    # Example parameters similar to porous medium beta>1
    beta = 2.0
    alpha = 1.0
    Nx = 80
    T = 0.2
    dt = 2e-4
    run_experiment(beta=beta, alpha=alpha, Nx=Nx, T=T, dt=dt, bc='periodic')
