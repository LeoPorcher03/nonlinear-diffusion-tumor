"""
2D finite-volume entropy-dissipative scheme for u_t = \Delta(u^beta)
Implicit Euler in time with Newton solver. Uses dense linear algebra (suitable for small grids Nx*Ny <= ~2500).
Computes discrete entropies E_alpha and F_alpha and plots their decay and snapshots.

Usage: python entropy_scheme_2d.py
Requirements: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------ Discrete operators ------------------------

def laplacian_1d_matrix(n, h, bc='periodic'):
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i - 1 >= 0:
            T[i, i - 1] = 1.0
        else:
            if bc == 'periodic':
                T[i, n - 1] = 1.0
        if i + 1 < n:
            T[i, i + 1] = 1.0
        else:
            if bc == 'periodic':
                T[i, 0] = 1.0
        # interior second-diff coefficient
    if bc == 'neumann':
        # simple Neumann: reflect neighbors (one-sided second difference)
        # This construction yields a symmetric tridiagonal with modified corners
        for i in range(n):
            if i == 0:
                if n > 1:
                    T[0, 1] = 1.0
                    T[0, 0] = -1.0
            elif i == n - 1:
                T[n - 1, n - 2] = 1.0
                T[n - 1, n - 1] = -1.0
            else:
                T[i, i] = -2.0
        return T / (h * h)
    else:
        # periodic
        for i in range(n):
            T[i, i] = -2.0
        return T / (h * h)


def build_L_2d(Nx, Ny, dx, dy, bc='periodic'):
    # build 2D Laplacian as Kronecker sum: L = Ix \otimes Tx/dx^2 + Ty/dy^2 \otimes Iy
    Tx = laplacian_1d_matrix(Nx, dx, bc=bc)
    Ty = laplacian_1d_matrix(Ny, dy, bc=bc)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    # kron use: L = kron(Iy, Tx) + kron(Ty, Ix)
    L = np.kron(Iy, Tx) + np.kron(Ty, Ix)
    return L


def compute_E_alpha_2d(u, dx, dy, alpha):
    mass = np.sum(dx * dy * u)
    term = np.sum(dx * dy * (u ** (alpha + 1)))
    return (1.0 / (alpha + 1.0)) * (term - mass ** (alpha + 1.0))


def compute_F_alpha_2d(u, dx, dy, alpha, Nx, Ny, bc='periodic'):
    # sum over faces: horizontal and vertical
    tau_x = dy / dx
    tau_y = dx / dy
    s = 0.0
    U = u.reshape((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx - 1):
            s += tau_x * (U[j, i] ** (alpha / 2.0) - U[j, i + 1] ** (alpha / 2.0)) ** 2
    for j in range(Ny - 1):

        for i in range(Nx):
            s += tau_y * (U[j, i] ** (alpha / 2.0) - U[j + 1, i] ** (alpha / 2.0)) ** 2
    if bc == 'periodic':
        # periodic wrap in x
        for j in range(Ny):
            s += tau_x * (U[j, -1] ** (alpha / 2.0) - U[j, 0] ** (alpha / 2.0)) ** 2
        # periodic wrap in y
        for i in range(Nx):
            s += tau_y * (U[-1, i] ** (alpha / 2.0) - U[0, i] ** (alpha / 2.0)) ** 2
    # For Neumann, boundary faces are not counted (no-flux)
    return 0.5 * s


# ------------------------ Newton solver (implicit step) ------------------------

def implicit_step_newton_2d(u_old, L, dx, dy, dt, beta, Nx, Ny, tol_newton=1e-8, max_iter=30):
    N = u_old.size
    Mdiag = np.full(N, dx * dy)
    u = u_old.copy()

    for it in range(max_iter):
        v = u ** beta
        R = Mdiag * (u - u_old) / dt - L.dot(v)
        res_norm = np.linalg.norm(R)
        if res_norm < tol_newton:
            return u
        D = beta * (u ** (beta - 1.0))
        # J = diag(M/dt) - L * diag(D)
        J = np.diag(Mdiag / dt) - L.dot(np.diag(D))
        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            raise RuntimeError("Newton linear solve failed (singular Jacobian)")
        lam = 1.0
        u_new = u + lam * delta
        u_new[u_new <= 0] = 1e-14
        u = u_new
    raise RuntimeError(f"Newton did not converge after {max_iter} iterations, last res {res_norm}")


# ------------------------ Experiment runner ------------------------

def run_experiment_2d(
    beta=2.0,
    alpha=1.0,
    Lx=1.0,
    Ly=1.0,
    Nx=32,
    Ny=32,
    T=0.1,
    dt=1e-4,
    bc='periodic',
    plot=True,
    verbose=True,
):
    dx = Lx / Nx
    dy = Ly / Ny
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    X, Y = np.meshgrid(x, y)

    # initial condition: background + gaussian bump
    r2 = (X - Lx / 2.0) ** 2 + (Y - Ly / 2.0) ** 2
    u0 = 1.0 + 0.6 * np.exp(-r2 / (2 * (0.05 ** 2)))
    u = u0.copy().reshape(-1)

    L = build_L_2d(Nx, Ny, dx, dy, bc=bc)

    nt = int(np.ceil(T / dt))
    dt = T / nt

    times = [0.0]
    Elist = [compute_E_alpha_2d(u.reshape(-1), dx, dy, alpha)]
    Flist = [compute_F_alpha_2d(u.reshape(-1), dx, dy, alpha, Nx, Ny, bc=bc)]
    mass0 = np.sum(dx * dy * u)

    tstart = time.time()
    for k in range(nt):
        t = (k + 1) * dt
        u = implicit_step_newton_2d(u, L, dx, dy, dt, beta, Nx, Ny)
        if (k + 1) % max(1, nt // 40) == 0 or k == nt - 1:
            times.append(t)
            Elist.append(compute_E_alpha_2d(u.reshape(-1), dx, dy, alpha))
            Flist.append(compute_F_alpha_2d(u.reshape(-1), dx, dy, alpha, Nx, Ny, bc=bc))
        if (k + 1) % max(1, nt // 8) == 0 and verbose:
            mass = np.sum(dx * dy * u)
            if abs(mass - mass0) > 1e-8:
                print(f"[step {k+1}] mass deviation = {mass - mass0:.2e}")
    tend = time.time()

    if verbose:
        print(f"Ran {nt} steps (2D), final time {T:.4f}, elapsed {tend - tstart:.3f}s")
        print(f"Initial E = {Elist[0]:.5e}, final E = {Elist[-1]:.5e}")

    if plot:
        U_final = u.reshape((Ny, Nx))
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(u0, origin='lower', extent=(0, Lx, 0, Ly))
        plt.title('u0')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(U_final, origin='lower', extent=(0, Lx, 0, Ly))
        plt.title('u(T)')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.loglog(times, np.maximum(Elist, 1e-20), '-o', label=f'E_alpha (alpha={alpha})')
        plt.loglog(times, np.maximum(Flist, 1e-20), '-s', label=f'F_alpha (alpha={alpha})')
        plt.xlabel('t')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return X, Y, u.reshape((Ny, Nx)), times, Elist, Flist


if __name__ == '__main__':
    run_experiment_2d(beta=2.0, alpha=1.0, Nx=32, Ny=32, T=0.05, dt=5e-5, bc='periodic')
