# Fast Diffusion Entropies (1D, Finite Volume)

This repo implements a 1D finite-volume implicit scheme (uniform mesh, Neumann BCs) for the fast diffusion equation (β = 1/2).  
It computes and plots the discrete entropies \( \log E_d^\alpha[u(t)] \) and \( \log F_d^\alpha[u(t)] \) vs time for several α, reproducing “Fig. 6”-style graphs.

## Features
- Fully implicit time stepping (backward Euler) + FV in space
- Newton with damping, nonnegativity clamp
- Discrete entropies \(E_d^\alpha\) and \(F_d^\alpha\) (1D FV versions)
- Minimal parameters to reproduce the plots (α ∈ {0.5, 1, 2, 6}, β=1/2)

## Quick start
```bash
python fast_diffusion_entropy_demo.py
