# quantum_chaos_sim/core/operators.py
import numpy as np

def kinetic_operator_momentum_space(k_grid, hbar, m):
    """Returns the kinetic energy operator T = p^2/(2m) in momentum space."""
    # p = hbar * k
    return (hbar**2 * k_grid**2) / (2 * m)

def potential_operator_position_space(V_xt):
    """Returns the potential energy operator V(x, t) in position space."""
    # In position space, the potential operator is just multiplication
    return V_xt

# --- Finite Difference Operators (Example, use FFT method for accuracy/stability) ---
def second_derivative_finite_diff(psi, dx):
    """Computes second derivative using centered finite differences."""
    d2psi_dx2 = np.zeros_like(psi, dtype=complex)
    # Use np.roll for periodic boundary conditions
    psi_p1 = np.roll(psi, -1)
    psi_m1 = np.roll(psi, 1)
    d2psi_dx2 = (psi_p1 - 2 * psi + psi_m1) / dx**2
    return d2psi_dx2

def kinetic_operator_position_space_fd(psi, dx, hbar, m):
    """Applies kinetic operator using finite difference (less preferred)."""
    d2psi_dx2 = second_derivative_finite_diff(psi, dx)
    return (-hbar**2 / (2 * m)) * d2psi_dx2