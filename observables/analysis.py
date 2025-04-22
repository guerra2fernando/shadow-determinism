# --- START OF FILE analysis.py ---

# quantum_chaos_sim/observables/analysis.py
import numpy as np
from ..core.operators import kinetic_operator_momentum_space, second_derivative_finite_diff

def probability_density(psi):
    """Calculates the probability density |psi(x)|^2."""
    return np.abs(psi)**2

def calculate_norm(psi, dx):
    """Calculates the L2 norm: sqrt(integral |psi|^2 dx)."""
    return np.sqrt(np.sum(probability_density(psi)) * dx)

def check_norm_conservation(psi, dx, initial_norm=1.0):
    """Calculates the squared norm to check conservation (should be close to initial_norm^2, typically 1)."""
    current_norm_sq = np.sum(probability_density(psi)) * dx
    return current_norm_sq # Return squared norm for direct comparison to 1

def position_expectation(psi, x_grid, dx):
    """Calculates the expectation value of position <x>."""
    prob_dens = probability_density(psi)
    return np.sum(x_grid * prob_dens) * dx

def momentum_expectation(psi, k_grid, hbar):
    """
    Calculates the expectation value of momentum <p> using FFT.
    Assumes k_grid corresponds to the output of np.fft.fftfreq.
    """
    psi_k = np.fft.fft(psi)
    prob_dens_k = np.abs(psi_k)**2
    # Normalize probability density in k-space (optional but good practice for interpretation)
    norm_k_sq = np.sum(prob_dens_k)
    if norm_k_sq < 1e-15: return 0.0 # Avoid division by zero

    # Calculate expectation value <k> = sum(k * |psi_k|^2) / sum(|psi_k|^2)
    avg_k = np.sum(k_grid * prob_dens_k) / norm_k_sq
    # Return <p> = hbar * <k>
    return hbar * avg_k

def energy_expectation(psi, x_grid, V_xt, k_grid, dx, hbar, m):
    """
    Calculates the expectation value of energy <H> = <T> + <V>.
    Assumes psi is normalized in position space (integral |psi|^2 dx = 1).
    """
    # Potential Energy Expectation <V> = integral psi* V psi dx
    prob_dens = probability_density(psi)
    potential_energy = np.sum(V_xt * prob_dens) * dx

    # Kinetic Energy Expectation <T> = integral psi_k* T_k psi_k dk / integral |psi_k|^2 dk
    # Using FFT: <T> = sum[ T_k * |fft(psi)|^2 ] / sum[ |fft(psi)|^2 ]
    psi_k = np.fft.fft(psi)
    T_k = kinetic_operator_momentum_space(k_grid, hbar, m) # T(k) = (hbar*k)^2 / (2m)
    prob_dens_k = np.abs(psi_k)**2
    norm_k_sq = np.sum(prob_dens_k)

    if norm_k_sq < 1e-15:
        kinetic_energy = 0.0
    else:
        kinetic_energy = np.sum(T_k * prob_dens_k) / norm_k_sq

    # Total Energy <H> = <T> + <V>
    return kinetic_energy + potential_energy

def spatial_variance(psi, x_grid, dx):
    """Calculates the spatial variance Var(x) = <x^2> - <x>^2."""
    prob_dens = probability_density(psi)
    mean_x = np.sum(x_grid * prob_dens) * dx
    mean_x2 = np.sum(x_grid**2 * prob_dens) * dx
    variance = mean_x2 - mean_x**2
    # Variance should be non-negative, numerical errors might make it slightly negative
    return max(0.0, variance)

def shannon_entropy_spatial(psi, dx):
    """
    Calculates Shannon entropy S = - sum(p_i * log2(p_i)) of the spatial probability density.
    Uses base 2 logarithm (bits).
    """
    prob_dens = probability_density(psi)
    # Convert density to probability mass p_i = |psi_i|^2 * dx
    # Filter out zero probabilities to avoid log(0)
    p = prob_dens[prob_dens > 1e-18] * dx
    if len(p) == 0:
        return 0.0

    # Ensure probabilities sum to 1 (or close to it) before entropy calculation
    # This handles potential normalization drift if psi is not perfectly normalized
    p_sum = np.sum(p)
    if p_sum < 1e-15: # Check if sum is practically zero
        return 0.0
    p_normalized = p / p_sum # Normalize locally just for entropy

    # Calculate entropy: S = - sum(p * log2(p))
    entropy = -np.sum(p_normalized * np.log2(p_normalized))
    return entropy

# --- END OF FILE analysis.py ---