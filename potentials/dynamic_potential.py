# --- START OF FILE quantum_chaos_sim/potentials/dynamic_potential.py ---

import numpy as np
import logging

logger = logging.getLogger(__name__)

def dynamic_potential(x_grid, t, z_t, V0_static, epsilon, k_potential, omega_potential, alpha, feedback_adjustment=0.0):
    """
    Calculates the potential V(x, t) modulated by the external driving signal z(t).
    Now handles potentially time-dependent epsilon and alpha values (passed directly).

    Args:
        x_grid (np.ndarray): Spatial grid points.
        t (float): Current time.
        z_t (float): Value of the external driving signal at time t.
        V0_static (float or np.ndarray): Base static potential (can be spatially dependent).
        epsilon (float): Current strength of the intrinsic potential modulation at time t.
        k_potential (float): Spatial frequency in potential modulation.
        omega_potential (float): Temporal frequency in potential modulation (intrinsic).
        alpha (float): Current coupling strength to the external driving signal z(t) at time t.
        feedback_adjustment (float, optional): Additive term to the potential from feedback control. Defaults to 0.0.

    Returns:
        np.ndarray: Potential values on the x_grid at time t.
    """
    # Ensure V0 is broadcastable if it's a scalar
    if isinstance(V0_static, (int, float)):
        V0 = np.full_like(x_grid, V0_static)
    elif isinstance(V0_static, np.ndarray) and V0_static.shape == x_grid.shape:
        V0 = V0_static
    else:
        logger.error(f"V0_static must be a scalar or a numpy array matching x_grid shape. Got type {type(V0_static)} and shape {getattr(V0_static, 'shape', 'N/A')}")
        # Return a zero potential or raise error? Let's return zero for now.
        return np.zeros_like(x_grid)

    # Check if z_t is a valid number (it should be a float passed from the main loop)
    if not isinstance(z_t, (int, float, np.number)):
        logger.warning(f"dynamic_potential received non-numeric z_t value (type: {type(z_t)}, value: {z_t}) at t={t:.2f}. Using z_t = 0.")
        z_t = 0.0
    elif not np.isfinite(z_t):
        logger.warning(f"dynamic_potential received non-finite z_t value ({z_t}) at t={t:.2f}. Using z_t = 0.")
        z_t = 0.0

    # Current alpha and epsilon are passed directly (potentially looked up from a schedule in main.py)
    current_alpha = alpha
    current_epsilon = epsilon

    # Calculate the combined modulation term using the current parameters
    # V(x,t) = V0 + epsilon(t) * sin( k*x + omega*t + alpha(t)*z_t ) + feedback
    combined_modulation = current_epsilon * np.sin(k_potential * x_grid + omega_potential * t + current_alpha * z_t)

    # Total potential including static part, modulation, and optional feedback
    V_total = V0 + combined_modulation + feedback_adjustment

    # --- Alternative Interpretation: Additive Coupling (as discussed before, kept for reference) ---
    # If the intention was V(x,t) = V0 + intrinsic_term(epsilon(t)) + driver_term(alpha(t), z_t)
    # intrinsic_term = current_epsilon * np.sin(k_potential * x_grid + omega_potential * t)
    # driver_term = current_alpha * z_t * np.cos(k_potential * x_grid) # Example additive driver term
    # V_total_alternative = V0 + intrinsic_term + driver_term + feedback_adjustment
    # Choose the interpretation that matches the theory being tested. Sticking to the original structure.

    # Check for NaNs/Infs in the final potential, which could crash the simulation
    if not np.all(np.isfinite(V_total)):
        logger.error(f"NaN or Inf detected in calculated potential V(x,t) at t={t:.2f}. Check parameters/schedules.")
        # Optionally, return a default safe potential or raise an error
        # For now, replace non-finite values with 0 to potentially allow continuation (use with caution)
        V_total = np.nan_to_num(V_total, nan=0.0, posinf=0.0, neginf=0.0)


    return V_total

# --- END OF FILE quantum_chaos_sim/potentials/dynamic_potential.py ---