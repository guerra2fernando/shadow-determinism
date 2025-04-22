# --- START OF FILE quantum_chaos_sim/classical_mechanics/models.py ---

"""
Defines the differential equations for various classical mechanics models,
particularly focusing on 4D systems intended to project onto 3D.
Includes modifications for optional external driver coupling (Phase 3).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Model 1: Coupled 4D Harmonic Oscillators ---
# Equation: d^2(q_i)/dt^2 = -w_i^2 * q_i - sum_{j!=i} k_ij * (q_i - q_j) + F_drive_i
# Example Drive: F_drive_i = coupling_strength * z(t) applied to one oscillator (e.g., i=0)

def coupled_4d_oscillators_deriv(t, state, freqs, couplings, z_t=None, coupling_strength=0.0, driven_oscillator_index=0):
    """
    Differential equations for 4 coupled harmonic oscillators with optional driving.
    state = [x, y, z, w, vx, vy, vz, vw] (position, velocity)
    freqs = [wx, wy, wz, ww] (natural angular frequencies)
    couplings = 4x4 symmetric matrix (k_ij), diagonal is ignored
    z_t (float, optional): Value of the external driving signal at time t.
    coupling_strength (float, optional): Strength of the coupling to z(t).
    driven_oscillator_index (int, optional): Index (0-3) of the oscillator being driven by z(t).
    """
    q = state[:4]  # x, y, z, w
    v = state[4:]  # vx, vy, vz, vw
    w2 = np.array(freqs)**2

    # Calculate acceleration for each oscillator
    a = np.zeros(4)
    for i in range(4):
        # Natural frequency term
        a[i] = -w2[i] * q[i]
        # Coupling terms between oscillators
        for j in range(4):
            if i != j:
                # Ensure k_ij = k_ji (using upper triangle for definition)
                k = couplings[min(i,j)][max(i,j)]
                a[i] -= k * (q[i] - q[j]) # Force proportional to displacement difference

        # Add driving term if applicable
        if z_t is not None and coupling_strength != 0.0 and i == driven_oscillator_index:
             # Check for valid z_t (can be NaN/Inf if driver fails)
             if not isinstance(z_t, (int, float, np.number)) or not np.isfinite(z_t):
                 logger.warning(f"Coupled Osc Deriv: Invalid z_t ({z_t}) at t={t:.2f}. Ignoring drive term.")
             else:
                 a[i] += coupling_strength * z_t

    # Derivatives: dq/dt = v, dv/dt = a
    derivs = np.concatenate((v, a))
    return derivs

def get_coupled_4d_oscillator_params(config_obj):
    """ Safely retrieves parameters for the coupled oscillator model. """
    # Provide default values if not found in config
    freqs = getattr(config_obj, 'classical_oscillator_freqs', [1.0, 1.1, 1.2, 1.3])
    # Default coupling matrix (e.g., nearest-neighbor coupling)
    default_couplings = np.array([
        [0.0, 0.1, 0.0, 0.05], # Couplings for x (x-y, x-w)
        [0.1, 0.0, 0.1, 0.0],  # Couplings for y (y-x, y-z)
        [0.0, 0.1, 0.0, 0.1],  # Couplings for z (z-y, z-w)
        [0.05, 0.0, 0.1, 0.0]  # Couplings for w (w-x, w-z)
    ])
    couplings = getattr(config_obj, 'classical_oscillator_couplings', default_couplings)

    # Validate shapes
    if len(freqs) != 4:
        logger.warning(f"classical_oscillator_freqs should have length 4. Using default: {freqs}")
        freqs = [1.0, 1.1, 1.2, 1.3]
    if not isinstance(couplings, np.ndarray) or couplings.shape != (4, 4):
        logger.warning(f"classical_oscillator_couplings should be a 4x4 numpy array. Using default.")
        couplings = default_couplings
    # Ensure symmetry (optional, based on definition)
    # couplings = (couplings + couplings.T) / 2.0

    # --- Phase 3: Get Coupling Parameters ---
    enable_coupling = getattr(config_obj, 'classical_enable_driver_coupling', False)
    coupling_param_name = getattr(config_obj, 'classical_driver_coupling_param', 'alpha')
    base_coupling_param_value = getattr(config_obj, coupling_param_name, 0.0) # Get quantum alpha/epsilon
    classical_strength_scaling = getattr(config_obj, 'classical_driver_coupling_strength', 1.0)
    driven_osc_idx = getattr(config_obj, 'classical_driven_oscillator_index', 0)

    # Calculate final coupling strength for the classical model
    final_coupling_strength = classical_strength_scaling * base_coupling_param_value if enable_coupling else 0.0

    logger.debug(f"Classical Oscillator Coupling to Driver: Enabled={enable_coupling}, "
                 f"Strength={final_coupling_strength:.3f} (Scale={classical_strength_scaling:.3f} * Quantum_{coupling_param_name}={base_coupling_param_value:.3f}), "
                 f"Driven Osc Index={driven_osc_idx}")

    return freqs, couplings, final_coupling_strength, driven_osc_idx


# --- Model 2: Geometric Projection (Example: 4D Torus Flow) ---
# Can we add driving? E.g., modulate one flow component?
# dx/dt = f1 + coupling * z(t), dy/dt = f2, ...

def torus_4d_flow_deriv(t, state, flow_vector, z_t=None, coupling_strength=0.0, driven_dimension_index=0):
    """
    Differential equations for linear flow on a 4D torus, with optional driving.
    state = [x, y, z, w]
    flow_vector = [fx, fy, fz, fw] (constant base velocities)
    z_t (float, optional): External driving signal value.
    coupling_strength (float, optional): Strength of coupling to z(t).
    driven_dimension_index (int, optional): Index (0-3) of the dimension being driven.
    """
    derivs = np.array(flow_vector, dtype=float)
    # Add driving term if applicable
    if z_t is not None and coupling_strength != 0.0:
        if not isinstance(z_t, (int, float, np.number)) or not np.isfinite(z_t):
            logger.warning(f"Torus Flow Deriv: Invalid z_t ({z_t}) at t={t:.2f}. Ignoring drive term.")
        else:
            if 0 <= driven_dimension_index < len(derivs):
                 derivs[driven_dimension_index] += coupling_strength * z_t
            else:
                 logger.warning(f"Torus Flow Deriv: Invalid driven_dimension_index ({driven_dimension_index}). Ignoring drive term.")
    return derivs

def get_torus_4d_flow_params(config_obj):
    """ Safely retrieves parameters for the torus flow model. """
    # Frequencies should ideally be incommensurate for interesting projections
    default_flow = [1.0, np.sqrt(2.0), np.sqrt(3.0), np.sqrt(5.0)]
    flow_vector = getattr(config_obj, 'classical_torus_flow_vector', default_flow)
    if len(flow_vector) != 4:
        logger.warning(f"classical_torus_flow_vector should have length 4. Using default: {default_flow}")
        flow_vector = default_flow

    # --- Phase 3: Get Coupling Parameters ---
    enable_coupling = getattr(config_obj, 'classical_enable_driver_coupling', False)
    coupling_param_name = getattr(config_obj, 'classical_driver_coupling_param', 'alpha')
    base_coupling_param_value = getattr(config_obj, coupling_param_name, 0.0) # Get quantum alpha/epsilon
    classical_strength_scaling = getattr(config_obj, 'classical_driver_coupling_strength', 1.0)
    driven_dim_idx = getattr(config_obj, 'classical_driven_dimension_index', 0) # Reuse oscillator index name? Let's use a new one.

    final_coupling_strength = classical_strength_scaling * base_coupling_param_value if enable_coupling else 0.0

    logger.debug(f"Classical Torus Coupling to Driver: Enabled={enable_coupling}, "
                 f"Strength={final_coupling_strength:.3f} (Scale={classical_strength_scaling:.3f} * Quantum_{coupling_param_name}={base_coupling_param_value:.3f}), "
                 f"Driven Dim Index={driven_dim_idx}")

    return flow_vector, final_coupling_strength, driven_dim_idx


# --- Model 3: Rössler Hyperchaos ---
# Add driving term, e.g., to the dw/dt equation?
# dw/dt = -d*z + e*w + coupling * z(t)

def rossler_hyperchaos_deriv(t, state, a, b, c, d, e, z_t=None, coupling_strength=0.0):
    """
    Differential equations for the Rössler hyperchaos system with optional driving.
    state = [x, y, z, w]
    z_t (float, optional): External driving signal value.
    coupling_strength (float, optional): Strength of coupling to z(t) (added to dw/dt).
    """
    x, y, z, w = state
    dx = -y - z
    dy = x + a * y + w
    dz = b + z * (x - c)
    dw = -d * z + e * w

    # Add driving term to dw/dt
    if z_t is not None and coupling_strength != 0.0:
        if not isinstance(z_t, (int, float, np.number)) or not np.isfinite(z_t):
            logger.warning(f"Rossler Deriv: Invalid z_t ({z_t}) at t={t:.2f}. Ignoring drive term.")
        else:
            dw += coupling_strength * z_t

    return np.array([dx, dy, dz, dw])

def get_rossler_hyperchaos_params(config_obj):
    """ Safely retrieves parameters for the Rössler hyperchaos model. """
    a = getattr(config_obj, 'classical_rossler_a', 0.25)
    b = getattr(config_obj, 'classical_rossler_b', 3.0)
    c = getattr(config_obj, 'classical_rossler_c', 0.5) # Often varied
    d = getattr(config_obj, 'classical_rossler_d', 0.5)
    e = getattr(config_obj, 'classical_rossler_e', 0.05)

    # --- Phase 3: Get Coupling Parameters ---
    enable_coupling = getattr(config_obj, 'classical_enable_driver_coupling', False)
    coupling_param_name = getattr(config_obj, 'classical_driver_coupling_param', 'alpha')
    base_coupling_param_value = getattr(config_obj, coupling_param_name, 0.0) # Get quantum alpha/epsilon
    classical_strength_scaling = getattr(config_obj, 'classical_driver_coupling_strength', 1.0)

    final_coupling_strength = classical_strength_scaling * base_coupling_param_value if enable_coupling else 0.0

    logger.debug(f"Classical Rossler Coupling to Driver: Enabled={enable_coupling}, "
                 f"Strength={final_coupling_strength:.3f} (Scale={classical_strength_scaling:.3f} * Quantum_{coupling_param_name}={base_coupling_param_value:.3f})")

    return a, b, c, d, e, final_coupling_strength


# --- Dispatcher Function ---
def get_model_deriv_and_params(config_obj):
    """
    Selects the appropriate derivative function and parameters based on config.
    Now includes parameters related to driver coupling.

    Args:
        config_obj: Configuration object with classical model settings.

    Returns:
        tuple: (deriv_func, base_params_tuple, coupling_params_dict, state_dim)
               deriv_func: The function defining the model's ODEs.
               base_params_tuple: Tuple containing the model's intrinsic parameters.
               coupling_params_dict: Dict with {'coupling_strength': val, ...} needed if coupling enabled.
               state_dim: The dimension of the state vector for this model.
               Returns (None, None, None, 0) if model type is invalid.
    """
    model_type = getattr(config_obj, 'classical_model_type', 'coupled_4d_oscillator').lower()
    logger.info(f"Selected classical model type: {model_type}")
    enable_coupling = getattr(config_obj, 'classical_enable_driver_coupling', False)

    if model_type == 'coupled_4d_oscillator':
        freqs, couplings, coupling_strength, driven_idx = get_coupled_4d_oscillator_params(config_obj)
        base_params = (freqs, couplings)
        coupling_params = {
            'coupling_strength': coupling_strength,
            'driven_oscillator_index': driven_idx
        } if enable_coupling else {}
        return coupled_4d_oscillators_deriv, base_params, coupling_params, 8 # State includes velocities
    elif model_type == 'torus_4d_flow':
        flow_vector, coupling_strength, driven_idx = get_torus_4d_flow_params(config_obj)
        base_params = (flow_vector,)
        coupling_params = {
            'coupling_strength': coupling_strength,
            'driven_dimension_index': driven_idx
        } if enable_coupling else {}
        return torus_4d_flow_deriv, base_params, coupling_params, 4
    elif model_type == 'rossler_hyperchaos':
        a, b, c, d, e, coupling_strength = get_rossler_hyperchaos_params(config_obj)
        base_params = (a, b, c, d, e)
        coupling_params = {
            'coupling_strength': coupling_strength
        } if enable_coupling else {}
        return rossler_hyperchaos_deriv, base_params, coupling_params, 4
    # Add other models here
    # elif model_type == 'some_other_model':
    #    params = get_some_other_model_params(config_obj)
    #    return some_other_model_deriv, (params,), state_dim
    else:
        logger.error(f"Invalid classical_model_type '{model_type}' specified in config.")
        return None, None, None, 0

# --- END OF FILE quantum_chaos_sim/classical_mechanics/models.py ---