# quantum_chaos_sim/chaos/rossler.py
import numpy as np
from scipy.integrate import solve_ivp
import logging

logger = logging.getLogger(__name__)

def rossler_deriv(t, state, a, b, c):
    """
    Differential equations for the standard Rossler attractor.

    Args:
        t (float): Time (required by solve_ivp, but not used in equations).
        state (np.ndarray): Array containing current [x, y, z] values.
        a (float): Rossler parameter a.
        b (float): Rossler parameter b.
        c (float): Rossler parameter c.

    Returns:
        np.ndarray: Array containing [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])


def simulate_rossler_scipy(initial_state, t_span, t_eval, a, b, c):
    """
    Simulates the Rossler attractor using scipy.integrate.solve_ivp.

    Args:
        initial_state (np.ndarray): Starting [x, y, z] values.
        t_span (tuple): Tuple containing (start_time, end_time).
        t_eval (np.ndarray): Array of time points at which to store the solution.
        a (float): Rossler parameter a.
        b (float): Rossler parameter b.
        c (float): Rossler parameter c.

    Returns:
        tuple: (sol.t, sol.y.T) where sol.t is the array of time points
               (should match t_eval) and sol.y.T is the (N_points, 3) array
               of [x, y, z] states. Returns (None, None) if integration fails.
    """
    logger.info(f"Simulating Rossler attractor using scipy.integrate.solve_ivp...")
    logger.debug(f"Rossler params: a={a}, b={b}, c={c}, IC={initial_state}")
    logger.debug(f"Rossler t_span: {t_span}, evaluating at {len(t_eval)} points.")
    try:
        sol = solve_ivp(
            fun=rossler_deriv,
            t_span=t_span,
            y0=initial_state,
            args=(a, b, c),
            t_eval=t_eval,
            dense_output=False,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success:
            logger.error(f"Rossler integration failed: {sol.message}")
            return None, None # Indicate failure

        logger.info("Rossler scipy simulation complete.")
        # sol.y is shape (3, len(t_eval)), transpose it to (len(t_eval), 3)
        return sol.t, sol.y.T

    except Exception as e:
        logger.error(f"Exception during Rossler integration: {e}", exc_info=True)
        return None, None

def get_rossler_signal_from_sim(config_obj):
    """
    Generates the Rossler chaotic signal using parameters from the config object.

    Args:
        config_obj: Config object with Rossler parameters and t_grid.

    Returns:
        tuple: (t_signal, full_state, extracted_signal) or (None, None, None) on failure.
    """
    logger.info("Generating Rossler signal from config object...")
    required_attrs = ['t_grid', 'rossler_initial_state', 'rossler_a', 'rossler_b', 'rossler_c']
    for attr in required_attrs:
        if not hasattr(config_obj, attr):
             logger.error(f"get_rossler_signal_from_sim requires attribute '{attr}'.")
             return None, None, None

    t_span = (config_obj.t_grid[0], config_obj.t_grid[-1])
    t_eval = config_obj.t_grid

    t_signal, full_state = simulate_rossler_scipy(
        config_obj.rossler_initial_state, t_span, t_eval,
        config_obj.rossler_a, config_obj.rossler_b, config_obj.rossler_c
    )

    if t_signal is None or full_state is None:
        logger.error("Failed to obtain Rossler signal from simulation.")
        return None, None, None

    # Extract the specified component
    component_name = getattr(config_obj, 'rossler_use_component', 'x').lower()
    component_idx = {'x': 0, 'y': 1, 'z': 2}.get(component_name, 0) # Default to x
    extracted_signal = full_state[:, component_idx]

    # Verify length consistency
    if len(extracted_signal) != len(config_obj.t_grid):
         logger.warning(f"Mismatch in lengths after Rossler integration: signal ({len(extracted_signal)}) vs t_grid ({len(config_obj.t_grid)}).")

    logger.info(f"Successfully generated Rossler signal component '{component_name}' with {len(extracted_signal)} points.")
    return t_signal, full_state, extracted_signal