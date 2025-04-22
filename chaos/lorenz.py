# quantum_chaos_sim/chaos/lorenz.py
import numpy as np
from scipy.integrate import solve_ivp
import logging

logger = logging.getLogger(__name__)

def lorenz_deriv(t, state, sigma, rho, beta):
    """
    Differential equations for the Lorenz attractor.

    Args:
        t (float): Time (required by solve_ivp, but not used in equations).
        state (np.ndarray): Array containing current [x, y, z] values.
        sigma (float): Lorenz parameter sigma.
        rho (float): Lorenz parameter rho.
        beta (float): Lorenz parameter beta.

    Returns:
        np.ndarray: Array containing [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

# --- RK4 implementation (optional, kept for reference but not used by get_chaotic_signal) ---
def rk4_step(f, state, t, dt, *args):
    """
    Performs a single Runge-Kutta 4th order step for an ODE.

    Args:
        f: Function defining the derivatives (e.g., lorenz_deriv). Signature f(t, state, *args).
        state (np.ndarray): Current state vector.
        t (float): Current time.
        dt (float): Time step size.
        *args: Additional arguments required by the function f.

    Returns:
        np.ndarray: State vector at time t + dt.
    """
    k1 = dt * f(t, state, *args)
    k2 = dt * f(t + dt/2, state + k1/2, *args)
    k3 = dt * f(t + dt/2, state + k2/2, *args)
    k4 = dt * f(t + dt, state + k3, *args)
    new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return new_state

def simulate_lorenz_rk4(initial_state, t_span, dt, sigma, rho, beta):
    """
    Simulates the Lorenz attractor using a custom RK4 implementation.

    Args:
        initial_state (np.ndarray): Starting [x, y, z] values.
        t_span (tuple): Tuple containing (start_time, end_time).
        dt (float): Fixed time step for RK4 integration.
        sigma (float): Lorenz parameter sigma.
        rho (float): Lorenz parameter rho.
        beta (float): Lorenz parameter beta.

    Returns:
        tuple: (t_values, states) where t_values is an array of time points
               and states is a (N_steps, 3) array of [x, y, z] values.
    """
    logger.info(f"Simulating Lorenz attractor using custom RK4 (dt={dt})...")
    t_values = np.arange(t_span[0], t_span[1] + dt, dt) # Include end time
    num_steps = len(t_values)
    states = np.zeros((num_steps, 3))
    states[0] = initial_state

    for i in range(num_steps - 1):
        states[i+1] = rk4_step(lorenz_deriv, states[i], t_values[i], dt, sigma, rho, beta)
        if (i+1) % 5000 == 0: # Log progress periodically
             logger.debug(f"Lorenz RK4 step {i+1}/{num_steps-1}")

    logger.info("Lorenz RK4 simulation complete.")
    return t_values, states
# --- End of RK4 implementation ---


def simulate_lorenz_scipy(initial_state, t_span, t_eval, sigma, rho, beta):
    """
    Simulates the Lorenz attractor using scipy.integrate.solve_ivp (recommended).

    Args:
        initial_state (np.ndarray): Starting [x, y, z] values.
        t_span (tuple): Tuple containing (start_time, end_time).
        t_eval (np.ndarray): Array of time points at which to store the solution.
        sigma (float): Lorenz parameter sigma.
        rho (float): Lorenz parameter rho.
        beta (float): Lorenz parameter beta.

    Returns:
        tuple: (sol.t, sol.y.T) where sol.t is the array of time points
               (should match t_eval) and sol.y.T is the (N_points, 3) array
               of [x, y, z] states. Returns (None, None) if integration fails.
    """
    logger.info(f"Simulating Lorenz attractor using scipy.integrate.solve_ivp...")
    try:
        sol = solve_ivp(
            fun=lorenz_deriv,
            t_span=t_span,
            y0=initial_state,
            args=(sigma, rho, beta),
            t_eval=t_eval,  # Evaluate solution only at these specific times
            dense_output=False, # We only need values at t_eval points
            method='RK45',  # Adaptive Runge-Kutta method (good default)
            # Optional: Set tolerances for higher accuracy if needed
            # rtol=1e-6,
            # atol=1e-9
        )

        if not sol.success:
            logger.error(f"Lorenz integration failed: {sol.message}")
            return None, None # Indicate failure

        logger.info("Lorenz scipy simulation complete.")
        # sol.y is shape (3, len(t_eval)), transpose it to (len(t_eval), 3)
        return sol.t, sol.y.T

    except Exception as e:
        logger.error(f"Exception during Lorenz integration: {e}", exc_info=True)
        return None, None


def get_chaotic_signal(config_obj):
    """
    Generates the chaotic signal z(t) over the quantum simulation time grid,
    using parameters from the provided config object.
    Uses the more robust scipy integrator.

    Args:
        config_obj: An object (like SimpleNamespace) holding configuration
                   attributes (e.g., t_grid, lorenz_initial_state,
                   lorenz_sigma, lorenz_rho, lorenz_beta).

    Returns:
        tuple: (t_chaos, states_chaos, z_signal)
               t_chaos: Array of time points for the chaos simulation.
               states_chaos: Array of shape (N_points, 3) containing [x,y,z].
               z_signal: Array of the z-component values at t_chaos times.
               Returns (None, None, None) if simulation fails.
    """
    logger.info("Generating chaotic signal from config object...")
    # Validate necessary attributes
    required_attrs = ['t_grid', 'lorenz_initial_state', 'lorenz_sigma', 'lorenz_rho', 'lorenz_beta']
    for attr in required_attrs:
        if not hasattr(config_obj, attr):
             logger.error(f"get_chaotic_signal requires attribute '{attr}' in the config object.")
             return None, None, None

    # Define the time span and evaluation points from the quantum sim config
    t_span = (config_obj.t_grid[0], config_obj.t_grid[-1])
    # Evaluate Lorenz at the exact quantum time steps for direct use
    t_eval = config_obj.t_grid

    logger.debug(f"Lorenz params: sigma={config_obj.lorenz_sigma}, rho={config_obj.lorenz_rho}, beta={config_obj.lorenz_beta:.3f}")
    logger.debug(f"Lorenz initial state: {config_obj.lorenz_initial_state}")
    logger.debug(f"Lorenz t_span: {t_span}, evaluating at {len(t_eval)} points.")


    # Call the scipy integrator
    t_chaos, states_chaos = simulate_lorenz_scipy(
        config_obj.lorenz_initial_state, t_span, t_eval,
        config_obj.lorenz_sigma, config_obj.lorenz_rho, config_obj.lorenz_beta
    )

    # Check if simulation was successful
    if t_chaos is None or states_chaos is None:
        logger.error("Failed to obtain chaotic signal.")
        return None, None, None

    # Extract the z-component
    z_signal = states_chaos[:, 2]

    # Verify length consistency (solve_ivp with t_eval should guarantee this if successful)
    if len(z_signal) != len(config_obj.t_grid):
         logger.warning(f"Mismatch in lengths after integration: z_signal ({len(z_signal)}) vs t_grid ({len(config_obj.t_grid)}). This might indicate an issue.")
         # Depending on severity, could return None or try to proceed
         # For now, just issue warning.

    logger.info(f"Successfully generated chaotic signal z(t) with {len(z_signal)} points.")
    return t_chaos, states_chaos, z_signal