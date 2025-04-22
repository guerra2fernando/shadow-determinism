# --- START OF FILE quantum_chaos_sim/classical_mechanics/simulation.py ---

"""
Handles the simulation of classical mechanics models using scipy.integrate.
Phase 3: Modified to optionally accept and use an external driver signal z(t).
Handles potential mismatches between quantum driver signal grid and classical sim grid.
"""

import numpy as np
from scipy.integrate import solve_ivp
import logging
import time
from .models import get_model_deriv_and_params # Import from sibling module

logger = logging.getLogger(__name__)

def simulate_classical_system(config_obj, t_signal=None, z_signal=None):
    """
    Simulates the selected classical model using parameters from the config object.
    Optionally uses a pre-computed external driver signal z(t).

    Args:
        config_obj: Configuration object containing classical simulation settings
                   (model type, parameters, initial state, time span, dt).
        t_signal (np.ndarray, optional): Time points for the external driver signal.
                                         Must match the classical simulation time grid,
                                         OR interpolation will be attempted.
        z_signal (np.ndarray, optional): Values of the external driver signal z(t).
                                         Required if config_obj.classical_enable_driver_coupling is True.

    Returns:
        tuple: (t_values, states)
               t_values: Array of time points at which the solution was evaluated.
               states: Array of shape (N_points, state_dim) containing the system state.
               Returns (None, None) if simulation fails or model is invalid.
    """
    start_time = time.time()
    logger.info("--- Starting Classical Simulation ---")

    enable_coupling = getattr(config_obj, 'classical_enable_driver_coupling', False)

    # --- Get Model Specifics ---
    # Returns: (deriv_func, base_params_tuple, coupling_params_dict, state_dim)
    deriv_func, base_params_tuple, coupling_params_dict, state_dim = get_model_deriv_and_params(config_obj)
    if deriv_func is None:
        logger.error("Cannot run classical simulation: Invalid model type.")
        return None, None

    # --- Define Classical Time Grid ---
    # This grid is used for the simulation itself and for evaluating results.
    # It might differ from the quantum grid (t_signal).
    target_T = getattr(config_obj, 'classical_T', 100.0)
    target_dt = getattr(config_obj, 'classical_dt', 0.01)
    t_classical_eval = np.arange(0, target_T + target_dt, target_dt)
    if not np.isclose(t_classical_eval[-1], target_T):
        t_classical_eval = np.append(t_classical_eval, target_T)

    # --- Validate Driver Signal if Coupling is Enabled ---
    z_classical_grid = None # Define variable for z signal on classical grid

    if enable_coupling:
        if z_signal is None or t_signal is None:
            logger.error("Classical simulation requires t_signal and z_signal when classical_enable_driver_coupling=True. Coupling disabled.")
            enable_coupling = False # Disable coupling if signal is missing
            coupling_params_dict = {} # Clear coupling params
        else:
             # Check length consistency first
             if len(t_signal) != len(z_signal):
                 logger.error(f"Classical driver signal length mismatch: t_signal ({len(t_signal)}), z_signal ({len(z_signal)}). Coupling disabled.")
                 enable_coupling = False
                 coupling_params_dict = {}
             # Now check if grids ALREADY match in length and values
             elif len(t_signal) == len(t_classical_eval) and np.allclose(t_signal, t_classical_eval):
                 logger.info("Classical time grid matches provided driver signal time grid. Using signal directly.")
                 z_classical_grid = z_signal # Use the provided signal directly
             else:
                 # Grids do not match in length or values, interpolation is needed
                 logger.warning(f"Classical time grid (Steps: {len(t_classical_eval)}, dt: {target_dt:.4f}, T: {target_T:.2f}) "
                                f"does not match provided driver signal time grid (Steps: {len(t_signal)}, dt: {(t_signal[1]-t_signal[0]):.4f}, T: {t_signal[-1]:.2f}).")
                 logger.info("Interpolating external driver z(t) onto the classical time grid...")
                 try:
                     # Interpolate z_signal (values) onto the classical time grid (t_classical_eval)
                     z_interp = np.interp(t_classical_eval, t_signal, z_signal)
                     z_classical_grid = z_interp # Store the interpolated z values for the wrapper
                     logger.info("Driver signal interpolation successful.")
                 except Exception as e_interp:
                     logger.error(f"Failed to interpolate driver signal: {e_interp}. Coupling disabled.", exc_info=True)
                     enable_coupling = False
                     coupling_params_dict = {}

    # --- Get Simulation Parameters ---
    try:
        # T and dt are already fetched for t_classical_eval
        T = target_T
        dt = target_dt
        initial_state = np.array(config_obj.classical_initial_state)
        if len(initial_state) != state_dim:
             logger.error(f"Initial state length ({len(initial_state)}) does not match model state dimension ({state_dim}) for model '{config_obj.classical_model_type}'.")
             raise ValueError("Initial state dimension mismatch.")

        t_span = (0, T)
        # t_eval is set to t_classical_eval later in solve_ivp call

        logger.info(f"Classical Sim Params: T={T}, dt={dt}, Steps={len(t_classical_eval)}, IC={np.array2string(initial_state, precision=3)}")
        logger.debug(f"Model base parameters: {base_params_tuple}")
        if enable_coupling:
            logger.debug(f"Model coupling parameters: {coupling_params_dict}")
            if z_classical_grid is not None:
                logger.debug(f"Using external driver z(t) interpolated onto classical grid ({len(z_classical_grid)} points).")
            else:
                # This case should ideally not happen if enable_coupling is true after validation
                logger.warning("Coupling enabled but classical grid driver signal is unexpectedly None.")


    except AttributeError as ae:
        logger.error(f"Missing configuration attribute for classical simulation: {ae}", exc_info=True)
        return None, None
    except ValueError as ve: # Catch the specific dimension mismatch error
         logger.error(f"ValueError during classical simulation setup: {ve}", exc_info=True)
         return None, None
    except Exception as e:
        logger.error(f"Error processing classical simulation parameters: {e}", exc_info=True)
        return None, None

    # --- Define Derivative Function Wrapper for solve_ivp ---
    # This wrapper handles passing z(t) to the model's derivative function if coupling is enabled.
    # Make classical grid variables accessible to the wrapper's scope
    _t_classical_eval_for_wrapper = t_classical_eval
    _z_classical_grid_for_wrapper = z_classical_grid # This holds the interpolated values

    def ode_func_wrapper(t, state, *args):
        # args will contain base_params_tuple
        # coupling_params_dict is accessed from the outer scope
        func_args = list(args) # Unpack base parameters
        kwargs = {}

        if enable_coupling:
            # Use the PRE-INTERPOLATED signal defined on the classical grid
            # Interpolate between the points on the classical grid at the solver's intermediate time 't'
            if _t_classical_eval_for_wrapper is not None and _z_classical_grid_for_wrapper is not None:
                try:
                     # Interpolate using the classical time grid and the z values on that grid
                     z_now = np.interp(t, _t_classical_eval_for_wrapper, _z_classical_grid_for_wrapper)
                     # Add a check for NaN/Inf just in case interpolation yields bad values at edges
                     if not np.isfinite(z_now):
                          logger.warning(f"Interpolation result is non-finite ({z_now}) at t={t:.3f}. Using z=0.")
                          z_now = 0.0
                except Exception as e_interp_runtime:
                     logger.warning(f"Interpolation failed during ODE solve at t={t:.3f}: {e_interp_runtime}. Using z=0.")
                     z_now = 0.0
            else:
                 # This shouldn't happen if enable_coupling is True and validation passed, but safeguard.
                 logger.error(f"ODE wrapper called with coupling enabled, but classical grid data missing at t={t:.3f}. Using z=0.")
                 z_now = 0.0

            kwargs['z_t'] = z_now
            # Add other coupling parameters from the dictionary
            kwargs.update(coupling_params_dict)

        # Call the actual derivative function
        return deriv_func(t, state, *func_args, **kwargs)


    # --- Run Integration ---
    logger.info(f"Integrating classical system '{config_obj.classical_model_type}' (Coupling: {enable_coupling})...")
    try:
        sol = solve_ivp(
            fun=ode_func_wrapper, # Use the wrapper
            t_span=t_span,
            y0=initial_state,
            args=base_params_tuple, # Pass only base params here; wrapper handles coupling
            t_eval=t_classical_eval, # Use the classical time grid for evaluation points
            dense_output=False,
            method='RK45',  # Or 'LSODA' for potentially stiff systems
            rtol=1e-6,     # Adjust tolerances as needed
            atol=1e-9
        )

        if not sol.success:
            logger.error(f"Classical integration failed: {sol.message}")
            return None, None

        elapsed = time.time() - start_time
        logger.info(f"Classical integration successful ({elapsed:.2f} seconds).")
        # sol.y is shape (state_dim, N_points), transpose to (N_points, state_dim)
        return sol.t, sol.y.T

    except Exception as e:
        logger.error(f"Exception during classical integration: {e}", exc_info=True)
        return None, None

# --- END OF FILE quantum_chaos_sim/classical_mechanics/simulation.py ---