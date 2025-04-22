# quantum_chaos_sim/chaos/logistic_map.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_logistic_sequence(r, x0, num_points, skip_transients=1000):
    """
    Generates a sequence of values from the logistic map:
    x_{n+1} = r * x_n * (1 - x_n)

    Args:
        r (float): The logistic map parameter (typically between 0 and 4).
        x0 (float): The initial value (must be between 0 and 1).
        num_points (int): The number of points desired in the final sequence.
        skip_transients (int): Number of initial iterations to discard.

    Returns:
        np.ndarray or None: Array containing the generated sequence (length num_points),
                            or None if input is invalid or map diverges.
    """
    if not (0 < x0 < 1):
        logger.error(f"logistic_initial_x ({x0}) must be between 0 and 1.")
        return None
    if not (0 <= r <= 4):
        logger.warning(f"logistic_r ({r}) is outside the typical chaotic range [~3.57, 4].")
    if num_points <= 0 or skip_transients < 0:
        logger.error("num_points must be positive and skip_transients non-negative.")
        return None

    logger.debug(f"Generating logistic sequence: r={r}, x0={x0}, points={num_points}, skip={skip_transients}")

    total_iterations = skip_transients + num_points
    x_values = np.zeros(total_iterations)
    x_values[0] = x0

    try:
        current_x = x0
        for i in range(total_iterations - 1):
            next_x = r * current_x * (1 - current_x)
            # Check for divergence or invalid states, especially if r > 4
            if not (0 <= next_x <= 1) and r <= 4: # Check bounds if r is in typical range
                 logger.warning(f"Logistic map value {next_x:.4f} went outside [0, 1] at step {i+1} (r={r}). Clamping.")
                 next_x = np.clip(next_x, 0, 1)
            elif not np.isfinite(next_x):
                logger.error(f"Logistic map diverged to non-finite value at step {i+1} (r={r}, x={current_x:.4f}).")
                return None

            x_values[i+1] = next_x
            current_x = next_x

    except Exception as e:
         logger.error(f"Exception during logistic map iteration: {e}", exc_info=True)
         return None

    # Return the sequence after discarding transients
    final_sequence = x_values[skip_transients:]
    logger.info(f"Logistic map sequence generated ({len(final_sequence)} points).")
    return final_sequence


def get_logistic_map_signal(config_obj):
    """
    Generates the driving signal based on the logistic map using parameters
    from the config object.

    Args:
        config_obj: Config object with Logistic Map parameters and t_grid.

    Returns:
        tuple: (t_grid, signal, None) where signal is the scaled/offset map output,
               or (None, None, None) on failure.
    """
    logger.info("Generating Logistic Map signal from config object...")
    required_attrs = ['t_grid', 'logistic_r', 'logistic_initial_x']
    for attr in required_attrs:
        if not hasattr(config_obj, attr):
             logger.error(f"get_logistic_map_signal requires attribute '{attr}'.")
             return None, None, None

    t_grid = config_obj.t_grid
    num_points = len(t_grid)
    skip = getattr(config_obj, 'logistic_skip_transients', 1000)
    scale = getattr(config_obj, 'logistic_scale', 1.0)
    offset = getattr(config_obj, 'logistic_offset', 0.0)

    sequence = generate_logistic_sequence(
        r=config_obj.logistic_r,
        x0=config_obj.logistic_initial_x,
        num_points=num_points,
        skip_transients=skip
    )

    if sequence is None:
        logger.error("Failed to generate logistic map sequence.")
        return None, None, None

    # Apply scaling and offset
    signal = scale * sequence + offset

    # Check for NaNs/Infs in final signal (e.g., if scale/offset are problematic)
    if not np.all(np.isfinite(signal)):
        logger.error("NaN or Inf detected in final logistic map signal after scaling/offset.")
        return None, None, None

    logger.info(f"Successfully generated Logistic Map signal with {len(signal)} points.")
    # The 'full_state' for a map is just the sequence itself, return None
    return t_grid, signal, None