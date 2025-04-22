# quantum_chaos_sim/signals.py
"""
Generates different types of driving signals z(t) to be used in the potential.
Includes chaotic, periodic, quasi-periodic, map-based, and noise signals.
Acts as an interface to underlying chaos models and signal generators.
"""
import numpy as np
import logging
# Import specific chaos model functions
from .chaos import lorenz
from .chaos import rossler # NEW Import
from .chaos import logistic_map # NEW Import
# Import other necessary libraries
from scipy import signal as sp_signal # For filtered noise

logger = logging.getLogger(__name__)

# --- Lorenz Driver ---
def generate_lorenz_signal(config_obj):
    """
    Generates the Lorenz signal using the function from the chaos.lorenz module.
    """
    logger.info(f"Generating driver signal: Lorenz Attractor")
    # Use the function from the chaos module
    t_signal, full_state, extracted_signal = lorenz.get_chaotic_signal(config_obj)

    if t_signal is None:
        logger.error("Failed to generate Lorenz signal.")
        return None, None, None

    # Set the driver observable name based on config
    component_name = getattr(config_obj, 'lorenz_use_component', 'z').lower()
    try: setattr(config_obj, 'info_flow_observable_driver', component_name)
    except AttributeError: pass

    return t_signal, extracted_signal, full_state

# --- Rossler Driver ---
def generate_rossler_signal(config_obj):
    """
    Generates the Rossler signal using the function from the chaos.rossler module.
    """
    logger.info(f"Generating driver signal: Rossler Attractor")
    # Use the function from the chaos module
    t_signal, full_state, extracted_signal = rossler.get_rossler_signal_from_sim(config_obj)

    if t_signal is None:
        logger.error("Failed to generate Rossler signal.")
        return None, None, None

    # Set the driver observable name based on config
    component_name = getattr(config_obj, 'rossler_use_component', 'x').lower()
    try: setattr(config_obj, 'info_flow_observable_driver', component_name)
    except AttributeError: pass

    return t_signal, extracted_signal, full_state

# --- Logistic Map Driver ---
def generate_logistic_map_signal(config_obj):
    """
    Generates a signal based on the logistic map using the function from chaos.logistic_map.
    """
    logger.info(f"Generating driver signal: Logistic Map")
    # Use the function from the chaos module
    t_signal, signal, full_state = logistic_map.get_logistic_map_signal(config_obj) # full_state will be None

    if t_signal is None:
        logger.error("Failed to generate Logistic Map signal.")
        return None, None, None

    try: setattr(config_obj, 'info_flow_observable_driver', 'logistic')
    except AttributeError: pass

    return t_signal, signal, full_state # full_state is None here

# --- Filtered Noise Driver (Implementation remains here) ---
def generate_filtered_noise_signal(config_obj):
    """
    Generates a filtered noise signal (Gaussian or Uniform).
    """
    logger.info(f"Generating driver signal: Filtered Noise")
    try:
        t_grid = config_obj.t_grid
        num_points = len(t_grid)
        sampling_rate = getattr(config_obj, 'sampling_rate', 1.0 / (t_grid[1] - t_grid[0]))
        if sampling_rate <= 0: raise ValueError("sampling_rate must be positive.")

        noise_type = getattr(config_obj, 'noise_type', 'gaussian').lower()
        seed = getattr(config_obj, 'noise_seed', None)
        scale = getattr(config_obj, 'noise_scale', 1.0)
        filter_type = getattr(config_obj, 'filter_type', 'lowpass').lower()
        filter_order = getattr(config_obj, 'filter_order', 5)
        cutoff_low = getattr(config_obj, 'filter_cutoff_low', 0.1)
        cutoff_high = getattr(config_obj, 'filter_cutoff_high', 0.4) # Only for bandpass

        rng = np.random.default_rng(seed)
        logger.debug(f"Noise params: type={noise_type}, scale={scale}, seed={seed}")
        logger.debug(f"Filter params: type={filter_type}, order={filter_order}, cutoff_low={cutoff_low}, cutoff_high={cutoff_high}, Fs={sampling_rate:.2f}")

        # Generate raw noise
        if noise_type == 'gaussian':
            raw_noise = rng.normal(loc=0.0, scale=scale, size=num_points)
        elif noise_type == 'uniform':
            raw_noise = rng.uniform(low=-scale, high=scale, size=num_points)
        else:
            logger.error(f"Invalid noise_type '{noise_type}'. Use 'gaussian' or 'uniform'.")
            return None, None, None

        # Design filter
        nyquist = 0.5 * sampling_rate
        if filter_type == 'lowpass':
            if cutoff_low <= 0 or cutoff_low >= nyquist: logger.error(f"Invalid lowpass cutoff {cutoff_low}. Must be (0, {nyquist:.2f})."); return None, None, None
            b, a = sp_signal.butter(filter_order, cutoff_low / nyquist, btype='low', analog=False)
        elif filter_type == 'highpass':
             if cutoff_low <= 0 or cutoff_low >= nyquist: logger.error(f"Invalid highpass cutoff {cutoff_low}. Must be (0, {nyquist:.2f})."); return None, None, None
             b, a = sp_signal.butter(filter_order, cutoff_low / nyquist, btype='high', analog=False)
        elif filter_type == 'bandpass':
            if cutoff_low <= 0 or cutoff_high <= cutoff_low or cutoff_high >= nyquist: logger.error(f"Invalid bandpass cutoffs [{cutoff_low}, {cutoff_high}]. Must be 0 < low < high < {nyquist:.2f}."); return None, None, None
            b, a = sp_signal.butter(filter_order, [cutoff_low / nyquist, cutoff_high / nyquist], btype='band', analog=False)
        else:
            logger.error(f"Invalid filter_type '{filter_type}'. Use 'lowpass', 'highpass', or 'bandpass'.")
            return None, None, None

        # Apply filter (use filtfilt for zero phase distortion)
        filtered_noise = sp_signal.filtfilt(b, a, raw_noise)

        if not np.all(np.isfinite(filtered_noise)): logger.error("NaN or Inf detected after filtering noise signal."); return None, None, None

        driver_name = f'noise_{filter_type}'
        try: setattr(config_obj, 'info_flow_observable_driver', driver_name)
        except AttributeError: pass

        return t_grid, filtered_noise, None

    except AttributeError as ae:
        logger.error(f"Missing required config attribute for Filtered Noise driver: {ae}")
        return None, None, None
    except ValueError as ve:
        logger.error(f"ValueError generating Filtered Noise signal: {ve}", exc_info=True)
        return None, None, None
    except Exception as e:
        logger.error(f"Error generating Filtered Noise signal: {e}", exc_info=True)
        return None, None, None

# --- Sine Wave Driver ---
def generate_sine_signal(config_obj):
    """ Generates a sinusoidal signal z(t) = A * sin(omega * t). """
    logger.info(f"Generating driver signal: Sine Wave")
    t_signal = config_obj.t_grid
    amplitude = getattr(config_obj, 'sine_amplitude', 1.0)
    frequency = getattr(config_obj, 'sine_frequency', 0.5)
    z_signal = amplitude * np.sin(frequency * t_signal)
    try: setattr(config_obj, 'info_flow_observable_driver', 'sine')
    except AttributeError: pass
    return t_signal, z_signal, None

# --- Quasi-Periodic Driver ---
def generate_quasi_periodic_signal(config_obj):
    """ Generates a quasi-periodic signal z(t) = A1*sin(w1*t) + A2*sin(w2*t). """
    logger.info(f"Generating driver signal: Quasi-Periodic Wave")
    t_signal = config_obj.t_grid
    amp1 = getattr(config_obj, 'quasi_amplitude1', 1.0)
    freq1 = getattr(config_obj, 'quasi_frequency1', 0.5)
    amp2 = getattr(config_obj, 'quasi_amplitude2', 0.7)
    freq2 = getattr(config_obj, 'quasi_frequency2', 0.5 * np.sqrt(2.0))
    z_signal = amp1 * np.sin(freq1 * t_signal) + amp2 * np.sin(freq2 * t_signal)
    try: setattr(config_obj, 'info_flow_observable_driver', 'quasi')
    except AttributeError: pass
    return t_signal, z_signal, None

# --- Zero Driver ---
def generate_zero_signal(config_obj):
    """ Generates a zero signal z(t) = 0. """
    logger.info(f"Generating driver signal: Zero (z(t)=0)")
    t_signal = config_obj.t_grid
    z_signal = np.zeros_like(t_signal)
    try: setattr(config_obj, 'info_flow_observable_driver', 'zero')
    except AttributeError: pass
    return t_signal, z_signal, None

# --- Dispatcher Function ---
def get_driving_signal(config_obj):
    """
    Dispatcher function to get the appropriate driving signal based on config.
    Handles potential time grid mismatches by interpolating. Now uses chaos modules.

    Args:
        config_obj: Configuration object specifying 'driver_type' and parameters.

    Returns:
        tuple: (t_signal, z_signal, full_state) based on the selected driver type.
               t_signal will match config_obj.t_grid after potential interpolation.
               Returns (None, None, None) if type is invalid or generation fails.
    """
    driver_type = getattr(config_obj, 'driver_type', 'lorenz').lower()
    logger.info(f"Getting driver signal of type: {driver_type}")

    gen_func = None
    if driver_type == 'lorenz':
        gen_func = generate_lorenz_signal
    elif driver_type == 'rossler':
        gen_func = generate_rossler_signal # Uses updated function
    elif driver_type == 'logistic_map':
        gen_func = generate_logistic_map_signal # Uses updated function
    elif driver_type == 'filtered_noise':
        gen_func = generate_filtered_noise_signal
    elif driver_type == 'sine':
        gen_func = generate_sine_signal
    elif driver_type == 'quasi_periodic':
        gen_func = generate_quasi_periodic_signal
    elif driver_type == 'zero':
        gen_func = generate_zero_signal
    else:
        logger.error(f"Invalid driver_type '{config_obj.driver_type}' specified in config.")
        return None, None, None

    # Call the generator function
    t_gen, z_gen, state_gen = gen_func(config_obj)

    if t_gen is None or z_gen is None:
        logger.error(f"Failed to generate signal for driver type '{driver_type}'.")
        return None, None, None

    # --- Ensure signal aligns with config_obj.t_grid ---
    # Interpolation might still be needed if the underlying chaos simulation uses
    # a different time step than the quantum simulation, although we try to use t_eval.
    t_config = config_obj.t_grid
    if not np.allclose(t_gen, t_config):
        logger.warning(f"Time grid mismatch for driver '{driver_type}' (Signal: {len(t_gen)} pts, Config: {len(t_config)} pts). Interpolating signal onto config grid.")
        try:
            z_final = np.interp(t_config, t_gen, z_gen)
            # Interpolate full state if it exists and is compatible
            state_final = None
            if state_gen is not None and isinstance(state_gen, np.ndarray) and state_gen.ndim == 2 and state_gen.shape[0] == len(t_gen):
                 state_final = np.zeros((len(t_config), state_gen.shape[1]))
                 for dim in range(state_gen.shape[1]):
                     state_final[:, dim] = np.interp(t_config, t_gen, state_gen[:, dim])
            elif state_gen is not None:
                logger.warning(f"Cannot interpolate full state for driver '{driver_type}' due to unexpected shape: {state_gen.shape if hasattr(state_gen,'shape') else type(state_gen)}. Setting to None.")
            return t_config, z_final, state_final
        except Exception as e:
            logger.error(f"Failed to interpolate signal for driver '{driver_type}': {e}")
            return None, None, None
    else:
        # Time grids match, return generated signals directly
        return t_gen, z_gen, state_gen

# --- END OF FILE quantum_chaos_sim/signals.py ---