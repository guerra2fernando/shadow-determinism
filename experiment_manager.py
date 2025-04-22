# --- START OF FILE quantum_chaos_sim/experiment_manager.py ---
import numpy as np
import copy
import logging
from types import SimpleNamespace

LHS_AVAILABLE = False # Default to False
try:
    from pyDOE3 import lhs
    LHS_AVAILABLE = True
except ImportError:
    LHS_AVAILABLE = False
except Exception: # Catch other potential errors during import
    LHS_AVAILABLE = False


logger = logging.getLogger(__name__)

if not LHS_AVAILABLE:
    # This warning should now only appear if the import truly failed at runtime
    logger.warning("Package 'pyDOE3' not found. Install using 'pip install pyDOE3'. "
                   "Falling back to random sampling for Lorenz IC sweep.")
# --- Helper Function for Schedule Validation ---
def _validate_schedule(schedule):
    """ Basic validation for schedule format and sorting. """
    if schedule is None or not isinstance(schedule, list) or len(schedule) == 0:
        return False # Not a valid schedule list
    if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in schedule):
        return False # Items are not (time, value) pairs
    # Check if times are sorted
    times = [item[0] for item in schedule]
    if not all(times[i] <= times[i+1] for i in range(len(times)-1)):
         logger.warning("Schedule times are not sorted. Results may be unpredictable.")
         # Optionally sort it here, but better to fix at generation
         # schedule.sort(key=lambda x: x[0])
         return False # Consider it invalid if not pre-sorted for simplicity
    return True

# --- Existing Functions (from Phase 1/2 - Keep as is) ---

def generate_lorenz_ic_sweep(base_config, num_samples, method='lhs'):
    """
    Generates experiment configurations by sweeping Lorenz initial conditions.
    (Implementation from Phase 1/2 - no changes needed here)
    """
    logger.info(f"Generating {num_samples} Lorenz IC sweep configurations using '{method}' method.")
    experiments = []
    bounds = getattr(base_config, 'lorenz_ic_bounds', None)
    if not bounds:
        logger.error("Cannot generate Lorenz IC sweep: 'lorenz_ic_bounds' not found in config.")
        return []

    param_names = ['x', 'y', 'z']
    lower_bounds = [bounds[p][0] for p in param_names]
    upper_bounds = [bounds[p][1] for p in param_names]

    samples = None
    if method == 'lhs' and LHS_AVAILABLE:
        # Generate samples in [0, 1] range using LHS
        samples_norm = lhs(len(param_names), samples=num_samples, criterion='maximin', random_state=None)
        # Scale samples to the desired bounds
        samples = lower_bounds + samples_norm * (np.array(upper_bounds) - np.array(lower_bounds))
    elif method == 'random' or (method == 'lhs' and not LHS_AVAILABLE):
        if method == 'lhs': logger.warning("LHS requested but pyDOE2 not available, using random sampling.")
        samples = lower_bounds + np.random.rand(num_samples, len(param_names)) * (np.array(upper_bounds) - np.array(lower_bounds))
    elif method == 'grid':
        # Simple grid - limited practical use for >2D, but shown for example
        # Creates num_samples per dimension, resulting in num_samples^3 total configs
        logger.warning("Using 'grid' method for IC sweep - number of configs will be num_samples^D.")
        n_per_dim = int(round(num_samples**(1./len(param_names))))
        if n_per_dim < 2: n_per_dim = 2
        logger.info(f" Grid sweep: Using ~{n_per_dim} points per dimension.")
        grids = [np.linspace(lower_bounds[i], upper_bounds[i], n_per_dim) for i in range(len(param_names))]
        mesh = np.meshgrid(*grids)
        samples = np.vstack([m.flatten() for m in mesh]).T
        num_samples = len(samples) # Update num_samples based on grid result
        logger.info(f" Grid sweep created {num_samples} configurations.")
    else:
        logger.error(f"Invalid method '{method}' for Lorenz IC sweep.")
        return []

    for i in range(num_samples):
        # Ensure labels are unique even if sample coordinates are identical by chance
        label = f"LorenzIC_{method.upper()}_{i:0{len(str(num_samples))}d}"
        config = copy.deepcopy(base_config)
        config.lorenz_initial_state = samples[i, :].copy() # Assign the new IC
        # Ensure driver type is Lorenz
        config.driver_type = 'lorenz'
        # Make sure scheduling is off for these standard sweeps
        config.enable_parameter_scheduling = False
        experiments.append((label, config))

    logger.info(f"Generated {len(experiments)} Lorenz IC configurations.")
    return experiments


def generate_coupling_sweep(base_config, param_name='alpha', sweep_values=None, num_values=5):
    """
    Generates experiments sweeping a coupling parameter (e.g., 'alpha', 'epsilon').
    (Implementation from Phase 1/2 - ensure scheduling is disabled for these)
    """
    if sweep_values is None:
        if not hasattr(base_config, param_name):
            logger.error(f"Cannot generate sweep: Parameter '{param_name}' not found in base_config.")
            return []
        base_value = getattr(base_config, param_name)
        sweep_values = np.linspace(0.0, base_value, num_values)
    else:
        num_values = len(sweep_values)

    logger.info(f"Generating {num_values} configurations for '{param_name}' sweep.")
    experiments = []
    for i, val in enumerate(sweep_values):
        safe_val_str = f"{val:.3f}".replace('.', 'p')
        label = f"Sweep_{param_name}_{i:0{len(str(num_values))}d}_val{safe_val_str}" # Pad label index
        config = copy.deepcopy(base_config)
        try:
            setattr(config, param_name, val)
            # Maybe adjust other params consistently? (e.g., if sweeping alpha, keep epsilon fixed)
            if param_name == 'alpha': config.epsilon = base_config.epsilon
            if param_name == 'epsilon': config.alpha = base_config.alpha
            # Ensure scheduling is off for these simple sweeps
            config.enable_parameter_scheduling = False
            experiments.append((label, config))
        except Exception as e:
             logger.error(f"Error setting sweep parameter '{param_name}' for value {val}: {e}")

    return experiments


def generate_driver_family_configs(base_config, driver_list=['zero', 'sine', 'quasi_periodic', 'lorenz']):
    """
    Generates configurations for a list of standard driver types.
    (Implementation from Phase 1/2 - ensure scheduling is disabled)
    """
    logger.info(f"Generating configurations for driver types: {driver_list}")
    experiments = []
    for driver_type in driver_list:
        label = f"Driver_{driver_type.capitalize()}"
        config = copy.deepcopy(base_config)
        config.driver_type = driver_type
        # Special handling for 'zero' driver (typically means alpha=0)
        if driver_type == 'zero':
            config.alpha = 0.0
            config.epsilon = base_config.epsilon # Keep intrinsic potential? Or zero it too? Keep for now.
        else:
             # Ensure coupling is potentially active for other drivers
             config.alpha = base_config.alpha
             config.epsilon = base_config.epsilon
        # Ensure scheduling is off
        config.enable_parameter_scheduling = False
        experiments.append((label, config))
    return experiments


def define_standard_experiments(base_config):
    """
    Defines the original set of standard experiments (Activation, Control, Sensitivities).
    (Implementation from Phase 1/2 - ensure scheduling is disabled)
    """
    logger.info("Defining standard experiments (Activation, Control, Sensitivities)...")
    experiments = []
    # Exp 1: Base Activation
    exp1_config = copy.deepcopy(base_config)
    exp1_config.enable_parameter_scheduling = False
    experiments.append(("1_BaseDriver_Activation", exp1_config))
    # Exp 2: Ablation Control
    exp2_config = copy.deepcopy(base_config)
    exp2_config.alpha = getattr(base_config, 'control_alpha', 0.0)
    exp2_config.epsilon = getattr(base_config, 'control_epsilon', base_config.epsilon)
    exp2_config.enable_parameter_scheduling = False
    experiments.append(("2_Ablation_Control", exp2_config))
    # Exp 3: Driver IC Sensitivity
    if getattr(base_config, 'enable_sensitivity_test', False) and base_config.driver_type == 'lorenz':
        exp3_config = copy.deepcopy(base_config)
        perturb = getattr(base_config, 'sensitivity_lorenz_initial_state_perturbation', np.array([1e-5, 0, 0]))
        exp3_config.lorenz_initial_state = base_config.lorenz_initial_state + perturb
        exp3_config.enable_parameter_scheduling = False
        experiments.append(("3_Sensitivity_DriverIC", exp3_config))
    # Exp 4: Quantum IC Sensitivity
    if getattr(base_config, 'enable_quantum_IC_sensitivity_test', False):
        exp4_config = copy.deepcopy(base_config)
        perturb_param = getattr(base_config, 'quantum_IC_perturbation_type', 'x0')
        perturb_scale = getattr(base_config, 'quantum_IC_perturbation_scale', 1e-5)
        try:
            current_val = getattr(exp4_config, perturb_param)
            setattr(exp4_config, perturb_param, current_val + perturb_scale)
            exp4_config.enable_parameter_scheduling = False
            experiments.append(("4_Sensitivity_QuantumIC", exp4_config))
        except AttributeError as e:
             logger.warning(f"Could not setup Quantum IC sensitivity: Param '{perturb_param}' not found. Error: {e}")
        except Exception as e:
             logger.error(f"Error setting up Quantum IC sensitivity test: {e}. Skipping.", exc_info=True)

    return experiments

def generate_repeatability_configs(base_config, run_label_prefix, num_repeats, enable_noise=False):
    """
    Generates configurations for testing repeatability.
    (Implementation from Phase 2 - ensure scheduling is handled if base_config uses it)
    """
    logger.info(f"Generating {num_repeats} repeatability configurations for prefix '{run_label_prefix}' (Noise Enabled: {enable_noise}).")
    experiments = []
    for i in range(num_repeats):
        label = f"{run_label_prefix}_Repeat_{i:0{len(str(num_repeats))}d}"
        config = copy.deepcopy(base_config)
        # Ensure scheduling settings are copied correctly from base_config
        config.enable_parameter_scheduling = getattr(base_config, 'enable_parameter_scheduling', False)
        config.alpha_schedule = getattr(base_config, 'alpha_schedule', None)
        config.epsilon_schedule = getattr(base_config, 'epsilon_schedule', None)

        if enable_noise:
             config.inject_runtime_noise = True
             # Optional: Could also slightly perturb QIC here if desired,
             # e.g., config.x0 += np.random.normal(0, 1e-9)
        else:
             config.inject_runtime_noise = False # Ensure it's off if not requested

        experiments.append((label, config))
    return experiments


def generate_sensitivity_configs(base_config, params_to_perturb, perturbation_scale=1e-5):
    """
    Generates configurations for sensitivity analysis using finite differences.
    (Implementation from Phase 2 - ensure scheduling is handled if base_config uses it)
    """
    logger.info(f"Generating sensitivity analysis configurations for params: {list(params_to_perturb.keys())}")
    experiments = []

    # 1. Add the baseline configuration (respecting its scheduling settings)
    base_label = "Sensitivity_Base"
    config_base = copy.deepcopy(base_config)
    config_base.enable_parameter_scheduling = getattr(base_config, 'enable_parameter_scheduling', False)
    config_base.alpha_schedule = getattr(base_config, 'alpha_schedule', None)
    config_base.epsilon_schedule = getattr(base_config, 'epsilon_schedule', None)
    experiments.append((base_label, config_base))

    # 2. Add perturbed configurations for each parameter
    for param_name, specific_scale in params_to_perturb.items():
        scale = specific_scale if specific_scale is not None else perturbation_scale

        # Perturbation currently only makes sense for non-scheduled parameters
        # Skip perturbing scheduled parameters for now
        if param_name in ['alpha', 'epsilon'] and config_base.enable_parameter_scheduling and getattr(config_base, f"{param_name}_schedule", None) is not None:
            logger.warning(f"Sensitivity: Skipping perturbation for '{param_name}' as it is currently scheduled. Sensitivity analysis assumes fixed base parameters.")
            continue

        if isinstance(scale, (list, np.ndarray)): # Perturbation is an array (e.g., for IC)
            if not hasattr(config_base, param_name):
                logger.warning(f"Sensitivity: Parameter '{param_name}' not found in base config. Skipping.")
                continue
            base_value = getattr(config_base, param_name)
            if not isinstance(base_value, np.ndarray) or base_value.shape != np.array(scale).shape:
                 logger.warning(f"Sensitivity: Shape mismatch for parameter '{param_name}'. Base shape {getattr(base_value, 'shape', 'N/A')}, Perturbation shape {np.array(scale).shape}. Skipping.")
                 continue

            # Positive perturbation
            label_pos = f"Sensitivity_{param_name}_Pert_Pos"
            config_pos = copy.deepcopy(config_base)
            setattr(config_pos, param_name, base_value + np.array(scale))
            experiments.append((label_pos, config_pos))

            # Negative perturbation
            label_neg = f"Sensitivity_{param_name}_Pert_Neg"
            config_neg = copy.deepcopy(config_base)
            setattr(config_neg, param_name, base_value - np.array(scale))
            experiments.append((label_neg, config_neg))

        else: # Perturbation is a scalar
            if not hasattr(config_base, param_name):
                logger.warning(f"Sensitivity: Parameter '{param_name}' not found in base config. Skipping.")
                continue
            base_value = getattr(config_base, param_name)
            if not isinstance(base_value, (int, float, np.number)):
                logger.warning(f"Sensitivity: Parameter '{param_name}' is not a scalar ({type(base_value)}). Skipping scalar perturbation.")
                continue

            # Positive perturbation
            label_pos = f"Sensitivity_{param_name}_Pert_Pos"
            config_pos = copy.deepcopy(config_base)
            setattr(config_pos, param_name, base_value + scale)
            experiments.append((label_pos, config_pos))

            # Negative perturbation
            label_neg = f"Sensitivity_{param_name}_Pert_Neg"
            config_neg = copy.deepcopy(config_base)
            setattr(config_neg, param_name, base_value - scale)
            experiments.append((label_neg, config_neg))

    logger.info(f"Generated {len(experiments)} sensitivity configurations (including base).")
    return experiments

# --- NEW Functions for Phase 3 ---

def generate_gating_configs(base_config, param_to_gate='alpha', on_value=None, off_value=0.0, switch_times=None):
    """
    Generates configurations for testing driver gating (switching coupling on/off).

    Args:
        base_config: The baseline configuration object.
        param_to_gate (str): The parameter to switch ('alpha' or 'epsilon').
        on_value (float, optional): The value when the gate is 'on'. Defaults to base_config value.
        off_value (float): The value when the gate is 'off'.
        switch_times (list, optional): List of times at which to switch the state (on->off or off->on).
                                       If None, defaults to a simple [T/3, 2*T/3] schedule.

    Returns:
        list: A list of tuples: [(run_label, config_object), ...]
    """
    logger.info(f"Generating gating configurations for parameter '{param_to_gate}'.")
    experiments = []

    if on_value is None:
        on_value = getattr(base_config, param_to_gate, 1.0) # Default to base value or 1.0

    if switch_times is None:
        T_total = getattr(base_config, 'T', 50.0)
        switch_times = [T_total / 3.0, 2 * T_total / 3.0]

    # --- Create On->Off->On Schedule ---
    schedule_on_off_on = []
    current_state = 'on'
    last_time = 0.0
    schedule_on_off_on.append((last_time, on_value)) # Start 'on'
    for t_switch in sorted(switch_times):
         if t_switch > last_time:
             if current_state == 'on':
                 schedule_on_off_on.append((t_switch, off_value))
                 current_state = 'off'
             else:
                 schedule_on_off_on.append((t_switch, on_value))
                 current_state = 'on'
             last_time = t_switch
    if _validate_schedule(schedule_on_off_on):
         label1 = f"Gating_{param_to_gate}_ON_OFF_ON"
         config1 = copy.deepcopy(base_config)
         config1.enable_parameter_scheduling = True
         setattr(config1, f"{param_to_gate}_schedule", schedule_on_off_on)
         # Ensure the non-scheduled parameter uses its base value
         if param_to_gate == 'alpha': config1.epsilon = base_config.epsilon; config1.epsilon_schedule = None
         else: config1.alpha = base_config.alpha; config1.alpha_schedule = None
         experiments.append((label1, config1))
    else:
         logger.error(f"Failed to generate valid ON-OFF-ON gating schedule for {param_to_gate}.")


    # --- Create Off->On->Off Schedule ---
    schedule_off_on_off = []
    current_state = 'off'
    last_time = 0.0
    schedule_off_on_off.append((last_time, off_value)) # Start 'off'
    for t_switch in sorted(switch_times):
         if t_switch > last_time:
             if current_state == 'off':
                 schedule_off_on_off.append((t_switch, on_value))
                 current_state = 'on'
             else:
                 schedule_off_on_off.append((t_switch, off_value))
                 current_state = 'off'
             last_time = t_switch
    if _validate_schedule(schedule_off_on_off):
        label2 = f"Gating_{param_to_gate}_OFF_ON_OFF"
        config2 = copy.deepcopy(base_config)
        config2.enable_parameter_scheduling = True
        setattr(config2, f"{param_to_gate}_schedule", schedule_off_on_off)
         # Ensure the non-scheduled parameter uses its base value
        if param_to_gate == 'alpha': config2.epsilon = base_config.epsilon; config2.epsilon_schedule = None
        else: config2.alpha = base_config.alpha; config2.alpha_schedule = None
        experiments.append((label2, config2))
    else:
         logger.error(f"Failed to generate valid OFF-ON-OFF gating schedule for {param_to_gate}.")

    # --- Add Baseline (Constant ON) for comparison ---
    label_on = f"Gating_{param_to_gate}_ConstantON"
    config_on = copy.deepcopy(base_config)
    setattr(config_on, param_to_gate, on_value)
    if param_to_gate == 'alpha': config_on.epsilon = base_config.epsilon
    else: config_on.alpha = base_config.alpha
    config_on.enable_parameter_scheduling = False # Not scheduled
    experiments.append((label_on, config_on))

    # --- Add Baseline (Constant OFF) for comparison ---
    label_off = f"Gating_{param_to_gate}_ConstantOFF"
    config_off = copy.deepcopy(base_config)
    setattr(config_off, param_to_gate, off_value)
    if param_to_gate == 'alpha': config_off.epsilon = base_config.epsilon
    else: config_off.alpha = base_config.alpha
    config_off.enable_parameter_scheduling = False # Not scheduled
    experiments.append((label_off, config_off))


    logger.info(f"Generated {len(experiments)} gating configurations.")
    return experiments


def generate_embedding_configs(base_config, message_bits, param_to_modulate='alpha', bit_high_mod=0.1, bit_low_mod=-0.1, bit_duration=None):
    """
    Generates configurations for testing signal embedding by modulating a parameter.

    Args:
        base_config: The baseline configuration object.
        message_bits (list or np.ndarray): Sequence of bits (0s and 1s) to embed.
        param_to_modulate (str): Parameter to modulate ('alpha' or 'epsilon').
        bit_high_mod (float): Additive modulation for bit '1'.
        bit_low_mod (float): Additive modulation for bit '0'.
        bit_duration (float, optional): Duration each bit's modulation is active.
                                        Defaults to T / len(message_bits).

    Returns:
        list: A list of tuples: [(run_label, config_object), ...]
    """
    logger.info(f"Generating embedding configurations for parameter '{param_to_modulate}' and message: {message_bits}")
    experiments = []
    T_total = getattr(base_config, 'T', 50.0)

    if not message_bits or len(message_bits) == 0:
        logger.warning("Cannot generate embedding config: message_bits is empty.")
        return []

    if bit_duration is None:
        bit_duration = T_total / len(message_bits)

    if bit_duration * len(message_bits) > T_total + 1e-6: # Allow small tolerance
        logger.warning(f"Total duration of message bits ({bit_duration * len(message_bits):.2f}) exceeds simulation time T ({T_total:.2f}). Message may be truncated.")

    base_value = getattr(base_config, param_to_modulate, 0.0)
    schedule = []
    current_time = 0.0
    # Initial value is the base value
    schedule.append((current_time, base_value))

    for i, bit in enumerate(message_bits):
        start_time = i * bit_duration
        end_time = (i + 1) * bit_duration

        if start_time >= T_total: # Stop if message exceeds simulation time
             break

        if bit == 1:
            mod_value = base_value + bit_high_mod
        else: # bit == 0
            mod_value = base_value + bit_low_mod

        # Add modulation start point
        if not np.isclose(start_time, current_time): # Avoid duplicate time points if duration is very small
             schedule.append((start_time, mod_value))
        elif len(schedule) > 0: # Update last point if time is the same
             schedule[-1] = (start_time, mod_value)

        current_time = start_time

        # Add modulation end point (return to base value)
        if end_time < T_total:
             if not np.isclose(end_time, current_time):
                  schedule.append((end_time, base_value))
             # If end_time is the same as start_time (shouldn't happen with duration>0), update the point
             elif len(schedule) > 0 and np.isclose(schedule[-1][0], end_time):
                   schedule[-1] = (end_time, base_value)
        current_time = end_time # Update current time regardless

    # Ensure final segment goes to T
    if current_time < T_total and len(schedule) > 0 and not np.isclose(schedule[-1][0], T_total):
        # If last action was modulation, ensure it returns to base at T_total
        if schedule[-1][1] != base_value:
              # Check if the last time point added was the end of modulation
              if current_time > schedule[-1][0]: # current_time is the end_time
                  schedule.append((current_time, base_value)) # Add return to base point
              # Else: the last point added was start of modulation, update it if needed? Unlikely scenario.

        # If schedule ends before T, add a final point ensuring base value until T
        # This might already be handled if last segment returned to base
        # Let's ensure a point exists *near* T if needed (though interpolation handles it)
        pass


    if _validate_schedule(schedule):
         msg_str = "".join(map(str, message_bits))
         label = f"Embed_{param_to_modulate}_Msg{msg_str[:10]}{'...' if len(msg_str)>10 else ''}" # Truncate long message in label
         config = copy.deepcopy(base_config)
         config.enable_parameter_scheduling = True
         setattr(config, f"{param_to_modulate}_schedule", schedule)
         # Ensure the non-scheduled parameter uses its base value
         if param_to_modulate == 'alpha': config.epsilon = base_config.epsilon; config.epsilon_schedule = None
         else: config.alpha = base_config.alpha; config.alpha_schedule = None
         experiments.append((label, config))
         logger.debug(f"Generated schedule for {label}: {schedule}")
    else:
         logger.error(f"Failed to generate valid embedding schedule for {param_to_modulate}.")

    # --- Add Baseline (No Modulation) for comparison ---
    label_base = f"Embed_{param_to_modulate}_Baseline"
    config_base = copy.deepcopy(base_config)
    # Ensure base values are set correctly
    config_base.alpha = base_config.alpha
    config_base.epsilon = base_config.epsilon
    config_base.enable_parameter_scheduling = False # No schedule for baseline
    experiments.append((label_base, config_base))

    logger.info(f"Generated {len(experiments)} embedding configurations.")
    return experiments

# --- END OF FILE quantum_chaos_sim/experiment_manager.py ---