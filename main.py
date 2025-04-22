# --- START OF FILE quantum_chaos_sim/main.py ---
import numpy as np
import logging
import time
import os
import copy # For deep copying config SimpleNamespace object
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt # Needed for FFT plot saving and sweep plot
import pandas as pd # Import pandas
from types import SimpleNamespace # Import SimpleNamespace

# Import local modules
from . import config as config_module # Import the config module itself

from .analysis.common_analysis_utils import get_plot_dir
# --- Attempt Core Module Imports ---
# These are generally required regardless of mode
try:
    from . import signals
    from .potentials.dynamic_potential import dynamic_potential
    from .core import schrodinger
    from .observables import analysis
    from .visualization import plot_wavefunction
    from .validation import metrics
    CORE_MODULES_OK = True
except ImportError as e:
    # Basic config for early error - might not work if logging itself fails
    try:
        # Configure basic logging JUST for this critical error if not already configured
        if not logging.getLogger().hasHandlers():
             logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).critical(f"Failed to import core simulation modules: {e}. Cannot proceed.", exc_info=True)
    except Exception:
        print(f"CRITICAL ERROR: Failed to import core simulation modules: {e}. Logging failed.")
    CORE_MODULES_OK = False
    # Exit if core modules fail? Or let subsequent checks handle it? Let checks handle.

# --- Attempt Optional/Mode-Dependent Module Imports ---

# Import classical mechanics modules IF enabled in config
CLASSICAL_SIM_AVAILABLE = False
_classical_import_error = None
if getattr(config_module.cfg, 'enable_classical_simulation', False):
    try:
        from .classical_mechanics import simulation as classical_sim
        from .classical_mechanics import analysis as classical_analysis
        CLASSICAL_SIM_AVAILABLE = True
    except ImportError as e:
        CLASSICAL_SIM_AVAILABLE = False
        _classical_import_error = e

# --- Required for Suite Execution ---
RESULTS_HANDLER_AVAILABLE = False
_results_handler_import_error = None
try:
    from . import results_handler
    RESULTS_HANDLER_AVAILABLE = True
except ImportError as e:
    _results_handler_import_error = e

EXPERIMENT_MANAGER_AVAILABLE = False
_exp_manager_import_error = None
try:
    from . import experiment_manager
    EXPERIMENT_MANAGER_AVAILABLE = True
except ImportError as e:
    _exp_manager_import_error = e

# --- Optional for Analysis/Reporting ---
META_ANALYSIS_AVAILABLE = False
_meta_analysis_import_error = None
try:
    from . import meta_analysis
    META_ANALYSIS_AVAILABLE = True
except ImportError as e:
    _meta_analysis_import_error = e

INFO_FLOW_AVAILABLE = False
_info_flow_import_error = None
if getattr(config_module.cfg, 'enable_information_flow', True):
    try:
        from .validation import information_flow as info_flow_module
        INFO_FLOW_AVAILABLE = info_flow_module.PYINFORM_AVAILABLE
    except ImportError as e:
        _info_flow_import_error = e

# --- Phase 4 Imports (LLM Interface, Agents, Orchestrator) ---
LLM_INTERFACE_AVAILABLE = False
_llm_interface_import_error = None
AGENTS_AVAILABLE = False
_agents_import_error = None
ORCHESTRATOR_AVAILABLE = False
_orchestrator_import_error = None
REPORTING_AVAILABLE = False # Reporting module itself
_reporting_import_error = None


# Import LLM interface first as agents depend on it
if getattr(config_module.cfg, 'enable_orchestration', False) or getattr(config_module.cfg, 'enable_openai_reporting', True):
     try:
         from . import llm_interface
         LLM_INTERFACE_AVAILABLE = True
     except ImportError as e:
         _llm_interface_import_error = e

# Import Reporting module (needs LLM interface conceptually, but import separately)
if getattr(config_module.cfg, 'enable_openai_reporting', True):
     try:
         from . import reporting
         REPORTING_AVAILABLE = True
     except ImportError as e:
         _reporting_import_error = e


# Import Agents and Orchestrator only if Orchestration is enabled
if getattr(config_module.cfg, 'enable_orchestration', False):
    # Agents depend on LLM Interface
    if LLM_INTERFACE_AVAILABLE:
        try:
            from . import agents
            AGENTS_AVAILABLE = True # Agents module imported
            # Further check if agents module itself had internal import issues
            if hasattr(agents, 'MODULES_AVAILABLE') and not agents.MODULES_AVAILABLE:
                 AGENTS_AVAILABLE = False # Mark as unavailable if internal imports failed
                 _agents_import_error = getattr(agents, '_import_error', ImportError("Agents module internal import failed"))
        except ImportError as e:
            _agents_import_error = e
            AGENTS_AVAILABLE = False
    else:
         AGENTS_AVAILABLE = False
         _agents_import_error = ImportError("Cannot import agents, llm_interface unavailable")

    # Orchestrator depends on Agents
    if AGENTS_AVAILABLE:
        try:
            from . import orchestrator
            ORCHESTRATOR_AVAILABLE = True # Orchestrator module imported
            if hasattr(orchestrator, 'MODULES_AVAILABLE') and not orchestrator.MODULES_AVAILABLE:
                 ORCHESTRATOR_AVAILABLE = False
                 _orchestrator_import_error = getattr(orchestrator, '_import_error', ImportError("Orchestrator module internal import failed"))
        except ImportError as e:
            _orchestrator_import_error = e
            ORCHESTRATOR_AVAILABLE = False
    else:
         ORCHESTRATOR_AVAILABLE = False
         _orchestrator_import_error = ImportError("Cannot import orchestrator, agents unavailable")



# --- Set LOKY_MAX_CPU_COUNT ---
# Attempt to silence joblib/loky warning about physical cores.
# Use the logical core count detected by os.cpu_count() as a safe default.
try:
    logical_cores = os.cpu_count()
    if logical_cores:
        os.environ['LOKY_MAX_CPU_COUNT'] = str(logical_cores)
        # Use print temporarily for debugging import order issues, then switch to logger
        # print(f"Set LOKY_MAX_CPU_COUNT = {logical_cores}")
except Exception as e_loky:
    # Use print temporarily, as logging might not be set up yet
    print(f"Warning: Failed to set LOKY_MAX_CPU_COUNT: {e_loky}")
# --- End Set LOKY_MAX_CPU_COUNT ---

# --- Configuration & Setup ---
base_config = config_module.cfg # Use base_config to refer to the global config object

# --- Basic Logging Setup ---
# Configure logging early to capture import warnings/errors
root_logger = logging.getLogger()
log_filepath = os.path.join(getattr(base_config, 'log_dir', 'results/logs'), "simulation_suite.log")
# Ensure log directory exists before setting up handler
os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
# Prevent duplicate handlers if run multiple times (e.g., in notebooks)
if root_logger.hasHandlers():
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Define logger after basicConfig

# --- Ensure other output directories exist (now that logging is safe) ---
os.makedirs(base_config.plot_dir, exist_ok=True)
os.makedirs(base_config.log_dir, exist_ok=True) # Redundant but safe
if base_config.save_results and base_config.enable_openai_reporting:
    os.makedirs(base_config.report_dir, exist_ok=True)
if base_config.enable_classical_simulation:
     os.makedirs(base_config.classical_results_dir, exist_ok=True)
# Ensure main results directory exists (for DataFrame)
os.makedirs(base_config.results_dir, exist_ok=True)
# Ensure directory for saved observable arrays exists
if getattr(base_config, 'save_observable_arrays', False):
    os.makedirs(os.path.join(base_config.results_dir, 'observables_data'), exist_ok=True)
# Ensure orchestrator state directory exists
if getattr(base_config, 'enable_orchestration', False):
    os.makedirs(base_config.orchestrator_state_dir, exist_ok=True)


# Log import status warnings now that logging is configured
if not CORE_MODULES_OK:
    logger.critical("Core simulation modules failed to import. Functionality severely limited.")
if getattr(base_config, 'enable_openai_reporting', True) and not REPORTING_AVAILABLE:
    logger.error(f"AI reporting disabled due to reporting module import error: {_reporting_import_error}")
if getattr(base_config, 'enable_information_flow', True) and not INFO_FLOW_AVAILABLE:
    logger.warning(f"Info flow analysis disabled due to information_flow module import error: {_info_flow_import_error}")
if getattr(base_config, 'enable_classical_simulation', False) and not CLASSICAL_SIM_AVAILABLE:
    logger.error(f"Classical simulation disabled due to classical_mechanics import error: {_classical_import_error}")
if not RESULTS_HANDLER_AVAILABLE:
    logger.critical(f"Results handler module import failed: {_results_handler_import_error}. Suite cannot run.")
if not EXPERIMENT_MANAGER_AVAILABLE:
    logger.critical(f"Experiment manager module import failed: {_exp_manager_import_error}. Suite cannot run.")
if not META_ANALYSIS_AVAILABLE:
    logger.warning(f"Meta analysis module import failed: {_meta_analysis_import_error}. Post-processing may be limited.")
if getattr(base_config, 'enable_orchestration', False) and not ORCHESTRATOR_AVAILABLE:
    logger.error(f"Orchestration disabled due to orchestrator module import error: {_orchestrator_import_error}")
# Log status of Phase 4 components specifically
if getattr(base_config, 'enable_orchestration', False) or getattr(base_config, 'enable_openai_reporting', True):
    if not LLM_INTERFACE_AVAILABLE:
        logger.error(f"LLM Interface import failed: {_llm_interface_import_error}")
if getattr(base_config, 'enable_orchestration', False):
    if not AGENTS_AVAILABLE:
        logger.error(f"Agents import failed: {_agents_import_error}")

def calculate_l2_state_difference(label1, label2, results_cache, config):
    """Calculates the L2 distance between the final states of two runs."""
    if not results_cache:
        logger.warning(f"Cannot calculate L2 diff for {label1} vs {label2}: results_cache is empty.")
        return np.nan

    run1_data = results_cache.get(label1)
    run2_data = results_cache.get(label2)

    if not run1_data or not run1_data.get('success'):
        logger.warning(f"Cannot calculate L2 diff: Run '{label1}' missing, failed, or has no data.")
        return np.nan
    if not run2_data or not run2_data.get('success'):
        logger.warning(f"Cannot calculate L2 diff: Run '{label2}' missing, failed, or has no data.")
        return np.nan

    psi1_key = f'psi_final_{label1}'
    psi2_key = f'psi_final_{label2}'

    psi1 = run1_data.get('observables', {}).get(psi1_key)
    psi2 = run2_data.get('observables', {}).get(psi2_key)
    dx = getattr(config, 'dx', None) # Get dx from the main config

    if psi1 is None:
        logger.warning(f"Cannot calculate L2 diff: Final state '{psi1_key}' not found.")
        return np.nan
    if psi2 is None:
        logger.warning(f"Cannot calculate L2 diff: Final state '{psi2_key}' not found.")
        return np.nan
    if dx is None:
        logger.error("Cannot calculate L2 diff: 'dx' not found in config.")
        return np.nan
    if not isinstance(psi1, np.ndarray) or not isinstance(psi2, np.ndarray) or psi1.shape != psi2.shape:
        logger.warning(f"Cannot calculate L2 diff: State shapes mismatch or not arrays for {label1} vs {label2}.")
        return np.nan

    try:
        diff = psi1 - psi2
        # L2 distance = sqrt( integral |psi1 - psi2|^2 dx )
        # Approximation: sqrt( sum(|psi1_i - psi2_i|^2) * dx )
        l2_distance_sq = np.sum(np.abs(diff)**2) * dx
        l2_distance = np.sqrt(l2_distance_sq)

        logger.info(f"Calculated L2 distance between '{label1}' and '{label2}': {l2_distance:.4e}")
        return l2_distance
    except Exception as e:
        logger.error(f"Error calculating L2 distance for {label1} vs {label2}: {e}", exc_info=True)
        return np.nan
# --- END NEW HELPER FUNCTION ---

# --- Helper Function for Schedule Lookup ---
def _get_value_from_schedule(schedule, time, default):
    """
    Gets the appropriate value from a schedule list based on time.
    Schedule format: [(time1, value1), (time2, value2), ...] sorted by time.
    Returns the value active at the given 'time'.
    """
    if schedule is None or not schedule:
        return default
    if not isinstance(schedule, list) or not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in schedule):
        logger.warning(f"Invalid schedule format encountered: {schedule}. Using default: {default}")
        return default

    # Find the last time point in the schedule that is less than or equal to the current time
    current_value = default # Start with default (or value at t=0 if defined)
    if schedule[0][0] <= time:
         current_value = schedule[0][1]

    for t_point, val_point in schedule:
        if time >= t_point:
            current_value = val_point
        else:
            # Since schedule is sorted, we found the interval
            break
    return current_value

# --- Helper to get config value ---
def _get_config_value(config_obj, attr, default):
    """Safely gets an attribute from config_obj or returns default."""
    if isinstance(config_obj, dict):
        return config_obj.get(attr, default)
    return getattr(config_obj, attr, default) if config_obj else default

# --- Simulation Execution Function ---
def run_single_experiment(current_config_obj, run_label="simulation"):
    """
    Runs ONE quantum simulation experiment and returns structured results.
    Handles setup, signal generation, evolution, basic analysis, and plotting.
    Includes handling for parameter scheduling and triggering classical simulation.
    Optionally saves full observable arrays based on config.
    """
    start_time = time.time()
    logger.info(f"--- Starting Single Experiment Run: {run_label} ---")
    # Log key parameters for this specific run
    driver_type = getattr(current_config_obj, 'driver_type', 'lorenz')
    scheduling_enabled = getattr(current_config_obj, 'enable_parameter_scheduling', False)
    log_extra = f"Driver={driver_type}, Sched={scheduling_enabled}"
    lorenz_ic_val = getattr(current_config_obj, 'lorenz_initial_state', None)
    if driver_type == 'lorenz' and lorenz_ic_val is not None:
        lorenz_ic_str = np.array2string(np.array(lorenz_ic_val), precision=5, separator=',')
        log_extra += f", Lorenz IC=[{lorenz_ic_str}] (Comp: {getattr(current_config_obj, 'lorenz_use_component', 'z')})"
    elif driver_type == 'rossler' and hasattr(current_config_obj, 'rossler_initial_state'):
        rossler_ic_str = np.array2string(np.array(current_config_obj.rossler_initial_state), precision=5, separator=',')
        log_extra += f", Rossler IC=[{rossler_ic_str}] (Comp: {getattr(current_config_obj, 'rossler_use_component', 'x')})"
    elif driver_type == 'logistic_map':
         log_extra += f", Logistic r={current_config_obj.logistic_r:.3f}, x0={current_config_obj.logistic_initial_x:.3f}"
    elif driver_type == 'filtered_noise':
         log_extra += f", Noise Type={getattr(current_config_obj, 'noise_type', 'N/A')}, Filter={getattr(current_config_obj, 'filter_type', 'N/A')}"
    elif driver_type == 'sine':
        log_extra += f", Sine A={current_config_obj.sine_amplitude:.3f}, Freq={current_config_obj.sine_frequency:.3f}"
    elif driver_type == 'quasi_periodic':
         log_extra += f", Quasi A1={current_config_obj.quasi_amplitude1:.3f}, F1={current_config_obj.quasi_frequency1:.3f}, A2={current_config_obj.quasi_amplitude2:.3f}, F2={current_config_obj.quasi_frequency2:.3f}"

    qic_str = f"x0={current_config_obj.x0:.3f}, k0={current_config_obj.k0_psi:.3f}"
    alpha_schedule = getattr(current_config_obj, 'alpha_schedule', None) if scheduling_enabled else None
    epsilon_schedule = getattr(current_config_obj, 'epsilon_schedule', None) if scheduling_enabled else None
    alpha_log = "Sched" if alpha_schedule else f"{current_config_obj.alpha:.3f}"
    epsilon_log = "Sched" if epsilon_schedule else f"{current_config_obj.epsilon:.3f}"
    logger.info(f"Quantum Config: epsilon={epsilon_log}, alpha={alpha_log}, "
                f"{log_extra}, Quantum IC=({qic_str})")
    logger.info(f"Noise Injection: {getattr(current_config_obj, 'inject_runtime_noise', False)}")

    # --- Initialize Results Dictionary ---
    result_data = {
        "config": copy.deepcopy(current_config_obj), # Store config used for this run
        "run_label": run_label,
        "success": False,
        "error_message": None,
        "observables": {}, # Store time series, final state, plots etc.
        "metrics": {}, # Store scalar metrics like LLE, DET
        "classical_results": None
    }

    try:
        # --- Setup Solver and Initial State ---
        solver = schrodinger.SplitOperatorSolver(current_config_obj)
        psi0 = solver.initial_wavefunction(current_config_obj)
        psi = psi0.copy()

        # --- Generate Driving Signal ---
        logger.info(f"Generating external driving signal (type: {current_config_obj.driver_type})...")
        t_signal, z_signal, driver_full_state = signals.get_driving_signal(current_config_obj)
        if t_signal is None or z_signal is None:
            raise RuntimeError(f"Failed to generate driving signal for run '{run_label}'.")
        if not np.allclose(t_signal, current_config_obj.t_grid):
             raise RuntimeError("CRITICAL TIME GRID MISMATCH post signal generation.")

        # --- Initialize History and Observables ---
        generated_plots_this_run = [] # Collect plots generated for this specific run
        observables = {}
        if current_config_obj.track_norm: observables["Norm Conservation"] = []
        if current_config_obj.track_position: observables["Position <x>"] = []
        if current_config_obj.track_momentum: observables["Momentum <p>"] = []
        if current_config_obj.track_energy: observables["Energy <H>"] = []
        if current_config_obj.track_spatial_variance: observables["Spatial Variance Var(x)"] = []
        if current_config_obj.track_shannon_entropy: observables["Shannon Entropy S(x)"] = []
        if current_config_obj.enable_information_flow:
            driver_obs_name = getattr(current_config_obj, 'info_flow_observable_driver', 'driver_signal')
            observables[f'driver_signal_{driver_obs_name}'] = z_signal.copy()
            if driver_full_state is not None: observables['driver_full_state'] = driver_full_state.copy()

        keep_history = current_config_obj.animate or _get_config_value(current_config_obj, 'save_heatmap', True)
        psi_history = None
        if keep_history:
            psi_history = np.zeros((current_config_obj.M, current_config_obj.N), dtype=complex)
            psi_history[0, :] = psi
        # Initialize potential history only if animating
        potential_history = None
        if current_config_obj.animate:
            potential_history = np.zeros((current_config_obj.M, current_config_obj.N), dtype=float)
            z_initial = z_signal[0] if z_signal is not None and len(z_signal) > 0 else 0.0
            alpha_initial = _get_value_from_schedule(alpha_schedule, 0.0, current_config_obj.alpha)
            epsilon_initial = _get_value_from_schedule(epsilon_schedule, 0.0, current_config_obj.epsilon)
            potential_history[0, :] = dynamic_potential(
                current_config_obj.x_grid, current_config_obj.t_grid[0], z_initial,
                current_config_obj.V0_static, epsilon_initial, current_config_obj.k_potential,
                current_config_obj.omega_potential, alpha_initial
            )

        # --- Plot driver trajectory (if enabled and applicable) ---
        is_stateful_driver = current_config_obj.driver_type in ['lorenz', 'rossler']
        if is_stateful_driver and driver_full_state is not None:
             safe_label = plot_wavefunction._sanitize_filename(run_label)
             traj_fname = f"{safe_label}_{current_config_obj.driver_type}_trajectory.png"
             # Call plot function - it will check save_plots_per_run internally
             saved_traj_path = plot_wavefunction.plot_chaos_trajectory(
                 t_signal, driver_full_state, z_signal, filename=traj_fname, config_obj=current_config_obj
             )
             if saved_traj_path:
                 generated_plots_this_run.append(os.path.basename(saved_traj_path))


        # --- Time Evolution Loop ---
        logger.info("Starting quantum time evolution...")
        for i in tqdm(range(current_config_obj.M), desc=f"Evolving Q {run_label:<25}", ncols=100, leave=False):
            t_now = current_config_obj.t_grid[i]
            z_now = z_signal[i]
            alpha_now = _get_value_from_schedule(alpha_schedule, t_now, current_config_obj.alpha)
            epsilon_now = _get_value_from_schedule(epsilon_schedule, t_now, current_config_obj.epsilon)

            V_current = dynamic_potential(
                current_config_obj.x_grid, t_now, z_now,
                current_config_obj.V0_static, epsilon_now, current_config_obj.k_potential,
                current_config_obj.omega_potential, alpha_now
            )
            if getattr(current_config_obj, 'inject_runtime_noise', False):
                V_current += np.random.normal(0, getattr(current_config_obj, 'noise_level_potential', 1e-4), V_current.shape)

            if potential_history is not None: # Store potential only if animating
                potential_history[i, :] = V_current

            # Calculate observables
            if current_config_obj.track_norm:
                observables.setdefault("Norm Conservation", []).append(analysis.check_norm_conservation(psi, current_config_obj.dx))
            if current_config_obj.track_position:
                observables.setdefault("Position <x>", []).append(analysis.position_expectation(psi, current_config_obj.x_grid, current_config_obj.dx))
            if current_config_obj.track_momentum:
                observables.setdefault("Momentum <p>", []).append(np.real(analysis.momentum_expectation(psi, current_config_obj.k_grid, current_config_obj.hbar)))
            if current_config_obj.track_energy:
                observables.setdefault("Energy <H>", []).append(np.real(analysis.energy_expectation(psi, current_config_obj.x_grid, V_current, current_config_obj.k_grid, current_config_obj.dx, current_config_obj.hbar, current_config_obj.m)))
            if current_config_obj.track_spatial_variance:
                observables.setdefault("Spatial Variance Var(x)", []).append(analysis.spatial_variance(psi, current_config_obj.x_grid, current_config_obj.dx))
            if current_config_obj.track_shannon_entropy:
                observables.setdefault("Shannon Entropy S(x)", []).append(analysis.shannon_entropy_spatial(psi, current_config_obj.dx))

            # Evolve state
            if i < current_config_obj.M - 1:
                 psi = solver.evolve(psi, V_current)
                 if keep_history:
                     psi_history[i + 1, :] = psi

        # --- End Time Evolution Loop ---
        logger.info("Quantum time evolution finished.")
        end_time_quantum = time.time()
        logger.info(f"Quantum experiment run '{run_label}' took {end_time_quantum - start_time:.2f} seconds.")

        # Convert observable lists to arrays and check for NaNs
        for key in list(observables.keys()):
            if isinstance(observables[key], list):
                observables[key] = np.array(observables[key])
                if np.any(np.isnan(observables[key])) or np.any(np.isinf(observables[key])):
                    logger.warning(f"NaN or Inf detected in observable '{key}' for run '{run_label}'.")

        # Store final state
        final_psi = None
        if keep_history and psi_history is not None and psi_history.shape[0] > 0:
            final_psi = psi_history[-1,:].copy()
        elif psi is not None:
             final_psi = psi.copy()

        if final_psi is not None:
            observables[f'psi_final_{run_label}'] = final_psi
        else:
            logger.warning(f"Could not store final state for {run_label}.")

        # --- Post-Simulation Analysis ---
        metrics_dict = {}
        generated_analysis_plots = []
        safe_label = plot_wavefunction._sanitize_filename(run_label)

        # LLE
        if current_config_obj.enable_lle_calculation and metrics.NOLDS_AVAILABLE:
            obs_key_lle = current_config_obj.observable_for_lle
            time_series = observables.get(obs_key_lle)
            if time_series is not None and len(time_series) > 100:
                plot_fname_lle, lle_plot_file = None, None
                if current_config_obj.lle_debug_plot:
                    plot_fname_lle = f"validation_lle_debug_{plot_wavefunction._sanitize_filename(obs_key_lle)}_{safe_label}.png"
                    os.makedirs(current_config_obj.plot_dir, exist_ok=True)
                    lle_plot_file = os.path.join(current_config_obj.plot_dir, plot_fname_lle)

                lyap_exp = metrics.compute_largest_lyapunov_exponent(
                    time_series, dt=current_config_obj.dt_quantum, emb_dim=current_config_obj.lle_emb_dim,
                    lag=current_config_obj.lle_lag, fit_method=current_config_obj.lle_fit_method,
                    debug_plot=current_config_obj.lle_debug_plot, plot_file=lle_plot_file
                )
                metrics_dict['LLE'] = lyap_exp
                metrics_dict['LLE_Observable'] = obs_key_lle
                if not np.isnan(lyap_exp):
                    logger.info(f"LLE ({run_label}, obs='{obs_key_lle}'): {lyap_exp:.4e}")
                    if plot_fname_lle and lle_plot_file and os.path.exists(lle_plot_file):
                        generated_analysis_plots.append(plot_fname_lle)
            else:
                logger.warning(f"LLE skipped for {run_label}: Observable '{obs_key_lle}' not found or too short.")
                metrics_dict['LLE'] = np.nan

        # FFT
        if current_config_obj.enable_fft_analysis:
            obs_key_fft = current_config_obj.observable_for_fft
            time_series = observables.get(obs_key_fft)
            if time_series is not None and len(time_series) > 1:
                freqs, spec = metrics.perform_fft_analysis(time_series, current_config_obj.dt_quantum)
                if freqs is not None and spec is not None and len(freqs) > 0:
                    fname_fft = f"validation_fft_{plot_wavefunction._sanitize_filename(obs_key_fft)}_{safe_label}.png"
                    fpath_fft = os.path.join(current_config_obj.plot_dir, fname_fft)
                    # Plot only if save_plots_per_run is True
                    if _get_config_value(current_config_obj, 'save_plots_per_run', False):
                        plt.figure(figsize=(10, 5))
                        plt.plot(freqs[1:], spec[1:], marker='.', linestyle='-', markersize=3)
                        plt.xlabel("Frequency (Hz or 1/time_unit)")
                        plt.ylabel("Amplitude Spectrum")
                        plt.title(f"FFT Analysis of {obs_key_fft} ({run_label})")
                        plt.grid(True, which='both')
                        plt.yscale('log')
                        plt.xscale('log')
                        plt.tight_layout()
                        saved_path = plot_wavefunction._save_or_show_plot(plt.gcf(), fpath_fft, True) # Assume saving
                        if saved_path:
                            generated_analysis_plots.append(os.path.basename(saved_path))
                    else:
                        logger.debug(f"Skipping FFT plot save for {run_label} due to config.")

        # RQA
        if current_config_obj.enable_rqa_analysis and metrics.RQA_AVAILABLE:
            obs_key_rqa = current_config_obj.observable_for_rqa
            time_series = observables.get(obs_key_rqa)
            if time_series is not None and len(time_series) > 1:
                # --- Store the full RQA output ---
                rqa_full_output = metrics.compute_rqa_metrics( # Store the whole dict
                    time_series=time_series, emb_dim=current_config_obj.rqa_embedding_dimension,
                    time_delay=current_config_obj.rqa_time_delay, threshold=current_config_obj.rqa_neighbourhood_threshold,
                    threshold_type=current_config_obj.rqa_neighbourhood_type, similarity_measure=current_config_obj.rqa_similarity_measure,
                    theiler_corrector=current_config_obj.rqa_theiler_corrector, min_diag_len=current_config_obj.rqa_min_diag_len,
                    min_vert_len=current_config_obj.rqa_min_vert_len, min_white_vert_len=current_config_obj.rqa_min_white_vert_len,
                    normalize=current_config_obj.rqa_normalize, analysis_class=metrics.Classic
                )
                # Store the complete result including the matrix
                result_data['rqa_full_output'] = rqa_full_output

                if rqa_full_output: # Check if calculation was successful
                    # Add scalar metrics to the main metrics dict for DataFrame
                    metrics_dict.update({k: v for k, v in rqa_full_output.items() if k != 'RP_Matrix'})
                    metrics_dict['RQA_Observable'] = obs_key_rqa
                    logger.info(f"RQA DET ({run_label}, obs='{obs_key_rqa}'): {rqa_full_output.get('DET', np.nan):.4f}, LAM: {rqa_full_output.get('LAM', np.nan):.4f}, ENTR: {rqa_full_output.get('ENTR', np.nan):.4f}")
                    # Plot recurrence matrix if available and configured
                    if 'RP_Matrix' in rqa_full_output:
                         fname_rqa = f"validation_rqa_{plot_wavefunction._sanitize_filename(obs_key_rqa)}_{safe_label}.png"
                         # Pass the dict containing the matrix to the plot function
                         saved_rqa_path = plot_wavefunction.plot_recurrence(rqa_full_output, filename=fname_rqa, config_obj=current_config_obj)
                         if saved_rqa_path:
                             generated_analysis_plots.append(os.path.basename(saved_rqa_path))
                else: # RQA calculation failed
                    metrics_dict.update({'DET': np.nan, 'LAM': np.nan, 'ENTR': np.nan}) # Mark RQA as failed
            else:
                logger.warning(f"RQA skipped for {run_label}: Observable '{obs_key_rqa}' not found or invalid.")
                metrics_dict.update({'DET': np.nan, 'LAM': np.nan, 'ENTR': np.nan})
        # CWT
        if current_config_obj.enable_cwt_analysis and metrics.CWT_AVAILABLE:
            obs_key_cwt = current_config_obj.observable_for_cwt
            time_series = observables.get(obs_key_cwt)
            if time_series is not None and len(time_series) > 1:
                coeffs, freqs = metrics.perform_cwt(
                    time_series, dt=current_config_obj.dt_quantum,
                    wavelet_type=current_config_obj.cwt_wavelet_type, scales=current_config_obj.cwt_scales
                )
                if coeffs is not None and freqs is not None:
                    metrics_dict['CWT_coeffs_computed'] = True
                    fname_cwt = f"validation_cwt_{plot_wavefunction._sanitize_filename(obs_key_cwt)}_{safe_label}.png"
                    # Call plot function - it checks save_plots_per_run internally
                    saved_cwt_path = plot_wavefunction.plot_cwt_scalogram(
                        coeffs, current_config_obj.t_grid, freqs, filename=fname_cwt, config_obj=current_config_obj
                    )
                    if saved_cwt_path:
                        generated_analysis_plots.append(os.path.basename(saved_cwt_path))
                else:
                    metrics_dict['CWT_coeffs_computed'] = False
            else:
                logger.warning(f"CWT skipped for {run_label}: Observable '{obs_key_cwt}' not found or invalid.")
                metrics_dict['CWT_coeffs_computed'] = False

        # Correlation Dimension
        if current_config_obj.enable_correlation_dimension and metrics.NOLDS_AVAILABLE:
            obs_key_cd = current_config_obj.observable_for_corr_dim
            time_series = observables.get(obs_key_cd)
            if time_series is not None and len(time_series) > 100:
                cd_plot_file = None
                cd_plot_fname = None
                if current_config_obj.corr_dim_debug_plot:
                     cd_plot_fname = f"validation_cd_debug_{plot_wavefunction._sanitize_filename(obs_key_cd)}_{safe_label}.png"
                     os.makedirs(current_config_obj.plot_dir, exist_ok=True)
                     cd_plot_file = os.path.join(current_config_obj.plot_dir, cd_plot_fname)

                corr_dim = metrics.compute_correlation_dimension(
                    time_series, emb_dim=current_config_obj.corr_dim_embedding_dim,
                    lag=current_config_obj.corr_dim_lag, rvals_count=current_config_obj.corr_dim_rvals_count,
                    fit=current_config_obj.lle_fit_method, debug_plot=current_config_obj.corr_dim_debug_plot,
                    plot_file=cd_plot_file
                )
                metrics_dict['CorrDim'] = corr_dim
                metrics_dict['CorrDim_Observable'] = obs_key_cd
                metrics_dict['CorrDim_EmbDim'] = current_config_obj.corr_dim_embedding_dim
                if not np.isnan(corr_dim):
                    logger.info(f"CorrDim ({run_label}, obs='{obs_key_cd}', Emb={current_config_obj.corr_dim_embedding_dim}): {corr_dim:.4f}")
                    if cd_plot_fname and cd_plot_file and os.path.exists(cd_plot_file):
                        generated_analysis_plots.append(cd_plot_fname)
            else:
                logger.warning(f"CorrDim skipped for {run_label}: Observable '{obs_key_cd}' not found or too short.")
                metrics_dict['CorrDim'] = np.nan

        # Information Flow
        if current_config_obj.enable_information_flow and INFO_FLOW_AVAILABLE:
            obs_key_sys = current_config_obj.info_flow_observable_system
            driver_obs_name = getattr(current_config_obj, 'info_flow_observable_driver', 'driver_signal')
            driver_signal_key = f'driver_signal_{driver_obs_name}'
            driver_signal = observables.get(driver_signal_key)
            system_series = observables.get(obs_key_sys)
            if driver_signal is not None and system_series is not None and len(system_series) == len(driver_signal) and len(system_series) > current_config_obj.info_flow_k:
                te_val = info_flow_module.calculate_transfer_entropy(
                    source_series=driver_signal, target_series=system_series,
                    k=current_config_obj.info_flow_k, lag=current_config_obj.info_flow_lag,
                    local=current_config_obj.info_flow_local
                )
                if te_val is not None:
                    # Store mean if local, otherwise store the scalar value
                    mean_te = np.nanmean(te_val) if current_config_obj.info_flow_local and isinstance(te_val, np.ndarray) else te_val
                    metrics_dict['TransferEntropy'] = mean_te
                    metrics_dict['TE_Source'] = driver_obs_name
                    metrics_dict['TE_Target'] = obs_key_sys
                    metrics_dict['TE_k'] = current_config_obj.info_flow_k
                    log_val = metrics_dict['TransferEntropy']
                    if not np.isnan(log_val):
                        logger.info(f"TE ({run_label}, {driver_obs_name}->'{obs_key_sys}', k={current_config_obj.info_flow_k}): {log_val:.4f} {'(avg)' if current_config_obj.info_flow_local else ''}")
                else:
                    metrics_dict['TransferEntropy'] = np.nan # Mark TE calc failure
            else:
                logger.warning(f"Info flow skipped for {run_label}: Length mismatch, too short, or missing series.")
                metrics_dict['TransferEntropy'] = np.nan

        # --- Save Full Observable Arrays ---
        saved_array_filenames = []
        # *** Check this condition ***
        if _get_config_value(current_config_obj, 'save_observable_arrays', False):
            observables_to_save = _get_config_value(current_config_obj, 'observables_to_save', [])
            if observables_to_save:
                obs_data_dir = os.path.join(_get_config_value(current_config_obj, 'results_dir', '.'), 'observables_data')
                os.makedirs(obs_data_dir, exist_ok=True)
                safe_label_obs = plot_wavefunction._sanitize_filename(run_label)
                for obs_name in observables_to_save:
                     # *** Check this condition: Is 'Position <x>' in observables dict? ***
                    if obs_name in observables and isinstance(observables[obs_name], np.ndarray) and observables[obs_name].size > 1:
                        obs_array = observables[obs_name]
                        safe_obs_name = plot_wavefunction._sanitize_filename(obs_name)
                        obs_filename = f"{safe_label_obs}_observable_{safe_obs_name}.npy"
                        obs_filepath = os.path.join(obs_data_dir, obs_filename)
                        try:
                            # *** Check this path construction ***
                            np.save(obs_filepath, obs_array)
                            logger.info(f"Saved observable array to {obs_filepath}")
                            saved_array_filenames.append(obs_filename)
                        except Exception as e_save_obs:
                            logger.error(f"Failed to save observable array {obs_name} to {obs_filepath}: {e_save_obs}")
                    else:
                        logger.warning(f"Requested observable '{obs_name}' not suitable for saving for run {run_label}.")
        # Ensure the results dictionary includes the filenames if saved
        if saved_array_filenames:
            result_data["observables"]["saved_array_files"] = saved_array_filenames
        # --- Generate Core Plots for this run (Heatmap, Observables, Animation) ---
        safe_label_core = plot_wavefunction._sanitize_filename(run_label)
        fname_heatmap = f"{safe_label_core}_probability_heatmap.png"
        fname_obs = f"{safe_label_core}_observables.png"
        fname_anim = f"{safe_label_core}_animation.mp4"
        # Call plot functions - they check flags internally and return filename if saved
        saved_heatmap_path = plot_wavefunction.plot_probability_heatmap(
            psi_history, current_config_obj.x_grid, current_config_obj.t_grid, filename=fname_heatmap, config_obj=current_config_obj
        ) if keep_history else None
        saved_obs_path = plot_wavefunction.plot_observables(
            current_config_obj.t_grid, observables, filename=fname_obs, config_obj=current_config_obj
        )
        saved_anim_path = plot_wavefunction.animate_wavefunction(
            psi_history, current_config_obj.x_grid, current_config_obj.t_grid,
            potential_history=potential_history, filename=fname_anim, config_obj=current_config_obj
        ) if keep_history else None
        # Collect filenames of plots successfully generated *for this run*
        if saved_heatmap_path:
            generated_plots_this_run.append(os.path.basename(saved_heatmap_path))
        if saved_obs_path:
            generated_plots_this_run.append(os.path.basename(saved_obs_path))
        if saved_anim_path:
            generated_plots_this_run.append(os.path.basename(saved_anim_path))
        generated_plots_this_run.extend(generated_analysis_plots) # Add analysis plots too

        # --- Run Classical Simulation ---
        classical_run_results = None
        if _get_config_value(current_config_obj, 'enable_classical_simulation', False) and CLASSICAL_SIM_AVAILABLE:
            logger.info(f"--- Running Classical Simulation for config context: {run_label} ---")
            classical_config_to_use = current_config_obj
            classical_t_driver, classical_z_driver = None, None
            if _get_config_value(classical_config_to_use, 'classical_enable_driver_coupling', False):
                classical_t_driver = t_signal
                classical_z_driver = z_signal
                logger.info("Passing quantum driver signal z(t) to classical simulation.")

            classical_t, classical_states = classical_sim.simulate_classical_system(
                classical_config_to_use, t_signal=classical_t_driver, z_signal=classical_z_driver
            )
            if classical_t is not None and classical_states is not None:
                logger.info(f"Classical simulation successful for {run_label} context.")
                classical_metrics = classical_analysis.analyze_classical_trajectory(
                    classical_t, classical_states, classical_config_to_use
                )
                # --- Extract relevant classical config params ---
                classical_config_dict = {
                    k.replace('classical_', ''): v
                    for k, v in vars(classical_config_to_use).items()
                    if k.startswith('classical_')
                }
                # --- Ensure model_type is included ---
                if 'model_type' not in classical_config_dict:
                     classical_config_dict['model_type'] = getattr(classical_config_to_use, 'classical_model_type', 'N/A')
                # --- Store results ---
                classical_run_results = {
                    "metrics": classical_metrics,
                    "config": classical_config_dict # Store the extracted dict
                }
                result_data["classical_results"] = classical_run_results
            else:
                logger.error(f"Classical simulation failed for {run_label} context.")
                # Store failure info, ensuring config part might still exist if needed
                classical_config_dict = {
                    k.replace('classical_', ''): v
                    for k, v in vars(classical_config_to_use).items()
                    if k.startswith('classical_')
                }
                if 'model_type' not in classical_config_dict:
                     classical_config_dict['model_type'] = getattr(classical_config_to_use, 'classical_model_type', 'N/A')
                result_data["classical_results"] = {
                    "error": "Classical simulation failed",
                    "config": classical_config_dict,
                     "metrics": {}
                 }
        elif _get_config_value(current_config_obj, 'enable_classical_simulation', False):
            logger.warning(f"Classical sim enabled for {run_label} but modules unavailable.")
            result_data["classical_results"] = {"error": "Classical modules unavailable"}

        # --- Finalize Result Dictionary ---
        result_data["observables"] = observables # Reassign potentially modified dict
        result_data["metrics"] = metrics_dict
        result_data["success"] = True
        result_data["observables"]["plot_files_run"] = sorted(list(set(generated_plots_this_run))) # Store unique plot files generated for this run

        logger.info(f"--- Experiment Run {run_label} Completed Successfully ---")

    except Exception as e:
        logger.error(f"Experiment run '{run_label}' failed: {e}", exc_info=True)
        result_data["success"] = False
        result_data["error_message"] = str(e)
        if "observables" not in result_data: result_data["observables"] = {}
        if "metrics" not in result_data: result_data["metrics"] = {}
        if "plot_files_run" not in result_data["observables"]: result_data["observables"]["plot_files_run"] = []
        result_data["classical_results"] = {"error": f"Quantum simulation failed: {e}"}

    # --- Clean up large arrays before returning ---
    # Keep psi_final, driver_signal, and key analysis observables
    if 'observables' in result_data:
        save_arrays_enabled = _get_config_value(current_config_obj, 'save_observable_arrays', False)
        observables_to_save = _get_config_value(current_config_obj, 'observables_to_save', []) if save_arrays_enabled else []

        # Define observables crucial for standard meta-analysis/composite plots
        common_analysis_obs = [
            "Position <x>", "Momentum <p>", "Energy <H>",
            "Spatial Variance Var(x)", "Shannon Entropy S(x)",
            # Add other observables if you create composite plots for them
        ]
        # Also add the specific observable names used for analysis if not already covered
        common_analysis_obs.append(getattr(current_config_obj, 'observable_for_lle', "Position <x>"))
        common_analysis_obs.append(getattr(current_config_obj, 'observable_for_rqa', "Position <x>"))
        common_analysis_obs.append(getattr(current_config_obj, 'observable_for_cwt', "Position <x>"))
        common_analysis_obs.append(getattr(current_config_obj, 'observable_for_corr_dim', "Position <x>"))
        common_analysis_obs = list(set(common_analysis_obs)) # Ensure uniqueness

        keys_to_del_obs = []
        for k, v in result_data['observables'].items():
            if isinstance(v, np.ndarray) and v.size > 1000: # Only consider large arrays
                if k not in observables_to_save and \
                   not k.startswith('psi_final_') and \
                   not k.startswith('driver_signal_') and \
                   k not in common_analysis_obs and \
                   k != 'RP_Matrix' and \
                   k != 'driver_full_state':
                      keys_to_del_obs.append(k)

        logger.debug(f"Cleanup check for {run_label}: Will attempt to delete large arrays: {keys_to_del_obs}")
        for k in set(keys_to_del_obs):
            if k in result_data['observables']:
                del result_data['observables'][k]
                logger.debug(f"Deleted large observable '{k}' from result_dict for {run_label}.")

    return result_data

# --- Suite Execution Function ---
def run_suite(experiment_configs_list, results_filepath, results_format='hdf', hdf_key='simulation_data'):
    """
    Runs a list of experiments, collects results, and saves them to a DataFrame.
    Returns the DataFrame and a dictionary containing the full results.
    """
    if not RESULTS_HANDLER_AVAILABLE:
        logger.critical("Cannot run suite: results_handler module is not available.")
        return None, {}

    all_results_list = []
    num_experiments = len(experiment_configs_list)
    logger.info(f"\n--- Starting Simulation Suite Execution: {num_experiments} Experiments ---")
    logger.info(f"Results for this batch will be saved to: {results_filepath} (Format: {results_format})")
    full_results_dict_temp = {}

    for i, (label, config_obj) in enumerate(experiment_configs_list):
        logger.info(f"\n>>> Running Experiment {i+1}/{num_experiments}: {label} <<<")
        result_dict = run_single_experiment(config_obj, run_label=label)
        full_results_dict_temp[label] = copy.deepcopy(result_dict) # Store full result temporarily

        # Prepare row for DataFrame, including classical results if available
        df_row_dict = results_handler.prepare_result_for_dataframe(result_dict)
        if result_dict.get("classical_results") and isinstance(result_dict["classical_results"], dict):
            classical_metrics = result_dict["classical_results"].get("metrics", {})
            classical_config = result_dict["classical_results"].get("config", {}) # Get config dict

            # Flatten metrics
            df_row_dict.update(results_handler.flatten_metrics(classical_metrics, prefix='classical_metric_'))

            # Flatten config only if classical_config is valid
            class_cfg_sn = None
            if classical_config and isinstance(classical_config, dict):
                try:
                    class_cfg_sn = SimpleNamespace(**classical_config)
                except TypeError as e_cfg:
                    logger.error(f"Could not create SimpleNamespace from classical config for run {label}: {e_cfg}. Config was: {classical_config}")
            elif classical_config:
                 logger.warning(f"Classical config for run {label} is not a dictionary: {type(classical_config)}. Cannot flatten.")

            if class_cfg_sn:
                df_row_dict.update(results_handler.flatten_config(class_cfg_sn, prefix='classical_config_'))
            else:
                 df_row_dict['classical_config_flatten_error'] = True # Indicate issue in DF
                 logger.warning(f"Classical config columns will be missing for run {label} due to issues with config data.")

            # Add error message if present
            if result_dict["classical_results"].get("error"):
                df_row_dict['classical_error'] = result_dict["classical_results"]["error"]

        all_results_list.append(df_row_dict)
        del result_dict # Hint GC to release memory from the deepcopy

    logger.info("\n--- Suite Execution Finished. Compiling Results DataFrame... ---")
    if not all_results_list:
        logger.warning("No results generated from the suite execution.")
        return None, {}

    try:
        results_df = pd.DataFrame(all_results_list)
        # Convert object columns that should be numeric (handling potential errors)
        for col in results_df.columns:
             if col.startswith(('metric_', 'classical_metric_', 'config_', 'classical_config_', 'observable_')):
                  # Check if column exists and is object type
                  if col in results_df.columns and results_df[col].dtype == 'object':
                      results_df[col] = pd.to_numeric(results_df[col], errors='coerce') # Coerce errors to NaN
             elif col == 'cluster_label':
                 if col in results_df.columns:
                     results_df[col] = pd.Categorical(results_df[col])
        logger.info("DataFrame created successfully.")
        logger.info(f"DataFrame Shape: {results_df.shape}")
        logger.debug("DataFrame dtypes:\n" + str(results_df.dtypes))
    except Exception as e:
        logger.error(f"Failed to create Pandas DataFrame from results list: {e}", exc_info=True)
        return None, {}

    # Save the DataFrame (handler manages format)
    if results_handler.save_results_df(results_df, results_filepath, format=results_format, key=hdf_key):
        return results_df, full_results_dict_temp # Return DF and the cached full results
    else:
        logger.error("Failed to save the results DataFrame.")
        # Still return the DF and cache even if saving failed
        return results_df, full_results_dict_temp


# --- Plot Manifest Generation Function ---
def generate_plot_manifest(all_plot_files, results_df, manifest_filepath):
    """Generates a text file listing all plots and their context."""
    logger.info(f"Generating plot manifest file: {manifest_filepath}")
    try:
        os.makedirs(os.path.dirname(manifest_filepath), exist_ok=True)
        with open(manifest_filepath, 'w', encoding='utf-8') as f:
            f.write("# Plot Manifest\n\n")
            f.write("This file lists the plots generated during the simulation suite run.\n\n")
            f.write("-" * 40 + "\n\n")
            # Add DataFrame summary first
            f.write("## Results DataFrame Summary\n")
            f.write(f"Path: {results_df.attrs.get('filepath', 'N/A')}\n")
            f.write(f"Shape: {results_df.shape}\n")
            f.write("Columns:\n")
            for col in results_df.columns:
                f.write(f"- {col}\n")
            f.write("\n" + "-" * 40 + "\n\n")
            f.write("## Generated Plots\n\n")
            all_plot_files = sorted(list(set(filter(None, all_plot_files))))

            for filename in all_plot_files:
                f.write(f"**Filename:** `{filename}`\n")
                run_label, plot_type, params_str = "N/A", "Unknown", ""
                # Infer context from filename structure (best effort)
                if filename.startswith("meta_"):
                    plot_type = "Meta-Analysis " + filename.split('_', 1)[1].split('.')[0].replace('_', ' ').title()
                    run_label = "All Runs"
                elif filename.startswith("repeatability_"):
                    plot_type = "Repeatability Analysis"
                    run_label = filename.split('_')[1] # Get prefix
                elif filename.startswith("sensitivity_jacobian"):
                    plot_type = "Sensitivity Jacobian Heatmap"
                    run_label = "Sensitivity Runs"
                elif filename.startswith("comparison_"):
                    plot_type = "Comparison " + filename.split('_', 1)[1].split('.')[0].replace('_', ' ').title()
                    run_label = "Applicable Runs"
                elif filename.startswith("decoding_"):
                    plot_type = "Embedding Decoding"
                    run_label = filename.split('decoding_')[1].split('.')[0] if 'decoding_' in filename else "Embedding Run"
                elif filename.startswith("gating_transient_"):
                    plot_type = "Gating Transient Analysis"
                    # Extract parameter and observable if possible
                    parts = filename.split('.')[0].split('_')
                    if len(parts) >= 4:
                        plot_type += f" ({parts[2]} - {parts[3]})"
                    run_label = "Gating Runs"
                elif filename.startswith("composite_"):
                    plot_type = "Composite " + filename.split('_', 1)[1].split('.')[0].replace('_', ' ').title()
                    run_label = "Multiple Runs (see plot)"
                else: # Assume per-run plot: Format RunLabel_PlotType.png/mp4
                    parts = filename.split('_')
                    potential_label_parts = []
                    plot_type_parts = []
                    plot_type_found = False
                    # Try to reconstruct label by removing known plot type keywords from the end
                    known_plot_types = ["probability", "heatmap", "observables", "animation", "lorenz", "rossler", "trajectory", "fft", "rqa", "cwt", "final", "state", "comparison", "cd", "debug"]
                    reconstructed_label = filename.rsplit('.', 1)[0] # Remove extension
                    # Iteratively remove known plot types from the end
                    found_type_in_iter = True
                    while found_type_in_iter:
                        found_type_in_iter = False
                        for p_type in known_plot_types:
                            suffix = "_" + p_type
                            if reconstructed_label.endswith(suffix):
                                plot_type_parts.insert(0, p_type) # Add to beginning of type parts
                                reconstructed_label = reconstructed_label[:-len(suffix)]
                                plot_type_found = True
                                found_type_in_iter = True
                                break # Restart check with shortened label

                    run_label = reconstructed_label if plot_type_found else "N/A" # What's left is the label
                    plot_type = " ".join(plot_type_parts).replace('_', ' ').title() if plot_type_found else "Unknown Per-Run Plot"


                f.write(f"- **Type:** {plot_type}\n")
                f.write(f"- **Associated Run(s):** {run_label}\n")

                # Add key parameters if specific run label is identified
                if run_label not in ["N/A", "All Runs", "Multiple Runs (see plot)", "Gating Runs", "Sensitivity Runs", "Applicable Runs"]:
                    run_row = results_df[results_df['run_label'] == run_label]
                    if not run_row.empty:
                         row = run_row.iloc[0]
                         driver = row.get('config_driver_type', 'N/A')
                         alpha = row.get('config_alpha', 'N/A')
                         epsilon = row.get('config_epsilon', 'N/A')
                         params_str = f" (Driver: {driver}, Alpha: {reporting._format_metric(alpha)}, Epsilon: {reporting._format_metric(epsilon)})" # Use reporting helper
                         f.write(f"- **Key Params:**{params_str}\n")
                f.write("\n")

        logger.info(f"Plot manifest saved successfully to {manifest_filepath}")
    except Exception as e:
        logger.error(f"Failed to generate plot manifest: {e}", exc_info=True)


# --- Main Execution Block ---
if __name__ == "__main__":
    main_start_time = time.time()
    # --- Initial Info Logging ---
    logger.info("=============================================")
    logger.info(f" Starting Quantum Chaos Simulation Suite Run (PID: {os.getpid()})")
    logger.info(f"Results Dir: {base_config.results_dir}")
    logger.info(f"Plot Dir: {base_config.plot_dir}")
    logger.info(f"Log Dir: {base_config.log_dir}")
    logger.info(f"Report Dir: {base_config.report_dir}")
    logger.info(f"Base Config Highlights: N={base_config.N}, T={base_config.T}, M={base_config.M}, Driver={base_config.driver_type}")
    logger.info(f"Classical Sim Enabled: {base_config.enable_classical_simulation}")
    logger.info(f"Scheduling Enabled (Base): {base_config.enable_parameter_scheduling}")
    logger.info(f"LLM Reporting Enabled: {base_config.enable_openai_reporting} (Choice: {base_config.llm_choice})")
    logger.info(f"Orchestration Enabled: {base_config.enable_orchestration}")
    logger.info(f"Plot Control: PerRun={base_config.save_plots_per_run}, Meta={base_config.save_plots_meta}, Comparison={base_config.save_plots_comparison}, Manifest={base_config.generate_plot_manifest}")
    logger.info(f"Dependency Status: CoreOK={CORE_MODULES_OK}, ClassicalOK={CLASSICAL_SIM_AVAILABLE}, ResultsHandlerOK={RESULTS_HANDLER_AVAILABLE}, ExpManagerOK={EXPERIMENT_MANAGER_AVAILABLE}, MetaAnalysisOK={META_ANALYSIS_AVAILABLE}, InfoFlowOK={INFO_FLOW_AVAILABLE}, LLMOK={LLM_INTERFACE_AVAILABLE}, ReportingOK={REPORTING_AVAILABLE}, AgentsOK={AGENTS_AVAILABLE}, OrchestratorOK={ORCHESTRATOR_AVAILABLE}")
    logger.info("=============================================")

    # --- Exit early if critical modules missing ---
    if not CORE_MODULES_OK or not RESULTS_HANDLER_AVAILABLE or not EXPERIMENT_MANAGER_AVAILABLE:
         logger.critical("Cannot proceed without core simulation modules, results_handler, or experiment_manager. Exiting.")
         logging.shutdown()
         exit(1)

    all_generated_plot_files = [] # Initialize list to collect ALL plot filenames
    csv_filepath = "" # Initialize csv path

    # --- Check if running Orchestration or Standard Suite ---
    if getattr(base_config, 'enable_orchestration', False):
        logger.info(">>> Running in AI Orchestration Mode <<<")
        if ORCHESTRATOR_AVAILABLE:
             try:
                 orch = orchestrator.Orchestrator(base_config)
                 orch.run_main_loop()
                 # Orchestrator handles its own saving of DF and reports.
                 # Consider how to collect plot filenames during orchestration if needed.
             except Exception as e_orch:
                 logger.critical(f"Orchestrator failed to run: {e_orch}", exc_info=True)
                 logging.shutdown()
                 exit(1)
        else:
            logger.critical("Orchestration enabled but Orchestrator module unavailable. Exiting.")
            logging.shutdown()
            exit(1)

    else:
        # --- Standard Suite Execution ---
        logger.info(">>> Running in Standard Suite Mode <<<")
        experiments_to_run = []
        # Define experiments using experiment_manager
        experiments_to_run.extend(experiment_manager.define_standard_experiments(base_config))
        if base_config.enable_parametric_sweep:
            experiments_to_run.extend(experiment_manager.generate_coupling_sweep(
                base_config, param_name=base_config.sweep_parameter, num_values=base_config.sweep_values_count
            ))
        experiments_to_run.extend(experiment_manager.generate_driver_family_configs(
            base_config, driver_list=['zero', 'sine', 'quasi_periodic', 'lorenz', 'rossler', 'logistic_map', 'filtered_noise']
        ))
        if base_config.enable_repeatability_test:
            experiments_to_run.extend(experiment_manager.generate_repeatability_configs(
                base_config, run_label_prefix="Base_Repeat", num_repeats=base_config.repeatability_num_runs,
                enable_noise=base_config.repeatability_enable_noise
            ))
        if hasattr(base_config, 'sensitivity_params_to_perturb') and base_config.sensitivity_params_to_perturb:
            experiments_to_run.extend(experiment_manager.generate_sensitivity_configs(
                base_config, params_to_perturb=base_config.sensitivity_params_to_perturb,
                perturbation_scale=base_config.sensitivity_default_perturbation_scale
            ))
        # Add Gating/Embedding if enabled and functions exist
        if getattr(base_config, 'enable_gating_experiments', False) and hasattr(experiment_manager, 'generate_gating_configs'):
            experiments_to_run.extend(experiment_manager.generate_gating_configs(base_config))
        if getattr(base_config, 'enable_embedding_experiments', False) and hasattr(experiment_manager, 'generate_embedding_configs'):
            message_to_embed = [0, 1, 1, 0, 1] # Example message
            experiments_to_run.extend(experiment_manager.generate_embedding_configs(base_config, message_to_embed))

        # Ensure unique labels
        labels_seen = set()
        unique_experiments = []
        for label, config in experiments_to_run:
            original_label = label
            count = 1
            while label in labels_seen:
                label = f"{original_label}_{count}"
                count += 1
            if label != original_label:
                logger.warning(f"Duplicate experiment label '{original_label}'. Renaming to '{label}'.")
            unique_experiments.append((label, config))
            labels_seen.add(label)
        experiments_to_run = unique_experiments
        logger.info(f"Total unique experiments defined for standard suite: {len(experiments_to_run)}")

        # --- Run Quantum Simulation Suite ---
        quantum_results_df, full_results_cache = None, {}
        if experiments_to_run:
            quantum_results_df, full_results_cache = run_suite(
                experiments_to_run,
                results_filepath=base_config.results_dataframe_path,
                results_format=base_config.results_storage_format,
                hdf_key=base_config.hdf_key
            )
            if quantum_results_df is not None:
                # Add attributes to DataFrame for later use
                quantum_results_df.attrs['filepath'] = base_config.results_dataframe_path
                quantum_results_df.attrs['plot_dir'] = base_config.plot_dir
                quantum_results_df.attrs['results_dir'] = base_config.results_dir
                # Collect per-run plot filenames from the cache
                for label, result_data in full_results_cache.items():
                     if result_data and 'observables' in result_data and 'plot_files_run' in result_data['observables']:
                          all_generated_plot_files.extend(result_data['observables']['plot_files_run'])
        else:
            logger.warning("No quantum experiments defined to run.")

        # --- Meta Analysis & Reporting (Standard Suite Mode) ---
        comparison_metrics = {'plot_files': []} # Initialize comparison metrics dict
        repeatability_summary, sensitivity_jacobian, classical_overall_results = None, None, None
        meta_plot_files, comparison_plot_files = [], []

        if quantum_results_df is not None and not quantum_results_df.empty:
            logger.info("\n--- Starting Meta-Analysis and Reporting (Standard Suite Run) ---")
            # Define features for analysis
            features_for_analysis = [
                f for f in ['metric_LLE','metric_DET','metric_LAM','metric_ENTR','metric_CorrDim',
                            'metric_TransferEntropy','observable_final_entropy',
                            'classical_metric_LLE_FullDim','classical_metric_LLE_3D',
                            'classical_metric_CorrDim_FullDim','classical_metric_CorrDim_3D']
                if f in quantum_results_df.columns and quantum_results_df[f].notna().any()
            ]
            logger.info(f"Using valid features for meta-analysis: {features_for_analysis}")

            # Perform Meta-Analysis if module available and features exist
            if META_ANALYSIS_AVAILABLE and features_for_analysis:
                # Clustering
                if meta_analysis.SKLEARN_AVAILABLE:
                    logger.info("Running Clustering...")
                    quantum_results_df = meta_analysis.cluster_results(quantum_results_df, features=features_for_analysis)
                else:
                    logger.warning("Clustering skipped: scikit-learn not available.")
                # Dimensionality Reduction
                if meta_analysis.UMAP_AVAILABLE:
                    logger.info("Running Dimensionality Reduction (UMAP)...")
                    quantum_results_df = meta_analysis.perform_dimensionality_reduction(quantum_results_df, features=features_for_analysis)
                else:
                    logger.warning("Dimensionality reduction skipped: umap-learn not available.")

                # Meta-analysis Plotting (checks internal flags)
                logger.info("Running meta-analysis plotting functions...")
                plot_dir_meta = meta_analysis.get_plot_dir(quantum_results_df, base_config)
                cluster_col = 'cluster_label' if 'cluster_label' in quantum_results_df else None
                color_metric_map = next((f for f in ['metric_LLE', 'metric_DET', 'observable_final_entropy'] if f in features_for_analysis), features_for_analysis[0] if features_for_analysis else None)
                param_x = next((p for p in ['config_alpha', 'config_epsilon', 'config_lorenz_initial_state_0'] if p in quantum_results_df.columns and quantum_results_df[p].nunique() > 1), None)
                param_y = next((p for p in ['config_epsilon', 'config_lorenz_initial_state_1', 'config_lorenz_rho'] if p != param_x and p in quantum_results_df.columns and quantum_results_df[p].nunique() > 1), None)

                if param_x and param_y and color_metric_map:
                    fname = meta_analysis.plot_parameter_map(quantum_results_df, param_x, param_y, color_metric_map, cluster_col, filename=os.path.join(plot_dir_meta,"meta_param_map.png"), config_obj=base_config)
                    if fname: meta_plot_files.append(os.path.basename(fname))
                if 'embedding_X' in quantum_results_df and color_metric_map:
                    fname = meta_analysis.plot_embedding(quantum_results_df, 'embedding_X', 'embedding_Y', color_metric_map, cluster_col, filename=os.path.join(plot_dir_meta,"meta_embedding_plot.png"), config_obj=base_config)
                    if fname: meta_plot_files.append(os.path.basename(fname))

                # Run Repeatability Analysis (passing full_results_cache)
                if base_config.enable_repeatability_test and hasattr(meta_analysis, 'analyze_repeatability'):
                    logger.info("Running Repeatability Analysis...")
                    repeatability_summary = meta_analysis.analyze_repeatability(
                        quantum_results_df, "Base_Repeat", features_for_analysis, full_results_cache, base_config
                    )
                    if repeatability_summary and 'metrics_plot' in repeatability_summary and repeatability_summary['metrics_plot']:
                         meta_plot_files.append(repeatability_summary['metrics_plot'])

                # Run Sensitivity Analysis
                if hasattr(base_config, 'sensitivity_params_to_perturb') and base_config.sensitivity_params_to_perturb and hasattr(meta_analysis, 'calculate_sensitivity_jacobian'):
                    logger.info("Running Sensitivity (Jacobian) Analysis...")
                    sensitivity_jacobian = meta_analysis.calculate_sensitivity_jacobian(
                        quantum_results_df, "Sensitivity_Base", list(base_config.sensitivity_params_to_perturb.keys()), base_config
                    )
                    if sensitivity_jacobian is not None and not sensitivity_jacobian.empty:
                        meta_plot_files.append("sensitivity_jacobian_heatmap.png")

                # Run Quantum vs Classical Comparison
                if getattr(base_config, 'enable_classical_simulation', False) and hasattr(meta_analysis, 'compare_quantum_classical'):
                    logger.info("Running Quantum vs Classical Comparison...")
                    fname = meta_analysis.compare_quantum_classical(quantum_results_df, base_config)
                    if fname: meta_plot_files.append(os.path.basename(fname))

            # --- Calculate Specific L2 Differences ---
            logger.info("Calculating specific L2 state differences...")
            key_activation = "1_BaseDriver_Activation"
            key_control = "2_Ablation_Control"
            key_sens_driver = "3_Sensitivity_DriverIC"
            key_sens_qic = "4_Sensitivity_QuantumIC"

            # Use the helper function defined earlier in main.py
            comparison_metrics['diff_activation_vs_control_L2'] = calculate_l2_state_difference(
                key_activation, key_control, full_results_cache, base_config
            )
            comparison_metrics['diff_activation_vs_driverIC_sensitivity_L2'] = calculate_l2_state_difference(
                key_activation, key_sens_driver, full_results_cache, base_config
            )
            comparison_metrics['diff_activation_vs_qic_sensitivity_L2'] = calculate_l2_state_difference(
                key_activation, key_sens_qic, full_results_cache, base_config
            )
            # --- End L2 Calculation ---

            # --- Generate Specific Comparison Plots ---
            logger.info("Generating specific comparison/composite plots...")
            plot_dir_comp = get_plot_dir(quantum_results_df, base_config)
            # Compare final states of standard runs
            comp1_labels_exist = [l for l in ["1_BaseDriver_Activation", "2_Ablation_Control", "3_Sensitivity_DriverIC", "4_Sensitivity_QuantumIC"] if l in full_results_cache]
            if len(comp1_labels_exist) >= 2 and hasattr(plot_wavefunction, 'plot_composite_final_states'):
                fname = plot_wavefunction.plot_composite_final_states(full_results_cache, comp1_labels_exist, filename=os.path.join(plot_dir_comp,"composite_sensitivity_final_states.png"), config_obj=base_config)
                if fname: comparison_plot_files.append(os.path.basename(fname))
            # Compare FFT of different drivers
            comp2_labels_exist = [l for l in ["Driver_Lorenz", "Driver_Rossler", "Driver_Sine", "Driver_Zero"] if l in full_results_cache]
            if len(comp2_labels_exist) >= 2 and hasattr(plot_wavefunction, 'plot_composite_fft'):
                observable_fft = base_config.observable_for_fft
                fname = plot_wavefunction.plot_composite_fft(full_results_cache, comp2_labels_exist, observable_name=observable_fft, filename=os.path.join(plot_dir_comp,f"composite_driver_fft_{plot_wavefunction._sanitize_filename(observable_fft)}.png"), config_obj=base_config)
                if fname: comparison_plot_files.append(os.path.basename(fname))
            # Compare RQA of different drivers
            if len(comp2_labels_exist) >= 2 and hasattr(plot_wavefunction, 'plot_composite_rqa'):
                fname = plot_wavefunction.plot_composite_rqa(full_results_cache, comp2_labels_exist, filename=os.path.join(plot_dir_comp,"composite_driver_rqa.png"), config_obj=base_config)
                if fname: comparison_plot_files.append(os.path.basename(fname))

            # Gating/Embedding Analysis Plots
            if getattr(base_config, 'enable_gating_experiments', False) and hasattr(meta_analysis, 'analyze_gating_transients') and meta_analysis.ANALYSIS_MODULES_AVAILABLE:
                logger.info("Running Gating Transient Analysis...")
                gating_fnames = meta_analysis.analyze_gating_transients(quantum_results_df, base_config)
                if gating_fnames: comparison_plot_files.extend([os.path.basename(f) for f in gating_fnames if f])
            if getattr(base_config, 'enable_embedding_experiments', False) and hasattr(meta_analysis, 'decode_embedded_signal') and meta_analysis.ANALYSIS_MODULES_AVAILABLE:
                logger.info("Running Signal Embedding Decoding Analysis...")
                decoded_results, metrics, embedding_fnames = meta_analysis.decode_embedded_signal(quantum_results_df, base_config)
                if embedding_fnames: comparison_plot_files.extend([os.path.basename(f) for f in embedding_fnames if f])

            # --- Consolidate Plot Files ---
            all_generated_plot_files.extend(meta_plot_files)
            all_generated_plot_files.extend(comparison_plot_files)
            all_generated_plot_files = sorted(list(set(filter(None, all_generated_plot_files))))
            comparison_metrics['plot_files'] = all_generated_plot_files # Update plot files in metrics dict

            # --- Generate Plot Manifest ---
            if base_config.generate_plot_manifest:
                 manifest_path = os.path.join(base_config.report_dir, "plot_manifest.md")
                 generate_plot_manifest(all_generated_plot_files, quantum_results_df, manifest_path)

            # --- Final Reporting Setup ---
            # Check for classical results column and non-NA values
            classical_config_col = 'classical_config_model_type'
            classical_overall_results = None
            if classical_config_col in quantum_results_df.columns:
                 classical_run_rows = quantum_results_df[quantum_results_df[classical_config_col].notna()]
                 if not classical_run_rows.empty:
                     logger.info("Found successful classical runs in DataFrame. Extracting representative results.")
                     first_classical_run = classical_run_rows.iloc[0]
                     classical_overall_results = {
                          "metrics": {k.replace('classical_metric_', ''): v for k, v in first_classical_run.items() if k.startswith('classical_metric_') and pd.notna(v)},
                          "config": {k.replace('classical_config_', ''): v for k, v in first_classical_run.items() if k.startswith('classical_config_') and pd.notna(v)}
                     }
                 else:
                      logger.info(f"Column '{classical_config_col}' found, but no non-NA values. Assuming classical runs failed or metrics are missing.")
            else:
                 logger.warning(f"Column '{classical_config_col}' not found in results DataFrame. Assuming no successful classical runs for reporting.")

            # --- Call AI Reporting ---
            if base_config.enable_openai_reporting and REPORTING_AVAILABLE:
                logger.info("Generating AI Report Summary from DataFrame...")
                try:
                    # Example hypothesis abstract (customize as needed)
                    hypothesis_abstract = """
This study investigates complexity in a 1D quantum system subjected to various external potential modulations (chaotic, periodic, etc.) and potentially compares this to complexity arising from a higher-dimensional deterministic classical system.
Quantum Part: Tests if complexity originates from the external driver vs. internal QIC sensitivity. Key metrics: LLE, RQA, CorrDim, Transfer Entropy (TE), FFT/CWT, L2 state differences. Experiments: Base Driver Activation, Ablation Control, Sensitivities (DriverIC, QIC), Parametric Sweep, Different Drivers, Repeatability, Sensitivity Jacobian, Gating/Embedding.
Classical Part (if run): Tests if 3D chaotic-like behavior can emerge from a deterministic 4D system (e.g., coupled oscillators, Rössler hyperchaos). Key metrics: LLE, CorrDim for full system and projected 3D subspace. Optionally coupled to quantum driver.
Overall Goal: To understand how external drivers and/or higher-dimensional determinism can induce complexity in lower-dimensional quantum or classical systems, assess reliability/sensitivity, and explore control pathways.
"""
                    prompt = reporting.generate_report_prompt(
                        validation_results_df=quantum_results_df,
                        base_config=base_config,
                        comparison_metrics=comparison_metrics, # Pass dict with L2 diffs
                        hypothesis_abstract=hypothesis_abstract,
                        classical_results=classical_overall_results,
                        repeatability_summary=repeatability_summary,
                        sensitivity_jacobian=sensitivity_jacobian
                    )
                    logger.debug(f"Generated Prompt for {base_config.llm_choice} (Length: {len(prompt)} chars).")

                    if LLM_INTERFACE_AVAILABLE:
                        report_text = llm_interface.get_llm_response(prompt, provider=base_config.llm_choice, config_obj=base_config)
                    else:
                        logger.error("LLM Interface module not available for report generation.")
                        report_text = "Error: LLM Interface unavailable."

                    report_filename = os.path.join(base_config.report_dir, "ai_validation_report_standard_suite.md")
                    with open(report_filename, "w", encoding='utf-8') as f:
                        f.write(report_text if report_text else "Report generation failed.")
                    logger.info(f"AI-generated report saved to {report_filename}")

                except Exception as e:
                    logger.error(f"Failed to generate or save AI report: {e}", exc_info=True)

            elif base_config.enable_openai_reporting:
                logger.error("AI reporting enabled but required modules unavailable.")

            # --- Save DataFrame as CSV ---
            try:
                csv_filename = os.path.splitext(base_config.results_dataframe_filename)[0] + ".csv"
                csv_filepath = os.path.join(base_config.results_dir, csv_filename)
                logger.info(f"Saving final results DataFrame to CSV: {csv_filepath}")
                # Convert SimpleNamespace columns before saving to CSV
                df_to_save = quantum_results_df.copy()
                for col in df_to_save.columns:
                     first_val = df_to_save[col].dropna().iloc[0] if df_to_save[col].notna().any() else None
                     if isinstance(first_val, SimpleNamespace):
                          logger.debug(f"Converting SimpleNamespace column '{col}' to string for CSV.")
                          df_to_save[col] = df_to_save[col].apply(lambda x: str(vars(x)) if isinstance(x, SimpleNamespace) else x)
                df_to_save.to_csv(csv_filepath, index=False, encoding='utf-8')
                logger.info("DataFrame saved successfully as CSV.")
            except Exception as e_csv:
                logger.error(f"Failed to save DataFrame to CSV: {e_csv}", exc_info=True)

        else:
            logger.warning("Quantum results DataFrame is empty or None. Skipping meta-analysis, reporting, and CSV saving.")

    # --- Final Summary ---
    main_end_time = time.time()
    logger.info("=============================================")
    mode = "Orchestration" if getattr(base_config, 'enable_orchestration', False) else "Standard Suite"
    logger.info(f" Quantum Chaos Simulation ({mode}) Finished ({main_end_time - main_start_time:.2f} seconds)")
    if getattr(base_config, 'enable_orchestration', False):
         logger.info(f"Orchestrator state saved to: {getattr(base_config, 'orchestrator_state_file', 'N/A')}")
         logger.info(f"Final combined results DataFrame saved to: {getattr(base_config, 'results_dataframe_path', 'N/A')}")
         logger.info(f"Final orchestrator report saved in: {getattr(base_config, 'report_dir', 'N/A')}")
    elif quantum_results_df is not None:
        logger.info(f"Results DataFrame ({base_config.results_storage_format}) saved to: {base_config.results_dataframe_path}")
        if os.path.exists(csv_filepath):
            logger.info(f"Results CSV saved to: {csv_filepath}")
    else:
        logger.warning("Results DataFrame was not generated or saved.")
    logger.info(f"Results directory (plots, logs, reports): {base_config.results_dir}")
    if base_config.enable_classical_simulation:
        logger.info(f"Classical results directory: {base_config.classical_results_dir}")
    if not getattr(base_config, 'enable_orchestration', False) and base_config.generate_plot_manifest:
        manifest_path = os.path.join(base_config.report_dir, "plot_manifest.md")
        if os.path.exists(manifest_path):
            logger.info(f"Plot manifest generated at: {manifest_path}")
    logger.info(f"Log file: {log_filepath}")
    logger.info("=============================================")
    logging.shutdown()