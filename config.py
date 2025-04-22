# quantum_chaos_sim/config.py
"""
Configuration parameters for the simulation, stored in a SimpleNamespace object.
Includes Phase 4 Orchestration and LLM settings.
"""
import numpy as np
from types import SimpleNamespace
import os # For environment variable loading
from dotenv import load_dotenv # Import the function

# --- Load Environment Variables ---
# Get the directory containing this config.py file
package_root_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file expected to be in the same directory
dotenv_path = os.path.join(package_root_dir, '.env')
# Load the .env file if it exists
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    # Optional: Log that the .env file was loaded
    # print(f"Loaded environment variables from: {dotenv_path}") # Avoid print in library code
else:
    # Optional: Warn if the .env file is expected but not found
    # print(f"Warning: .env file not found at {dotenv_path}") # Avoid print in library code
    pass


# Create a SimpleNamespace object to hold the config values
cfg = SimpleNamespace()

# --- Base Directory ---
# package_root_dir already defined above

# --- Simulation Grid Parameters ---
cfg.L = 20.0       # Spatial extent (-L to L)
cfg.N = 512        # Number of spatial grid points (must be power of 2 for FFT efficiency)
cfg.T = 50.0       # Total simulation time
cfg.M = 2000       # Number of time steps

# --- Physical Constants (using dimensionless units where hbar=1, m=1) ---
cfg.hbar = 1.0
cfg.m = 1.0

# --- Initial Wavefunction Parameters (Gaussian wavepacket) ---
cfg.x0 = -5.0      # Initial position center
cfg.sigma_psi = 1.0 # Initial spatial width
cfg.k0_psi = 5.0   # Initial momentum (average)

# --- Potential Parameters ---
cfg.V0_static = 0.0         # Base static potential (set to 0 for free particle + chaos)
cfg.epsilon = 0.5           # Strength of general modulation (non-coupling part)
cfg.k_potential = 0.5       # Spatial frequency in potential modulation
cfg.omega_potential = 0.2   # Temporal frequency in potential modulation (intrinsic)
cfg.alpha = 2.0             # Coupling strength to the *external driver signal* z(t)

# --- External Driver Signal Selection ---
# Options: 'lorenz', 'sine', 'quasi_periodic', 'zero', 'rossler', 'logistic_map', 'filtered_noise'
cfg.driver_type = 'lorenz'

# --- Lorenz Driver Parameters (if driver_type='lorenz') ---
cfg.lorenz_sigma = 10.0
cfg.lorenz_rho = 28.0
cfg.lorenz_beta = 8.0 / 3.0
cfg.lorenz_initial_state = np.array([1.0, 1.0, 1.0])
cfg.lorenz_ic_bounds = { # Bounds for sweeps/sampling (Phase 1)
    'x': (0.0, 2.0),   # Min/Max for initial x
    'y': (0.0, 2.0),   # Min/Max for initial y
    'z': (20.0, 35.0)  # Min/Max for initial z
}
cfg.lorenz_use_component = 'z' # Which component (x, y, or z) to use as the signal

# --- Rossler Driver Parameters (if driver_type='rossler') ---
cfg.rossler_a = 0.2
cfg.rossler_b = 0.2
cfg.rossler_c = 5.7
cfg.rossler_initial_state = np.array([0.1, 0.1, 0.1]) # x, y, z
cfg.rossler_use_component = 'x' # Which component (x, y, or z) to use

# --- Logistic Map Driver Parameters (if driver_type='logistic_map') ---
cfg.logistic_r = 4.0 # Parameter for the map x_{n+1} = r * x_n * (1 - x_n). r=4.0 gives chaos.
cfg.logistic_initial_x = 0.618 # Initial value (must be between 0 and 1)
cfg.logistic_skip_transients = 1000 # Number of initial iterations to discard
cfg.logistic_scale = 1.0 # Factor to scale the output (0 to 1)
cfg.logistic_offset = 0.0 # Offset to add to the output

# --- Filtered Noise Driver Parameters (if driver_type='filtered_noise') ---
cfg.noise_type = 'gaussian' # 'gaussian' or 'uniform'
cfg.noise_seed = None       # Seed for reproducibility (None for random)
cfg.noise_scale = 1.0       # Scale of the raw noise before filtering
cfg.filter_type = 'lowpass' # 'lowpass', 'highpass', 'bandpass'
cfg.filter_order = 5        # Order of the Butterworth filter
cfg.filter_cutoff_low = 0.1  # Cutoff frequency (normalized 0 to 0.5*sampling_rate)
cfg.filter_cutoff_high = 0.4 # Upper cutoff for bandpass (normalized)

# --- Sine Driver Parameters (if driver_type='sine') ---
cfg.sine_amplitude = 1.0 # Amplitude of the z(t) = A*sin(freq*t) signal
cfg.sine_frequency = 0.5 # Frequency (angular) of the z(t) signal

# --- Quasi-Periodic Driver Parameters (if driver_type='quasi_periodic') ---
cfg.quasi_amplitude1 = 1.0
cfg.quasi_frequency1 = 0.5 # Needs to be incommensurate with freq2 for quasi-periodicity
cfg.quasi_amplitude2 = 0.7
cfg.quasi_frequency2 = 0.5 * np.sqrt(2.0) # Example incommensurate frequency

# --- Phase 3: Parameter Scheduling ---
cfg.enable_parameter_scheduling = False # Master flag to enable time-varying parameters
cfg.alpha_schedule = None      # Example: [(0.0, 1.0), (10.0, 0.0), (20.0, 1.0)] # List of (time, value) pairs
cfg.epsilon_schedule = None    # Example: [(0.0, 0.5), (cfg.T / 2, 0.1)]
# Future: cfg.driver_schedule = None # Example: [(0.0, 'lorenz'), (15.0, 'sine')] # More complex to implement

# --- Phase 3: Gating & Embedding Experiment Flags (used by experiment_manager) ---
cfg.enable_gating_experiments = True # Flag for experiment_manager to generate gating schedules
cfg.enable_embedding_experiments = True # Flag for experiment_manager to generate embedding schedules
# Specific parameters for generating gating/embedding schedules can be added here if needed
# cfg.gating_off_value_alpha = 0.0
# cfg.embedding_bit_high_alpha_mod = 0.1 # Example: modulation added for bit '1'
# cfg.embedding_bit_duration = 5.0 # Example: duration of each bit's influence

# --- Derived Quantum Parameters ---
cfg.x_grid = np.linspace(-cfg.L, cfg.L, cfg.N, endpoint=False) # Use endpoint=False for FFT periodicity
cfg.dx = cfg.x_grid[1] - cfg.x_grid[0]
cfg.t_grid = np.linspace(0, cfg.T, cfg.M)
cfg.dt_quantum = cfg.t_grid[1] - cfg.t_grid[0]
cfg.k_grid = 2 * np.pi * np.fft.fftfreq(cfg.N, d=cfg.dx) # Momentum space grid
cfg.sampling_rate = 1.0 / cfg.dt_quantum # Sampling rate for signal generation

# --- Control Run Parameters ---
# Define what constitutes a "control" run. Usually, alpha=0.
cfg.control_alpha = 0.0
cfg.control_epsilon = cfg.epsilon # Keep base modulation for comparison if desired

# --- Output Directories ---
cfg.results_dir = os.path.join(package_root_dir, "results") # e.g., .../quantum_chaos_sim/results
cfg.plot_dir = os.path.join(cfg.results_dir, "plots")
cfg.log_dir = os.path.join(cfg.results_dir, "logs")
cfg.report_dir = os.path.join(cfg.results_dir, "reports") # Put reports inside results dir
# NEW: Directory for orchestrator state
cfg.orchestrator_state_dir = os.path.join(cfg.results_dir, "orchestrator_state")

# --- Results Storage (Phase 1) ---
cfg.results_storage_format = 'hdf' # Options: 'hdf', 'csv'
cfg.results_dataframe_filename = "simulation_results.h5" # Or .csv
cfg.results_dataframe_path = os.path.join(cfg.results_dir, cfg.results_dataframe_filename)
cfg.hdf_key = 'simulation_data' # Key to use within the HDF5 file

# --- Visualization ---
cfg.plot_interval = 50 # Adjust frame skip for animation (remains)
cfg.animate = False    # Keep animation disabled by default (remains)
cfg.save_results = True # Master flag for saving *anything* (plots, reports, data)

# NEW Flags for finer plot control (only active if cfg.save_results is True)
cfg.save_plots_per_run = False   # Save individual run plots (heatmap, observables, FFT, RQA, CWT)? Set False to reduce plots drastically.
cfg.save_plots_meta = True      # Save meta-analysis plots (sweep maps, embedding, Jacobian, repeatability)? Usually desired.
cfg.save_plots_comparison = True # Save specific composite/comparison plots? Usually desired.
cfg.generate_plot_manifest = True # Generate the plot manifest file?
cfg.save_heatmap = True          # Specific control for heatmap (used if save_plots_per_run is True) - Keep this existing flag for now

# Debug plots should remain controllable individually
cfg.lle_debug_plot = False
cfg.corr_dim_debug_plot = False

# --- Observables Tracking ---
cfg.track_norm = True
cfg.track_position = True
cfg.track_momentum = True
cfg.track_energy = True
cfg.track_spatial_variance = True
cfg.track_shannon_entropy = True
cfg.save_observable_arrays = True # Optionally save full observable time series to .npy files
# List which observables to save if save_observable_arrays is True
cfg.observables_to_save = ["Position <x>", "Momentum <p>", "Shannon Entropy S(x)"]

# --- Validation Experiment Flags & Parameters ---
# Basic Analysis
cfg.enable_fft_analysis = True
cfg.observable_for_fft = "Position <x>" # Default observable for analysis

# LLE Analysis
cfg.enable_lle_calculation = True
cfg.observable_for_lle = "Position <x>"
cfg.lle_emb_dim = 5
cfg.lle_lag = 10  # Adjust based on autocorrelation/mutual info if possible
cfg.lle_fit_method = 'RANSAC' # Requires scikit-learn
cfg.lle_debug_plot = False

# --- Sensitivity Tests (Phase 1 & 2) ---
cfg.enable_sensitivity_test = True # Lorenz IC sensitivity (flag for define_standard_experiments)
cfg.sensitivity_lorenz_initial_state_perturbation = np.array([1e-5, 0, 0]) # Default perturbation for standard test
cfg.enable_quantum_IC_sensitivity_test = True # Quantum IC sensitivity (flag for define_standard_experiments)
cfg.quantum_IC_perturbation_scale = 1e-5      # Default perturbation scale for standard test
cfg.quantum_IC_perturbation_type = 'x0'       # Parameter to perturb in QIC (e.g., 'x0', 'k0_psi')
# Parameters for generate_sensitivity_configs (Phase 2)
cfg.sensitivity_params_to_perturb = {          # Dictionary defining parameters and their specific perturbation scales
    'alpha': 1e-4,
    'epsilon': 1e-4,
    'lorenz_initial_state': np.array([1e-5, 1e-5, 1e-5]), # Example: perturb all lorenz IC components
    'x0': 1e-5,
    'k0_psi': 1e-5
}
cfg.sensitivity_default_perturbation_scale = 1e-5 # Fallback scale

# --- Repeatability Testing (Phase 2) ---
cfg.enable_repeatability_test = True
cfg.repeatability_num_runs = 2
cfg.repeatability_enable_noise = False # Set to True to test robustness with noise injection

# --- Noise Injection Parameters (Phase 2 - if repeatability_enable_noise is True) ---
cfg.inject_runtime_noise = False       # Master flag, usually set by experiment_manager
cfg.noise_level_potential = 1e-4       # Std deviation of Gaussian noise added to V(x,t) at each step
cfg.noise_level_qic_x0 = 0             # Optional: Add noise to Quantum ICs between repeatability runs
cfg.noise_level_qic_k0 = 0

# --- Parametric Sweep (Phase 1) ---
cfg.enable_parametric_sweep = True
cfg.sweep_parameter = "alpha" # Parameter to sweep ('alpha' or 'epsilon')
cfg.sweep_values_count = 5    # Number of values in the sweep (including 0 and base value)

# --- RQA Analysis ---
cfg.enable_rqa_analysis = True
cfg.observable_for_rqa = "Position <x>"
cfg.rqa_embedding_dimension = cfg.lle_emb_dim # Use same embedding as LLE for consistency
cfg.rqa_time_delay = cfg.lle_lag             # Use same lag as LLE
cfg.rqa_similarity_measure = 'euclidean'     # Options: 'euclidean', 'maximum', 'manhattan'
cfg.rqa_neighbourhood_type = 'adaptive'      # 'fixed' or 'adaptive'
cfg.rqa_neighbourhood_threshold = 0.1        # Radius for 'fixed', std dev fraction for 'adaptive'
cfg.rqa_theiler_corrector = 1                # Exclude main diagonal (common practice)
cfg.rqa_min_diag_len = 2
cfg.rqa_min_vert_len = 2
cfg.rqa_min_white_vert_len = 2
cfg.rqa_normalize = False                    # Normalize time series before RQA? (Usually False if using adaptive threshold)

# --- CWT Analysis ---
cfg.enable_cwt_analysis = True
cfg.observable_for_cwt = "Position <x>"
cfg.cwt_wavelet_type = 'cmor1.5-1.0' # Complex Morlet wavelet (adjust B, C as needed)
cfg.cwt_scales_num = 64
cfg.cwt_scales_min = 1
cfg.cwt_scales_max = 128 # Adjust max scale based on expected frequencies and time series length
cfg.cwt_scales = np.geomspace(cfg.cwt_scales_min, cfg.cwt_scales_max, cfg.cwt_scales_num)

# --- Correlation Dimension Analysis ---
cfg.enable_correlation_dimension = True
cfg.observable_for_corr_dim = "Position <x>"
cfg.corr_dim_embedding_dim = cfg.rqa_embedding_dimension # Often use same embedding
cfg.corr_dim_lag = cfg.rqa_time_delay                   # Often use same lag
cfg.corr_dim_rvals_count = 20                           # Number of radii nolds should check
cfg.corr_dim_debug_plot = False                         # Generate log-log plot from nolds?

# --- Information Flow Analysis ---
cfg.enable_information_flow = True # Requires 'pyinform' library: pip install pyinform
cfg.info_flow_observable_driver = None # Set automatically based on driver type (e.g., 'z' for lorenz)
cfg.info_flow_observable_system = "Position <x>" # Which system observable to analyze
cfg.info_flow_k = 5 # History length for Transfer Entropy/Mutual Information
cfg.info_flow_lag = 1 # Time lag between driver history and system future (pyinform TE uses implicit 1)
cfg.info_flow_local = False # Calculate local transfer entropy (more intensive)?

# --- Classical Mechanics Simulation Parameters ---
cfg.enable_classical_simulation = True # Make sure this is True if you intend to run it
cfg.classical_results_dir = os.path.join(package_root_dir, "classical_mechanics", "results")
cfg.classical_model_type = 'coupled_4d_oscillator' # Example: 'projection_4d_torus', 'coupled_4d_oscillator', 'rossler_hyperchaos'
# Add specific parameters for the chosen classical model here...
cfg.classical_oscillator_freqs = [1.0, 1.1, 1.2, 1.3]
cfg.classical_oscillator_couplings = np.array([ [0.0, 0.1, 0.0, 0.05], [0.1, 0.0, 0.1, 0.0], [0.0, 0.1, 0.0, 0.1], [0.05, 0.0, 0.1, 0.0] ])
cfg.classical_torus_flow_vector = [1.0, np.sqrt(2.0), np.sqrt(3.0), np.sqrt(5.0)]
cfg.classical_rossler_a = 0.25; cfg.classical_rossler_b = 3.0; cfg.classical_rossler_c = 0.5; cfg.classical_rossler_d = 0.5; cfg.classical_rossler_e = 0.05
cfg.classical_T = 100.0 # Simulation time for classical model
cfg.classical_dt = 0.01 # Time step for classical model integrator

# --- CORRECTED Initial State ---
# NOTE: Initial state length MUST match the chosen model's state dimension:
# - coupled_4d_oscillator: state_dim=8 [x, y, z, w, vx, vy, vz, vw]
# - torus_4d_flow: state_dim=4 [x, y, z, w]
# - rossler_hyperchaos: state_dim=4 [x, y, z, w]
# Example for coupled_4d_oscillator (8 elements: 4 positions, 4 velocities)
cfg.classical_initial_state = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Example for rossler_hyperchaos or torus_4d_flow (4 elements)
# cfg.classical_initial_state = np.array([1.0, 0.0, 1.0, 0.0])
# --------------------------------

cfg.classical_analyze_3d_subspace = True # Calculate LLE/CorrDim on the projected 3D trajectory?
cfg.classical_save_plots = True
# Phase 3: Classical Driver Coupling
cfg.classical_enable_driver_coupling = True # Example: Enable coupling
cfg.classical_driver_coupling_param = 'alpha' # Which quantum coupling param relates to classical coupling strength?
cfg.classical_driver_coupling_strength = 1.0 # Scaling factor for classical coupling (multiplies quantum alpha/epsilon)
cfg.classical_driven_oscillator_index = 0 # Index (0-3) of oscillator driven in coupled model
cfg.classical_driven_dimension_index = 0 # Index (0-3) of dimension driven in torus model

# --- LLM Reporting (Phase 2/3/4) ---
cfg.enable_openai_reporting = True # Renamed, but functionally similar role for enabling any LLM reporting
# Phase 4: LLM Choice & Keys
cfg.llm_choice = 'gemini' # Options: 'openai', 'gemini', 'none'
cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_NOT_FOUND")
cfg.openai_model = "gpt-4-turbo-preview" # Or other suitable model like "gpt-4", "gpt-3.5-turbo"
cfg.gemini_api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_NOT_FOUND")
cfg.gemini_model = "gemini-1.5-pro-latest" # Or other suitable model

# --- Phase 4: Orchestration Configuration ---
cfg.enable_orchestration = False    # Master flag to run the AI orchestration loop
cfg.orchestration_goal = "Investigate how quantum system complexity (measured by LLE, DET, CorrDim) is influenced by different chaotic drivers (Lorenz vs Rossler vs Logistic Map) and coupling strength (alpha)." # Initial research goal for the orchestrator
cfg.orchestration_max_iterations = 3 # Maximum number of Plan-Execute-Analyze cycles
cfg.orchestration_planning_llm = 'gemini' # Which LLM to use for planning
cfg.orchestration_analysis_llm = 'gemini' # Which LLM to use for analysis interpretation
cfg.orchestration_reporting_llm = 'gemini'# Which LLM to use for report writing
# Simple heuristic for planner: how many samples in a sweep?
cfg.orchestration_planning_heuristic_num_samples = 5 # e.g., how many samples for a planned sweep
# Flag to control whether meta-analysis functions are called within the orchestrator loop
# vs. just generating the DataFrame (useful for debugging/performance)
cfg.orchestration_perform_meta_analysis = True
# Location for saving orchestrator state (optional but useful)
cfg.orchestrator_state_file = os.path.join(cfg.orchestrator_state_dir, "orchestrator_state.json")


# --- Sanity Checks ---
# Check if N is power of 2
if not (cfg.N > 0 and (cfg.N & (cfg.N - 1) == 0)):
     pass # Defer warning until logging is configured if run as main

# Check for API key placeholder AFTER attempting to load from .env/environ
if cfg.enable_openai_reporting and cfg.llm_choice == 'openai' and (not cfg.openai_api_key or cfg.openai_api_key == "YOUR_API_KEY_NOT_FOUND"):
    pass # Defer warning
if cfg.enable_openai_reporting and cfg.llm_choice == 'gemini' and (not cfg.gemini_api_key or cfg.gemini_api_key == "YOUR_API_KEY_NOT_FOUND"):
    pass # Defer warning

# Ensure info flow observable has a default if Lorenz is used
if cfg.driver_type == 'lorenz' and cfg.info_flow_observable_driver is None:
    cfg.info_flow_observable_driver = cfg.lorenz_use_component

# --- END OF FILE quantum_chaos_sim/config.py ---