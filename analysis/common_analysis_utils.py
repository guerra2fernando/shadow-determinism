# --- START OF FILE quantum_chaos_sim/analysis/common_analysis_utils.py ---
"""
Utility functions shared across different meta-analysis modules.
Includes reconstructing config objects from flattened DataFrame columns.
"""
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
from types import SimpleNamespace
import ast # For safely evaluating string representations of lists/tuples

logger = logging.getLogger(__name__)

# Attempt to import sanitize_filename, otherwise define a basic version
try:
    # Assuming plot_wavefunction is in a sibling 'visualization' directory
    from ..visualization.plot_wavefunction import _sanitize_filename as sanitize_filename
except ImportError:
    logger.warning("Could not import _sanitize_filename from visualization. Using basic version.")
    def sanitize_filename(name):
        """Basic version of filename sanitizer."""
        if not isinstance(name, str): name = str(name)
        chars_to_remove = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '[', ']', ',', '(', ')', '=']
        sanitized = name
        for char in chars_to_remove: sanitized = sanitized.replace(char, '')
        sanitized = sanitized.replace(' ', '_').replace('.', 'p')
        max_len = 100
        if len(sanitized) > max_len: sanitized = sanitized[:max_len] + "_trunc"
        return sanitized

# Attempt to import _save_or_show_plot, otherwise define a basic version
try:
    # Assuming plot_wavefunction is in a sibling 'visualization' directory
    from ..visualization.plot_wavefunction import _save_or_show_plot as save_or_show_plot
except ImportError:
    logger.warning("Could not import _save_or_show_plot from visualization. Using basic version.")
    def save_or_show_plot(fig, filepath, save_flag):
        """Basic version of saving plot."""
        plot_dir = os.path.dirname(filepath)
        if save_flag:
            try:
                os.makedirs(plot_dir, exist_ok=True) # Ensure directory exists
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Plot saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot to {filepath}: {e}", exc_info=True)
        else:
            logger.info("Displaying plot (basic fallback - may fail without GUI).")
            try:
                plt.show() # Might fail
            except Exception as e_show:
                 if "TclError" in str(e_show) or "no display name" in str(e_show).lower() or "cannot connect to X server" in str(e_show).lower():
                      logger.warning(f"Cannot display plot interactively (likely no GUI): {e_show}")
                 else:
                      logger.warning(f"Could not display plot interactively: {e_show}", exc_info=True)
        plt.close(fig)


def get_results_dir(df=None, config_obj=None):
    """ Safely gets the main results directory. """
    if config_obj and hasattr(config_obj, 'results_dir'): return config_obj.results_dir
    if df is not None and hasattr(df, 'attrs') and 'results_dir' in df.attrs: return df.attrs['results_dir']
    # Fallback
    try:
        # Assumes analysis/ is one level down from quantum_chaos_sim package root
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_dir = os.path.join(pkg_root, "results")
    except NameError: # __file__ might not be defined in some environments (e.g., interactive)
        default_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir

def get_plot_dir(df=None, config_obj=None):
    """ Safely gets the plot directory. """
    if config_obj and hasattr(config_obj, 'plot_dir'): return config_obj.plot_dir
    if df is not None and hasattr(df, 'attrs') and 'plot_dir' in df.attrs: return df.attrs['plot_dir']
    # Fallback to being inside results_dir
    results_dir = get_results_dir(df, config_obj)
    default_dir = os.path.join(results_dir, "plots")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


def load_observable_data(run_label, observable_name, results_dir):
    """
    Loads saved observable data array (.npy) for a specific run.

    Args:
        run_label (str): The unique label of the simulation run.
        observable_name (str): The name of the observable (e.g., "Position <x>").
        results_dir (str): Path to the main results directory.

    Returns:
        np.ndarray or None: The loaded numpy array, or None if the file
                           is not found or fails to load.
    """
    obs_data_dir = os.path.join(results_dir, 'observables_data')
    if not os.path.isdir(obs_data_dir):
        logger.debug(f"Observable data directory not found: {obs_data_dir}")
        return None

    safe_label = sanitize_filename(run_label)
    safe_obs_name = sanitize_filename(observable_name)
    obs_filename = f"{safe_label}_observable_{safe_obs_name}.npy"
    obs_filepath = os.path.join(obs_data_dir, obs_filename)

    # *** Add logging here to confirm the path being checked ***
    logger.debug(f"Attempting to load observable from: {obs_filepath}")

    if os.path.exists(obs_filepath):
        try:
            data = np.load(obs_filepath)
            logger.debug(f"Loaded observable '{observable_name}' for run '{run_label}' from {obs_filepath}")
            return data
        except Exception as e:
            logger.error(f"Failed to load observable file {obs_filepath}: {e}", exc_info=True)
            return None
    else:
        logger.debug(f"Observable file not found: {obs_filepath}")
        return None

def get_config_value(config_obj, attr, default=None):
    """Safely gets an attribute from config_obj or returns default."""
    # Handle case where config_obj might be a dictionary (e.g., from results_dict)
    # or a SimpleNamespace stored as a dict in the DataFrame
    if isinstance(config_obj, dict):
        return config_obj.get(attr, default)
    # Handle case where it's a SimpleNamespace or similar object
    return getattr(config_obj, attr, default) if config_obj else default

def find_config_in_df(df, run_label):
    """
    Finds and reconstructs the config object (SimpleNamespace) for a given
    run_label from the flattened config_* columns in the DataFrame.
    """
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty in find_config_in_df.")
        return None

    try:
        run_row = df[df['run_label'] == run_label]
    except KeyError:
        logger.error("DataFrame does not contain 'run_label' column.")
        return None
    except Exception as e:
         logger.error(f"Error filtering DataFrame for run_label '{run_label}': {e}")
         return None

    if run_row.empty:
        logger.warning(f"Could not find run_label '{run_label}' in DataFrame.")
        return None

    # --- Reconstruct config from flattened columns ---
    config_dict = {}
    # Use squeeze() to get a Series if only one row matches, otherwise take the first row
    row_data = run_row.iloc[0] if len(run_row) == 1 else run_row.squeeze() if len(run_row) == 1 else run_row.iloc[0]

    prefix = 'config_'
    for col_name in row_data.index:
        if col_name.startswith(prefix):
            param_name = col_name[len(prefix):]
            value = row_data[col_name]

            # Attempt basic type inference/conversion for non-numeric/non-bool values stored as strings/objects
            if isinstance(value, str):
                # Check for list/tuple/array representation '[...]' or '(...)'
                if (value.startswith('[') and value.endswith(']')) or \
                   (value.startswith('(') and value.endswith(')')):
                    try:
                        # Use literal_eval for safety (evaluates basic Python literals)
                        parsed_value = ast.literal_eval(value)
                        # Heuristic: Convert back to numpy array if it's a list of numbers
                        if isinstance(parsed_value, list) and parsed_value and all(isinstance(x, (int, float)) for x in parsed_value):
                             value = np.array(parsed_value)
                        # Heuristic: Convert tuple of numbers back to numpy array (e.g., lorenz_initial_state)
                        elif isinstance(parsed_value, tuple) and parsed_value and all(isinstance(x, (int, float)) for x in parsed_value):
                             value = np.array(parsed_value)
                        else:
                             value = parsed_value # Keep as list/tuple otherwise
                    except (ValueError, SyntaxError, TypeError, MemoryError) as e_parse:
                        # Keep as string if parsing fails, log warning
                        logger.debug(f"Could not parse string '{value}' for param '{param_name}' using ast.literal_eval: {e_parse}. Keeping as string.")
                        pass # Keep original string value
                # Check for boolean strings
                elif value.lower() == 'true': value = True
                elif value.lower() == 'none': value = None # Handle 'None' string
                elif value.lower() == 'false': value = False
                # Check for NaN strings
                elif value.lower() == 'nan': value = np.nan

            # Handle pandas NA specifically before numeric conversion attempt
            elif pd.isna(value):
                 value = None # Represent pandas NA as Python None in the config

            # Convert numeric types if they ended up as objects somehow, but skip None/bool
            elif not isinstance(value, (int, float, bool, list, tuple, np.ndarray)) and value is not None:
                # Use pd.to_numeric for robust conversion, coercing errors
                value_numeric = pd.to_numeric(value, errors='coerce')
                if pd.notna(value_numeric): # If conversion successful
                     # Preserve integer type if possible
                     if np.equal(np.mod(value_numeric, 1), 0):
                          value = int(value_numeric)
                     else:
                          value = float(value_numeric)
                # Else: keep original value if conversion to numeric fails

            config_dict[param_name] = value

    if not config_dict:
        logger.warning(f"No columns starting with '{prefix}' found for run '{run_label}'. Cannot reconstruct config.")
        return None

    # --- Post-processing: Handle Schedules specifically ---
    # Schedules might be stored as strings; attempt to parse them back.
    for schedule_key in ['alpha_schedule', 'epsilon_schedule']:
        if schedule_key in config_dict and isinstance(config_dict[schedule_key], str):
             schedule_str = config_dict[schedule_key]
             # Check if it looks like a list of tuples/lists before attempting parse
             if schedule_str.startswith('[') and schedule_str.endswith(']') and ('(' in schedule_str or '[' in schedule_str):
                 try:
                      parsed_schedule = ast.literal_eval(schedule_str)
                      # Validate format: list of (number, number) pairs
                      if isinstance(parsed_schedule, list) and \
                         all(isinstance(item, (tuple, list)) and len(item) == 2 and \
                             isinstance(item[0], (int, float)) and isinstance(item[1], (int, float)) \
                             for item in parsed_schedule):
                           config_dict[schedule_key] = parsed_schedule
                           logger.debug(f"Successfully parsed schedule string for '{schedule_key}'.")
                      else:
                           logger.warning(f"Parsed schedule for '{schedule_key}' is not a valid list of pairs: {parsed_schedule}. Keeping as string.")
                 except (ValueError, SyntaxError, TypeError, MemoryError) as e_sched_parse:
                      logger.warning(f"Could not parse schedule string for '{schedule_key}' using ast.literal_eval: {e_sched_parse}. Keeping as string: '{schedule_str}'")
        elif schedule_key in config_dict and config_dict[schedule_key] is None:
             config_dict[schedule_key] = None # Ensure None remains None if stored explicitly
        elif schedule_key not in config_dict:
             config_dict[schedule_key] = None # Add as None if missing entirely


    # --- Create SimpleNamespace ---
    try:
        config_ns = SimpleNamespace(**config_dict)
        logger.debug(f"Reconstructed config for run '{run_label}'.") # Removed dump for brevity
        return config_ns
    except Exception as e:
        logger.error(f"Failed to create SimpleNamespace from reconstructed config dict for run '{run_label}': {e}. Dict was: {config_dict}", exc_info=True)
        return None

# --- END OF FILE quantum_chaos_sim/analysis/common_analysis_utils.py ---