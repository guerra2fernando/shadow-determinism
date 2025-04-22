# --- START OF FILE quantum_chaos_sim/results_handler.py ---
import pandas as pd
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def flatten_config(config_obj, prefix='config_'):
    """ Flattens a SimpleNamespace config into a dictionary for DataFrame rows. """
    flat_dict = {}
    if not config_obj: return flat_dict
    for key, value in config_obj.__dict__.items():
        if isinstance(value, (int, float, str, bool)):
            flat_dict[prefix + key] = value
        elif isinstance(value, np.ndarray) and value.ndim == 1 and value.size < 10: # Store small arrays like ICs
             for i, item in enumerate(value):
                 flat_dict[f"{prefix}{key}_{i}"] = item
        # Add elif for other types if needed (e.g., lists)
        # Skip complex objects or large arrays
    return flat_dict

def flatten_metrics(metrics_dict, prefix='metric_'):
    """ Flattens a metrics dictionary. Handles potential nested dicts if needed."""
    flat_dict = {}
    if not metrics_dict: return flat_dict
    for key, value in metrics_dict.items():
         # Handle numpy arrays (e.g., local TE) - store mean/std? Or just flag existence?
         if isinstance(value, np.ndarray):
              if value.size == 1:
                   flat_dict[prefix + key] = value.item()
              elif value.size > 1 and value.size < 50: # Store small arrays directly? Risky for structure.
                   # Option 1: Store mean/std
                   # flat_dict[prefix + key + '_mean'] = np.nanmean(value)
                   # flat_dict[prefix + key + '_std'] = np.nanstd(value)
                   # Option 2: Store as string (less useful)
                   # flat_dict[prefix + key] = np.array2string(value, precision=3)
                   # Option 3: Skip for now
                   pass
         elif isinstance(value, (int, float, str, bool, np.number, np.bool_)):
              flat_dict[prefix + key] = value
         # Skip other types
    return flat_dict

def prepare_result_for_dataframe(result_dict):
    """ Combines flattened config and metrics with other run info. """
    if not result_dict: return {}

    row_data = {
        'run_label': result_dict.get('run_label', 'N/A'),
        'success': result_dict.get('success', False),
        'error_message': result_dict.get('error_message', None),
    }
    row_data.update(flatten_config(result_dict.get('config')))
    row_data.update(flatten_metrics(result_dict.get('metrics')))

    # Add key observable summaries explicitly
    observables = result_dict.get('observables', {})
    if 'Shannon Entropy S(x)' in observables:
         entropy_series = observables['Shannon Entropy S(x)']
         if isinstance(entropy_series, np.ndarray) and entropy_series.size > 0:
              row_data['observable_final_entropy'] = entropy_series[-1]

    # Add paths to plots if generated
    plot_files = observables.get('plot_files_run', [])
    if plot_files:
        # Store as a delimited string or just the count? String is better.
        row_data['plot_files'] = ";".join(plot_files)

    return row_data


def save_results_df(df, filepath, format='hdf', key='simulation_data'):
    """ Saves the results DataFrame to HDF5 or CSV. """
    logger.info(f"Saving results DataFrame ({df.shape[0]} rows, {df.shape[1]} cols) to {filepath} (Format: {format})...")
    try:
        if format == 'hdf':
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Use mode='a' to append if file exists, 'w' to overwrite.
            # For simplicity, let's overwrite ('w') for now.
            # For appending, need careful key handling or different strategy.
            df.to_hdf(filepath, key=key, mode='w', format='table', complevel=9, complib='blosc')
        elif format == 'csv':
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
        else:
            logger.error(f"Unsupported results storage format: {format}")
            return False
        logger.info("DataFrame saved successfully.")
        return True
    except ImportError as e:
        if 'tables' in str(e) and format == 'hdf':
            logger.error("Failed to save DataFrame to HDF5. Package 'tables' not found. Install using 'pip install tables'.")
        else:
            logger.error(f"Failed to save DataFrame: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {filepath}: {e}", exc_info=True)
        return False

def load_results_df(filepath, format='hdf', key='simulation_data'):
    """ Loads the results DataFrame from HDF5 or CSV. """
    logger.info(f"Loading results DataFrame from {filepath} (Format: {format})...")
    if not os.path.exists(filepath):
        logger.warning(f"Results file not found: {filepath}")
        return None
    try:
        if format == 'hdf':
            df = pd.read_hdf(filepath, key=key)
        elif format == 'csv':
            df = pd.read_csv(filepath)
        else:
            logger.error(f"Unsupported results storage format: {format}")
            return None
        logger.info(f"DataFrame loaded successfully ({df.shape[0]} rows, {df.shape[1]} cols).")
        return df
    except ImportError as e:
        if 'tables' in str(e) and format == 'hdf':
            logger.error("Failed to load DataFrame from HDF5. Package 'tables' not found. Install using 'pip install tables'.")
        else:
            logger.error(f"Failed to load DataFrame: {e}", exc_info=True)
        return None
    except KeyError as e:
         if format == 'hdf':
              logger.error(f"Failed to load DataFrame from HDF5: Key '{key}' not found in file {filepath}. Error: {e}")
         else:
              logger.error(f"Failed to load DataFrame: {e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Failed to load DataFrame from {filepath}: {e}", exc_info=True)
        return None

# --- END OF FILE quantum_chaos_sim/results_handler.py ---