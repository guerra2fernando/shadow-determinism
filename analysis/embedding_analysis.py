# --- START OF FILE quantum_chaos_sim/analysis/embedding_analysis.py ---
"""
Functions for analyzing the results of signal embedding experiments.
Attempts to decode the embedded message from the specified observable.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import os

# --- Import common utilities ---
try:
    from .common_analysis_utils import (get_results_dir, get_plot_dir,
                                        save_or_show_plot, load_observable_data,
                                        find_config_in_df, get_config_value,
                                        sanitize_filename)
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError as e:
    ANALYSIS_MODULES_AVAILABLE = False
    _analysis_import_error = e
    # Define dummy functions if import fails, although this analysis won't work anyway
    def get_results_dir(df=None, config_obj=None): return "."
    def get_plot_dir(df=None, config_obj=None): return "."
    def save_or_show_plot(fig, filepath, save_flag): pass
    def load_observable_data(run_label, observable_name, results_dir): return None
    def find_config_in_df(df, run_label): return None
    def get_config_value(config_obj, attr, default=None): return default
    def sanitize_filename(name): return str(name)

logger = logging.getLogger(__name__)

if not ANALYSIS_MODULES_AVAILABLE:
    logger.error(f"Failed to import common analysis utilities: {_analysis_import_error}. Embedding analysis functionality will be limited.")

# --- Decoding Function ---

def decode_embedded_signal(results_df, config_obj,
                            observable_name="Position <x>",
                            param_to_modulate='alpha',
                            method='thresholding'):
    """
    Attempts to decode a message embedded by modulating a parameter,
    observing the effect on a specified system observable.

    Args:
        results_df (pd.DataFrame): DataFrame containing experiment results.
        config_obj (SimpleNamespace): Base configuration object.
        observable_name (str): The observable to analyze for decoding.
        param_to_modulate (str): The parameter that was modulated to embed.
        method (str): Decoding method ('thresholding', 'correlation' - placeholder).

    Returns:
        tuple: (decoded_bits, metrics, plot_filenames)
               decoded_bits (list or None): The list of decoded bits (0s and 1s).
               metrics (dict or None): Dictionary containing decoding performance (e.g., BER).
               plot_filenames (list): List of filenames for generated plots.
               Returns (None, None, []) if decoding fails at any critical step.
    """
    logger.info(f"Attempting to decode embedded signal in '{param_to_modulate}' using observable '{observable_name}'.")
    plot_filenames = []
    results_dir = get_results_dir(results_df, config_obj)
    plot_dir = get_plot_dir(results_df, config_obj)

    # --- Find Baseline Run ---
    baseline_label = f"Embed_{param_to_modulate}_Baseline"
    baseline_row = results_df[results_df['run_label'] == baseline_label]
    if baseline_row.empty:
        logger.error(f"Could not find baseline run '{baseline_label}' in DataFrame. Skipping decoding.")
        return None, None, []
    baseline_config = find_config_in_df(results_df, baseline_label)
    if baseline_config is None:
        logger.error(f"Could not reconstruct config for baseline run '{baseline_label}'. Skipping decoding.")
        return None, None, []

    # --- Load Baseline Data ---
    baseline_data = load_observable_data(baseline_label, observable_name, results_dir)
    if baseline_data is None:
        logger.error(f"Could not load baseline observable '{observable_name}' for {baseline_label}. Skipping decoding.")
        # --- FIXED: Ensure 3 values are returned on error ---
        return None, None, []

    t_grid = get_config_value(baseline_config, 't_grid', None)
    if t_grid is None or len(t_grid) != len(baseline_data):
         logger.error(f"Time grid missing or length mismatch for baseline run {baseline_label}. Skipping decoding.")
         return None, None, []

    # --- Find Message Run(s) ---
    # Assume only one message run for simplicity in this example
    message_run_pattern = f"Embed_{param_to_modulate}_Msg"
    message_runs = results_df[results_df['run_label'].str.startswith(message_run_pattern, na=False)]

    if message_runs.empty:
        logger.warning(f"No message embedding runs found starting with '{message_run_pattern}'. Skipping decoding.")
        return None, None, []

    # Process the first message run found
    message_label = message_runs['run_label'].iloc[0]
    logger.info(f"Processing message run: {message_label}")
    message_config = find_config_in_df(results_df, message_label)
    if message_config is None:
        logger.error(f"Could not reconstruct config for message run '{message_label}'. Skipping decoding.")
        return None, None, []

    # --- Load Message Data ---
    message_data = load_observable_data(message_label, observable_name, results_dir)
    if message_data is None:
        logger.error(f"Could not load message observable '{observable_name}' for {message_label}. Skipping decoding.")
        return None, None, []
    if len(message_data) != len(baseline_data):
        logger.error(f"Length mismatch between message ({len(message_data)}) and baseline ({len(baseline_data)}) data for {observable_name}. Skipping.")
        return None, None, []

    # --- Get Original Message and Schedule Info ---
    original_schedule = get_config_value(message_config, f"{param_to_modulate}_schedule", None)
    if original_schedule is None or not isinstance(original_schedule, list) or len(original_schedule) < 2:
        logger.error(f"Could not retrieve valid modulation schedule for '{message_label}'. Skipping decoding.")
        return None, None, []

    # Infer original bits and bit duration from schedule
    try:
        base_value = original_schedule[0][1] # Value at t=0
        # Find high/low mod values from config (might be slightly different if noise added)
        bit_high_mod = get_config_value(baseline_config, 'embedding_bit_high_alpha_mod', 0.1) if param_to_modulate == 'alpha' else get_config_value(baseline_config, 'embedding_bit_high_epsilon_mod', 0.1)
        bit_low_mod = get_config_value(baseline_config, 'embedding_bit_low_alpha_mod', -0.1) if param_to_modulate == 'alpha' else get_config_value(baseline_config, 'embedding_bit_low_epsilon_mod', -0.1)

        expected_high_val = base_value + bit_high_mod
        expected_low_val = base_value + bit_low_mod

        original_bits = []
        bit_times = []
        for i in range(1, len(original_schedule)):
            t_start, val = original_schedule[i]
            t_prev = original_schedule[i-1][0]
            # If value changes from base, it's the start of a bit modulation
            if not np.isclose(val, base_value):
                 # Check if it matches expected high or low modulation
                 if np.isclose(val, expected_high_val):
                     original_bits.append(1)
                     bit_times.append(t_start)
                 elif np.isclose(val, expected_low_val):
                     original_bits.append(0)
                     bit_times.append(t_start)
                 else:
                     # Value changed but doesn't match expected modulation - error?
                     logger.warning(f"Unexpected modulation value {val:.3f} found at t={t_start:.2f} in schedule. Expected ~{expected_high_val:.3f} or ~{expected_low_val:.3f}.")
                     # Try guessing based on difference from base
                     if val > base_value: original_bits.append(1)
                     else: original_bits.append(0)
                     bit_times.append(t_start)

        if not original_bits:
            logger.error("Could not infer original message bits from the schedule.")
            return None, None, []

        # Estimate bit duration (assume constant)
        if len(bit_times) > 1:
            bit_duration_est = np.mean(np.diff(bit_times))
        elif len(original_schedule) > 2: # Try from first modulation end time
             bit_duration_est = original_schedule[2][0] - original_schedule[1][0]
        else: # Fallback from config if possible
             bit_duration_est = get_config_value(baseline_config, 'embedding_bit_duration', None)
             if bit_duration_est is None:
                 T_total = get_config_value(baseline_config, 'T', 50.0)
                 bit_duration_est = T_total / len(original_bits)
                 logger.warning(f"Estimating bit duration as T/N_bits = {bit_duration_est:.2f}")

        logger.info(f"Inferred Original Message: {original_bits} (Duration ~ {bit_duration_est:.2f})")

    except Exception as e_infer:
        logger.error(f"Error inferring original message from schedule: {e_infer}", exc_info=True)
        return None, None, []

    # --- Decoding Logic ---
    decoded_bits = []
    signal_difference = message_data - baseline_data
    time_points_per_bit = int(bit_duration_est / (t_grid[1] - t_grid[0])) if bit_duration_est > 0 else 1
    time_points_per_bit = max(1, time_points_per_bit) # Ensure at least 1 point

    if method == 'thresholding':
        # Simple method: Check the average difference during each bit interval
        for i in range(len(original_bits)):
            bit_start_index = int(bit_times[i] / (t_grid[1] - t_grid[0]))
            bit_end_index = bit_start_index + time_points_per_bit
            bit_end_index = min(bit_end_index, len(signal_difference)) # Ensure index is within bounds

            if bit_start_index >= bit_end_index:
                 logger.warning(f"Bit interval {i} is empty or invalid ({bit_start_index} >= {bit_end_index}). Assigning default bit 0.")
                 decoded_bits.append(0) # Assign default or handle error
                 continue

            avg_diff_in_bit = np.mean(signal_difference[bit_start_index:bit_end_index])

            # Simple thresholding (needs tuning): If diff > 0, guess 1, else 0
            # More robust: Compare to expected high/low differences if known
            # For now, use simple threshold > 0
            if avg_diff_in_bit > 0: # Assumes high mod increases the observable average
                decoded_bits.append(1)
            else:
                decoded_bits.append(0)

    # elif method == 'correlation':
        # Placeholder: Correlate signal_difference in each interval with expected templates
        # logger.warning("Correlation decoding method not implemented. Using thresholding.")
        # decoded_bits = [0] * len(original_bits) # Fallback
    else:
        logger.error(f"Unknown decoding method: {method}")
        return None, None, []

    # --- Calculate Metrics ---
    metrics = {}
    if len(decoded_bits) == len(original_bits):
        num_errors = sum(1 for i in range(len(original_bits)) if original_bits[i] != decoded_bits[i])
        ber = num_errors / len(original_bits)
        metrics['BitErrorRate'] = ber
        metrics['NumErrors'] = num_errors
        metrics['NumBits'] = len(original_bits)
        logger.info(f"Decoding complete. Decoded: {decoded_bits}. BER: {ber:.3f} ({num_errors}/{len(original_bits)} errors)")
    else:
        logger.error(f"Length mismatch between original ({len(original_bits)}) and decoded ({len(decoded_bits)}) bits.")
        metrics['BitErrorRate'] = np.nan
        metrics['NumErrors'] = np.nan
        metrics['NumBits'] = len(original_bits) # Report original length

    # --- Generate Plots ---
    save_flag = get_config_value(config_obj, 'save_results', False) and get_config_value(config_obj, 'save_plots_comparison', True)

    # Plot 1: Baseline vs Message Signal
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(t_grid, baseline_data, label=f"Baseline ({baseline_label})", alpha=0.7)
    ax1.plot(t_grid, message_data, label=f"Message ({message_label})", alpha=0.7)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(observable_name)
    ax1.set_title(f"Observable Signal Comparison: Baseline vs. Message ({param_to_modulate})")
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    plt.tight_layout()
    fname1 = f"decoding_signal_comparison_{sanitize_filename(message_label)}.png"
    fpath1 = os.path.join(plot_dir, fname1)
    saved_path1 = save_or_show_plot(fig1, fpath1, save_flag)
    if saved_path1: plot_filenames.append(os.path.basename(saved_path1))

    # Plot 2: Signal Difference and Decoded Bits
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax2a.plot(t_grid, signal_difference, label="Signal Difference (Message - Baseline)")
    ax2a.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax2a.set_ylabel("Difference in " + observable_name)
    ax2a.set_title(f"Signal Difference and Decoded Bits (BER: {metrics.get('BitErrorRate', 'N/A'):.3f})")
    ax2a.grid(True, alpha=0.5)
    ax2a.legend(loc='upper right')

    # Overlay original and decoded bits
    decoded_bit_signal = np.zeros_like(t_grid)
    original_bit_signal = np.zeros_like(t_grid)
    for i in range(len(original_bits)):
        bit_start_index = int(bit_times[i] / (t_grid[1] - t_grid[0]))
        bit_end_index = bit_start_index + time_points_per_bit
        bit_end_index = min(bit_end_index, len(signal_difference))
        if bit_start_index < bit_end_index:
             original_bit_signal[bit_start_index:bit_end_index] = original_bits[i]
             if i < len(decoded_bits):
                 decoded_bit_signal[bit_start_index:bit_end_index] = decoded_bits[i]

    ax2b.step(t_grid, original_bit_signal, where='post', label='Original Bits', linestyle='--')
    ax2b.step(t_grid, decoded_bit_signal, where='post', label='Decoded Bits', alpha=0.7)
    ax2b.set_yticks([0, 1])
    ax2b.set_ylim(-0.1, 1.1)
    ax2b.set_xlabel("Time")
    ax2b.set_ylabel("Bit Value")
    ax2b.legend(loc='upper right')
    ax2b.grid(True, alpha=0.5)

    plt.tight_layout()
    fname2 = f"decoding_difference_bits_{sanitize_filename(message_label)}.png"
    fpath2 = os.path.join(plot_dir, fname2)
    saved_path2 = save_or_show_plot(fig2, fpath2, save_flag)
    if saved_path2: plot_filenames.append(os.path.basename(saved_path2))

    return decoded_bits, metrics, plot_filenames

# --- END OF FILE quantum_chaos_sim/analysis/embedding_analysis.py ---