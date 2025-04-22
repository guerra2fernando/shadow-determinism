# quantum_chaos_sim/analysis/gating_analysis.py
"""
Provides functions for analyzing and plotting transient behavior in gating experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
from .common_analysis_utils import (load_observable_data, get_results_dir,
                                    get_plot_dir, save_or_show_plot,
                                    find_config_in_df, get_config_value,
                                    sanitize_filename)

logger = logging.getLogger(__name__)

def analyze_gating_transients(results_df, base_config, param_gated='alpha', observable='Position <x>'):
    """
    Analyzes and plots the transient behavior around gating switch times
    by loading saved observable data.

    Args:
        results_df (pd.DataFrame): DataFrame containing metadata about runs.
                                   Must contain 'run_label' and 'config' columns.
        base_config: The base configuration object (needed for t_grid, plot_dir etc.).
        param_gated (str): The parameter that was gated ('alpha' or 'epsilon'). Default 'alpha'.
        observable (str): The observable time series to plot (e.g., 'Position <x>').

    Returns:
        None: Generates and saves plots to the configured plot directory.
    """
    logger.info(f"Analyzing gating transients for '{param_gated}' using observable '{observable}'.")
    if results_df is None or results_df.empty:
        logger.warning("Gating analysis skipped: DataFrame is empty.")
        return
    if not base_config:
        logger.error("Gating analysis failed: Base config object is missing.")
        return

    results_dir = get_results_dir(results_df, base_config)
    plot_dir = get_plot_dir(results_df, base_config)
    t_grid = get_config_value(base_config, 't_grid', None)
    if t_grid is None:
        logger.error("Gating analysis failed: t_grid missing from base config.")
        return

    # Find relevant run labels
    label_on_off_on = f"Gating_{param_gated}_ON_OFF_ON"
    label_off_on_off = f"Gating_{param_gated}_OFF_ON_OFF"
    label_const_on = f"Gating_{param_gated}_ConstantON"
    label_const_off = f"Gating_{param_gated}_ConstantOFF"
    required_labels = [label_on_off_on, label_off_on_off, label_const_on, label_const_off]

    gating_runs_present = results_df[results_df['run_label'].isin(required_labels)]
    if gating_runs_present.empty:
        logger.warning(f"No gating runs found in DataFrame for parameter '{param_gated}'. Skipping analysis.")
        return

    # Load data for relevant runs
    observables_data = {}
    run_configs = {}
    successful_loads = 0
    for label in required_labels:
        if label in gating_runs_present['run_label'].values:
            run_config = find_config_in_df(results_df, label)
            if run_config is None:
                 logger.warning(f"Could not find config for run '{label}'.")
                 continue # Skip if config missing

            data = load_observable_data(label, observable, results_dir)
            if data is not None:
                if len(data) == len(t_grid):
                    observables_data[label] = data
                    run_configs[label] = run_config # Store config for schedule extraction
                    successful_loads += 1
                else:
                    logger.warning(f"Loaded observable '{observable}' for '{label}' has wrong length ({len(data)} vs {len(t_grid)}). Skipping run.")
            else:
                logger.warning(f"Could not load observable '{observable}' for run '{label}'. Skipping run.")
        else:
             logger.debug(f"Run label '{label}' not found in DataFrame.")


    if successful_loads < 2: # Need at least two curves to compare
        logger.error(f"Failed to load sufficient valid observable data ({successful_loads} runs loaded) for gating analysis of '{param_gated}'.")
        return

    # --- Extract Switch Times (best effort from loaded configs) ---
    switch_times = None
    schedule_label = label_on_off_on if label_on_off_on in run_configs else label_off_on_off
    if schedule_label in run_configs:
        schedule_config = run_configs[schedule_label]
        schedule = get_config_value(schedule_config, f"{param_gated}_schedule", None)
        if isinstance(schedule, list) and len(schedule) > 1:
            # Extract times where value changes (excluding t=0)
            switch_times = [item[0] for i, item in enumerate(schedule)
                            if i > 0 and not np.isclose(item[0], 0.0)
                            and not np.isclose(item[1], schedule[i-1][1])]
            if switch_times:
                logger.info(f"Found switch times for '{schedule_label}': {switch_times}")
            else:
                 logger.warning(f"Schedule found for {schedule_label}, but no value changes detected after t=0.")
        else:
            logger.warning(f"Could not find or parse '{param_gated}_schedule' in config for run '{schedule_label}'.")
    if switch_times is None:
        logger.warning("Could not determine switch times for gating analysis plots.")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        label_on_off_on: ('-', 'ON->OFF->ON'),
        label_off_on_off: ('--', 'OFF->ON->OFF'),
        label_const_on: (':', 'Constant ON'),
        label_const_off: ('-.', 'Constant OFF')
    }
    plotted_any = False

    for label, (style, plot_label) in styles.items():
        if label in observables_data:
            ax.plot(t_grid, observables_data[label], style, label=plot_label, alpha=0.85, lw=1.5)
            plotted_any = True

    if not plotted_any:
        logger.error(f"No data available to plot for gating transient analysis of '{param_gated}'.")
        plt.close(fig)
        return

    # Add switch time lines
    if switch_times:
        first_switch = True
        for t_switch in switch_times:
            ax.axvline(t_switch, color='r', linestyle='--', alpha=0.6, lw=1.0,
                       label='Switch Times' if first_switch else '_nolegend_')
            first_switch = False

    ax.set_xlabel("Time (t)")
    ax.set_ylabel(f"Observable: {observable}")
    ax.set_title(f"Gating Transient Analysis for '{param_gated}'")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    filename_base = f"gating_transient_{param_gated}_{sanitize_filename(observable)}"
    save_or_show_plot(fig, os.path.join(plot_dir, f"{filename_base}.png"), True)

    # --- Optional: Zoomed Plots around Switches ---
    if switch_times:
        logger.info("Generating zoomed plots around switch times...")
        window_half_width = (t_grid[-1] - t_grid[0]) * 0.05 # Zoom window +/- 5% of total time

        for i, t_switch in enumerate(switch_times):
            fig_zoom, ax_zoom = plt.subplots(figsize=(10, 6))
            plotted_any_zoom = False
            for label, (style, plot_label) in styles.items():
                if label in observables_data:
                    ax_zoom.plot(t_grid, observables_data[label], style, label=plot_label, alpha=0.85, lw=1.5)
                    plotted_any_zoom = True

            if plotted_any_zoom:
                ax_zoom.axvline(t_switch, color='r', linestyle='--', alpha=0.6, lw=1.0, label='Switch Time')
                ax_zoom.set_xlim(max(t_grid[0], t_switch - window_half_width),
                                 min(t_grid[-1], t_switch + window_half_width))
                # Auto-scale Y axis based on visible data
                visible_indices = (t_grid >= ax_zoom.get_xlim()[0]) & (t_grid <= ax_zoom.get_xlim()[1])
                min_y = np.inf; max_y = -np.inf
                for label in observables_data:
                     data_in_view = observables_data[label][visible_indices]
                     if len(data_in_view)>0:
                         min_y = min(min_y, np.min(data_in_view))
                         max_y = max(max_y, np.max(data_in_view))
                if np.isfinite(min_y) and np.isfinite(max_y):
                    yrange = max_y - min_y; buffer = max(0.05 * yrange, 1e-6)
                    ax_zoom.set_ylim(min_y - buffer, max_y + buffer)


                ax_zoom.set_xlabel("Time (t)")
                ax_zoom.set_ylabel(f"Observable: {observable}")
                ax_zoom.set_title(f"Gating Transient (Zoom @ t={t_switch:.2f}) - Param: '{param_gated}'")
                ax_zoom.legend(loc='best', fontsize='small')
                ax_zoom.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                filename_zoom = f"{filename_base}_zoom{i+1}_t{t_switch:.1f}.png".replace('.', 'p')
                save_or_show_plot(fig_zoom, os.path.join(plot_dir, filename_zoom), True)
            else:
                plt.close(fig_zoom)