# --- START OF FILE quantum_chaos_sim/reporting.py ---
import openai
import logging
import os
import numpy as np
import json # To handle numpy arrays in prompts if needed
import pandas as pd # Import pandas
from types import SimpleNamespace # Import SimpleNamespace

# --- Attempt to Import LLM Interface for API Call Fallback/Usage ---
try:
    from . import llm_interface
    LLM_INTERFACE_AVAILABLE = True
except ImportError:
    LLM_INTERFACE_AVAILABLE = False

# --- Attempt to Import plot_wavefunction for sanitization ---
# This might create a circular dependency if plot_wavefunction imports reporting,
# but it's only needed for _sanitize_filename. Consider moving sanitize elsewhere.
try:
    from .visualization.plot_wavefunction import _sanitize_filename
except ImportError:
    logger = logging.getLogger(__name__) # Define logger early for warning
    logger.warning("Could not import _sanitize_filename from visualization. Filenames in prompts might be less clean.")
    # Basic fallback sanitizer
    def _sanitize_filename(name):
        if not isinstance(name, str): name = str(name)
        chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '[', ']', ',', '(', ')', '=']
        sanitized = name
        for char in chars: sanitized = sanitized.replace(char, '')
        return sanitized.replace(' ', '_').replace('.', 'p')[:100]


logger = logging.getLogger(__name__)

# Helper to convert numpy types for JSON serialization if needed, though we focus on summaries
class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy types. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Format floats nicely for the prompt
            return _format_metric(obj) # Use consistent formatting
        if isinstance(obj, np.ndarray):
            # Convert arrays to lists (use with caution for large arrays)
            # Limit precision and size for prompts
            if obj.size > 50: # Limit array size in prompt
                first_elems_str = ""
                if obj.ndim == 1 and obj.size > 0:
                     first_elems_str = np.array2string(obj.flat[:min(10, obj.size)], precision=3, suppress_small=True, max_line_width=80) + "..."
                return f"Array (shape: {obj.shape} {first_elems_str})"
            elif obj.ndim == 1:
                 return np.array2string(obj, precision=3, suppress_small=True, separator=', ')
            else:
                 return f"Array (shape: {obj.shape})" # Simpler representation for prompt
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if hasattr(obj, 'tolist'): # Catch other numpy types like bool_
             return obj.tolist()
        if isinstance(obj, pd.Timestamp):
             return obj.isoformat()
        if obj is pd.NA:
            return None # Represent as None in JSON
        if isinstance(obj, SimpleNamespace):
             return vars(obj) # Convert to dictionary
        # Let the base class default method raise the TypeError
        return super(NpEncoder, self).default(obj)


def _format_metric(value, precision=4):
    """Formats a metric value nicely, handling NaN/NA/None and various types."""
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, np.integer):
        return f"{int(value)}"
    if isinstance(value, int):
        return f"{value}"
    if isinstance(value, np.floating):
        if np.isnan(value): return "N/A"
        return f"{value:.{precision}g}" # General format is good for prompts
    if isinstance(value, float):
        if np.isnan(value): return "N/A"
        return f"{value:.{precision}g}"
    if isinstance(value, np.ndarray):
        if value.size == 0: return "N/A (empty)"
        if value.ndim == 1 and value.size <= 10: # Show small 1D arrays
             return np.array2string(value, precision=precision, suppress_small=True, separator=', ')
        elif value.ndim == 1:
             avg = np.nanmean(value)
             return f"Avg: {_format_metric(avg, precision)} (Local, N={value.size})" # Indicate average of local values
        else:
             return f"Array (shape:{value.shape})"
    # Default to string conversion
    return str(value)


def generate_report_prompt(validation_results_df, base_config, comparison_metrics, hypothesis_abstract, classical_results=None, repeatability_summary=None, sensitivity_jacobian=None):
    """
    Formats results from a DataFrame and creates a prompt for an LLM (e.g., OpenAI, Gemini).
    Includes quantum experiment results summaries from the DataFrame, optional classical results,
    and optional repeatability/sensitivity summaries. Reflects Phase 3 capabilities.

    Args:
        validation_results_df (pd.DataFrame): DataFrame containing simulation results.
        base_config (SimpleNamespace): Base configuration object.
        comparison_metrics (dict): Dictionary containing L2 diffs and plot file list.
        hypothesis_abstract (str): The abstract/goal being tested.
        classical_results (dict, optional): Summary of representative classical run.
        repeatability_summary (dict, optional): Summary from analyze_repeatability.
        sensitivity_jacobian (pd.DataFrame, optional): Jacobian from calculate_sensitivity_jacobian.

    Returns:
        str: The formatted prompt string.
    """
    prompt = f"Analyze the results of a physics simulation validation suite.\n"
    prompt += f"The core hypothesis being tested is:\n'''\n{hypothesis_abstract}\n'''\n\n"
    prompt += f"--- Base Simulation Setup ---\n"

    # Safely access config attributes
    n_val = getattr(base_config, 'N', 'N/A')
    l_val = getattr(base_config, 'L', 'N/A')
    t_val = getattr(base_config, 'T', 'N/A')
    m_val = getattr(base_config, 'M', 'N/A')
    driver_type_val = getattr(base_config, 'driver_type', 'N/A')
    epsilon_val = getattr(base_config, 'epsilon', np.nan)
    alpha_val = getattr(base_config, 'alpha', np.nan)
    scheduling_enabled = getattr(base_config, 'enable_parameter_scheduling', False)

    prompt += f"Base parameters (from config): N={n_val}, L={l_val}, T={t_val}, M={m_val}\n"
    prompt += f"Default Driver Type: {driver_type_val}\n"
    prompt += f"Default coupling: epsilon={_format_metric(epsilon_val)}, alpha={_format_metric(alpha_val)}\n"
    prompt += f"Parameter Scheduling Enabled (Base): {scheduling_enabled}\n\n"

    prompt += "--- Quantum Experiments Summary (from DataFrame) ---\n"
    if validation_results_df is None or validation_results_df.empty:
        prompt += "No quantum simulation results available in the DataFrame.\n"
    else:
        num_runs = len(validation_results_df)
        num_successful = int(validation_results_df['success'].sum()) # Cast to int
        prompt += f"Total Runs Analyzed: {num_runs} ({num_successful} successful, {num_runs - num_successful} failed)\n"

        # Identify key parameters that varied
        config_cols = [col for col in validation_results_df.columns if col.startswith('config_')]
        varied_params = {}
        for col in config_cols:
             if validation_results_df[col].notna().any():
                  unique_vals = validation_results_df[col].nunique(dropna=True)
                  if unique_vals > 1:
                      param_name = col.replace('config_', '')
                      if param_name.endswith('_schedule'): continue # Skip schedule columns for brevity
                      if pd.api.types.is_numeric_dtype(validation_results_df[col].dropna()):
                          min_val = validation_results_df[col].min()
                          max_val = validation_results_df[col].max()
                          varied_params[param_name] = f"{unique_vals} unique values (Range: [{_format_metric(min_val)}, {_format_metric(max_val)}])"
                      elif pd.api.types.is_bool_dtype(validation_results_df[col].dropna()):
                           counts = validation_results_df[col].value_counts()
                           varied_params[param_name] = f"Varied (True: {counts.get(True, 0)}, False: {counts.get(False, 0)})"
                      else:
                           first_few = validation_results_df[col].dropna().unique()[:5]
                           extra = "..." if unique_vals > 5 else ""
                           vals_str = ", ".join(map(str, first_few)) + extra
                           varied_params[param_name] = f"{unique_vals} unique values (e.g., {vals_str})"

        if varied_params:
             prompt += "Key Parameters Varied:\n"
             for name, desc in varied_params.items():
                  prompt += f"- {name}: {desc}\n"
        else:
             prompt += "No significant parameter variations detected across runs (or only 1 run).\n"

        # Summary Table of Key Metrics (Quantum and Classical if present)
        metric_cols = [col for col in validation_results_df.columns if col.startswith(('metric_', 'classical_metric_'))]
        observable_summary_cols = [col for col in validation_results_df.columns if col.startswith('observable_')]
        summary_cols = metric_cols + observable_summary_cols

        if summary_cols:
            prompt += "\nSummary Statistics for Key Metrics/Observables (Successful Runs):\n"
            valid_summary_cols = [col for col in summary_cols if col in validation_results_df.columns]
            if valid_summary_cols:
                numeric_summary_cols = validation_results_df[valid_summary_cols].select_dtypes(include=np.number).columns.tolist()
                if numeric_summary_cols:
                     successful_df = validation_results_df[validation_results_df['success']]
                     if not successful_df.empty:
                          summary_stats = successful_df[numeric_summary_cols].describe().transpose()
                          summary_stats_selected = summary_stats[['count', 'mean', 'std', 'min', 'max']].copy()
                          summary_stats_selected.index.name = "Metric"
                          summary_stats_selected.index = summary_stats_selected.index.str.replace('metric_', '').str.replace('observable_', '').str.replace('classical_', 'Cls_')
                          try:
                              summary_stats_str = summary_stats_selected.to_markdown(floatfmt=".3g")
                              prompt += summary_stats_str + "\n"
                          except ImportError:
                              logger.error("Cannot format summary stats to markdown: Missing 'tabulate' library. Using plain string format.")
                              prompt += summary_stats_selected.to_string(float_format="%.3g") + "\n"
                              prompt += "(Install 'tabulate' for markdown table format)\n"
                          except Exception as e_md:
                              logger.error(f"Failed to convert summary stats to markdown/string: {e_md}")
                              prompt += "(Error formatting summary table)\n"
                     else:
                          prompt += "No successful runs found for summary statistics.\n"
                else:
                     prompt += "No numeric metric/observable columns found for summary statistics.\n"
            else:
                 prompt += "No valid summary columns found in the DataFrame for successful runs.\n"
        else:
            prompt += "No metric/observable columns found in DataFrame.\n"

        # Parametric Sweep Summary
        sweep_runs = validation_results_df[validation_results_df['run_label'].str.contains("Sweep_", na=False)]
        if not sweep_runs.empty:
            sweep_param_name_guess = 'UnknownSweepParam'
            first_label_parts = sweep_runs['run_label'].iloc[0].split('_')
            if len(first_label_parts) > 2 and first_label_parts[0] == "Sweep":
                 sweep_param_name_guess = first_label_parts[1]
            prompt += f"\nParametric Sweep ('{sweep_param_name_guess}') Summary:\n"
            try:
                 sweep_col = 'config_' + sweep_param_name_guess
                 if sweep_col in sweep_runs.columns:
                     cols_to_show = ['run_label', sweep_col]
                     for mc in ['metric_LLE', 'metric_DET', 'metric_TransferEntropy', 'observable_final_entropy', 'classical_metric_LLE_3D']:
                         if mc in sweep_runs.columns: cols_to_show.append(mc)
                     valid_cols_to_show = [c for c in cols_to_show if c in sweep_runs.columns]
                     if len(valid_cols_to_show) > 2:
                         sweep_summary = sweep_runs[sweep_runs['success']].sort_values(by=sweep_col)[valid_cols_to_show]
                         for col in sweep_summary.select_dtypes(include=np.number).columns:
                              sweep_summary[col] = sweep_summary[col].apply(lambda x: _format_metric(x, precision=3))
                         sweep_summary.columns = [c.replace('metric_', '').replace('observable_', '').replace('classical_', 'Cls_').replace('config_', '') for c in sweep_summary.columns]
                         try:
                             prompt += sweep_summary.to_markdown(index=False) + "\n"
                         except ImportError:
                             logger.error("Cannot format sweep summary to markdown: Missing 'tabulate' library. Using plain string format.")
                             prompt += sweep_summary.to_string(index=False) + "\n"
                             prompt += "(Install 'tabulate' for markdown table format)\n"
                     else:
                          prompt += f"- Could not generate detailed sweep summary table (Missing columns like {sweep_col} or metrics).\n"
                 else:
                      prompt += f"- Could not generate sweep summary table (Sweep parameter column '{sweep_col}' not found).\n"
            except Exception as e:
                 prompt += f"- Error generating sweep summary table: {e}\n"

        # Clustering/Embedding Info
        if 'cluster_label' in validation_results_df.columns and validation_results_df['cluster_label'].notna().any():
            prompt += "\nClustering Analysis:\n"
            try:
                 prompt += validation_results_df['cluster_label'].value_counts().to_markdown(header=["Cluster", "Count"]) + "\n"
            except ImportError:
                 logger.error("Cannot format cluster summary to markdown: Missing 'tabulate' library. Using plain string format.")
                 prompt += validation_results_df['cluster_label'].value_counts().to_string(header=["Cluster", "Count"]) + "\n"
                 prompt += "(Install 'tabulate' for markdown table format)\n"
            except Exception as e_cl:
                 prompt += f"(Error formatting cluster summary: {e_cl})\n"
        if 'embedding_X' in validation_results_df.columns and validation_results_df['embedding_X'].notna().any():
            prompt += "\nDimensionality Reduction (e.g., UMAP) was performed.\n"

        # Driver Type Comparison
        driver_col = 'config_driver_type'
        if driver_col in validation_results_df.columns and validation_results_df[driver_col].nunique() > 1:
            prompt += "\nDriver Type Comparison (Mean Metrics for Successful Runs):\n"
            metrics_to_compare_drv = [m for m in ['metric_LLE', 'metric_DET', 'metric_TransferEntropy', 'observable_final_entropy', 'metric_CorrDim'] if m in validation_results_df.columns]
            if metrics_to_compare_drv:
                try:
                    driver_comparison = validation_results_df[validation_results_df['success']].groupby(driver_col)[metrics_to_compare_drv].agg(['mean', 'std'])
                    for col_level0 in driver_comparison.columns.levels[0]:
                        for col_level1 in driver_comparison.columns.levels[1]:
                             driver_comparison[(col_level0, col_level1)] = driver_comparison[(col_level0, col_level1)].apply(lambda x: _format_metric(x, precision=3))
                    driver_comparison.columns = ['_'.join(col).replace('metric_', '').replace('observable_', '') for col in driver_comparison.columns.values]
                    driver_comparison.index.name = "DriverType"
                    try:
                        prompt += driver_comparison.to_markdown() + "\n"
                    except ImportError:
                        logger.error("Cannot format driver comparison to markdown: Missing 'tabulate' library.")
                        prompt += driver_comparison.to_string() + "\n"
                        prompt += "(Install 'tabulate' for markdown table format)\n"
                except Exception as e_drv:
                    prompt += f"(Error generating driver comparison table: {e_drv})\n"
            else:
                prompt += "- No common metrics found to compare across driver types.\n"

        # Scheduling/Gating Summary
        gating_runs = validation_results_df[validation_results_df['run_label'].str.contains("Gating_", na=False)]
        embedding_runs = validation_results_df[validation_results_df['run_label'].str.contains("Embed_", na=False)]
        if not gating_runs.empty or not embedding_runs.empty:
             prompt += "\nParameter Scheduling (Gating/Embedding) Runs Performed:\n"
             relevant_labels = []
             if not gating_runs.empty: relevant_labels.extend(gating_runs['run_label'].unique())
             if not embedding_runs.empty: relevant_labels.extend(embedding_runs['run_label'].unique())

             metrics_to_compare_sched = [m for m in ['metric_LLE', 'metric_DET', 'observable_final_entropy', 'metric_TransferEntropy'] if m in validation_results_df.columns]
             if metrics_to_compare_sched and relevant_labels:
                 scheduling_summary = validation_results_df[validation_results_df['run_label'].isin(relevant_labels)][['run_label'] + metrics_to_compare_sched].copy()
                 if not scheduling_summary.empty:
                     for col in scheduling_summary.select_dtypes(include=np.number).columns:
                          scheduling_summary[col] = scheduling_summary[col].apply(lambda x: _format_metric(x, precision=3))
                     scheduling_summary.columns = [c.replace('metric_', '').replace('observable_', '') for c in scheduling_summary.columns]
                     prompt += "Final Metrics for Scheduled Runs:\n"
                     try:
                          prompt += scheduling_summary.to_markdown(index=False) + "\n"
                     except ImportError:
                          logger.error("Cannot format scheduling summary to markdown: Missing 'tabulate'.")
                          prompt += scheduling_summary.to_string(index=False) + "\n"
                          prompt += "(Install 'tabulate' for markdown table format)\n"
                 else:
                     prompt += "- Could not extract metrics for scheduled runs.\n"
             else:
                  prompt += "- No metrics found to compare for scheduled runs.\n"

    # Repeatability Summary
    if repeatability_summary and isinstance(repeatability_summary, dict):
        prompt += "\n--- Repeatability Analysis Summary ---\n"
        notes = repeatability_summary.get('notes')
        if notes: prompt += f"Notes: {notes}\n"

        stats = repeatability_summary.get('metrics_stats')
        if stats:
            prompt += "Metric Statistics Across Repeats:\n"
            try:
                 stats_df = pd.DataFrame(stats).transpose()
                 for col in stats_df.select_dtypes(include=np.number).columns:
                      stats_df[col] = stats_df[col].apply(lambda x: _format_metric(x, precision=4))
                 stats_df.index = stats_df.index.str.replace('metric_', '').str.replace('observable_', '').str.replace('classical_', 'Cls_')
                 prompt += stats_df[['mean', 'std', 'min', 'max']].to_markdown() + "\n"
            except ImportError:
                 logger.error("Cannot format repeatability stats to markdown: Missing 'tabulate' library.")
                 stats_df = pd.DataFrame(stats).transpose()
                 for col in stats_df.select_dtypes(include=np.number).columns: stats_df[col] = stats_df[col].apply(lambda x: _format_metric(x, precision=4))
                 stats_df.index = stats_df.index.str.replace('metric_', '').str.replace('observable_', '').str.replace('classical_', 'Cls_')
                 prompt += stats_df[['mean', 'std', 'min', 'max']].to_string() + "\n"
                 prompt += "(Install 'tabulate' for markdown table format)\n"
            except Exception as e_rpt:
                 prompt += f"(Error formatting repeatability stats: {e_rpt})\n"
        else:
            prompt += "No metric statistics available for repeatability runs.\n"

        l2_mean = repeatability_summary.get('final_state_l2_mean')
        l2_std = repeatability_summary.get('final_state_l2_std')
        num_dists = repeatability_summary.get('num_l2_distances_calculated', 0)
        if pd.notna(l2_mean):
             prompt += f"Final State L2 Distance (vs baseline, N={num_dists} comparisons):\n"
             prompt += f"- Mean: {_format_metric(l2_mean, precision=4)}\n"
             prompt += f"- Std Dev: {_format_metric(l2_std, precision=4)}\n"
        elif "L2 analysis failed" not in (notes or ""): # Don't duplicate failure message
             prompt += f"Final State L2 Distance: Not calculated or failed.\n"

        if 'metrics_plot' in repeatability_summary and repeatability_summary['metrics_plot']:
            prompt += f"(See metrics distribution plot: {repeatability_summary['metrics_plot']})\n"

    # Sensitivity Summary
    if sensitivity_jacobian is not None and not sensitivity_jacobian.empty:
        prompt += "\n--- Sensitivity Analysis Summary (Jacobian dMetric/dParam) ---\n"
        try:
            jacobian_display = sensitivity_jacobian.copy()
            jacobian_display.index = jacobian_display.index.str.replace('metric_', '').str.replace('observable_', '').str.replace('classical_', 'Cls_')
            prompt += jacobian_display.to_markdown(floatfmt=".3e") + "\n"
        except ImportError:
             logger.error("Cannot format Jacobian to markdown: Missing 'tabulate' library.")
             prompt += sensitivity_jacobian.to_string(float_format="%.3e") + "\n"
             prompt += "(Install 'tabulate' for markdown table format)\n"
        except Exception as e_jac_md:
            prompt += f"(Error formatting sensitivity Jacobian: {e_jac_md})\n"
        prompt += "(See plot: sensitivity_jacobian_heatmap.png)\n"

    # Key Quantum Comparisons (External L2 Diffs - Placeholder)
    prompt += "\n--- Key Quantum Comparisons (Specific Runs - External L2 Diffs) ---\n"
    if comparison_metrics and any(k.startswith('diff_') for k in comparison_metrics):
        key_activation = "1_BaseDriver_Activation"
        key_control = "2_Ablation_Control"
        key_sens_driver = "3_Sensitivity_DriverIC"
        key_sens_qic = "4_Sensitivity_QuantumIC"
        def log_diff(metric_key, label1, label2):
            val = comparison_metrics.get(metric_key)
            l1_short = label1.split('_', 1)[1] if '_' in label1 else label1
            l2_short = label2.split('_', 1)[1] if '_' in label2 else label2
            formatted_val = _format_metric(val)
            return f"- {l1_short} vs {l2_short}: {formatted_val}\n" if formatted_val != "N/A" else ""
        prompt += log_diff('diff_activation_vs_control_L2', key_activation, key_control)
        prompt += log_diff('diff_activation_vs_driverIC_sensitivity_L2', key_activation, key_sens_driver)
        prompt += log_diff('diff_activation_vs_qic_sensitivity_L2', key_activation, key_sens_qic)
        if not any(log_diff(k, "", "").strip() for k in comparison_metrics if k.startswith('diff_')):
            prompt += "- Placeholder L2 difference metrics were NaN or not provided.\n"
    else:
        prompt += "- No specific L2 difference metrics provided externally or calculated.\n"

    # Classical Simulation Summary
    classical_config_col = 'classical_config_model_type' # Use the actual column name from DataFrame
    classical_run_exists = False
    if validation_results_df is not None and classical_config_col in validation_results_df.columns and validation_results_df[classical_config_col].notna().any():
        classical_run_exists = True # Flag that classical sim was attempted and results are in DF

    if classical_results and isinstance(classical_results, dict) and classical_results.get('metrics'):
        prompt += "\n--- Classical Simulation Results (Representative Run) ---\n"
        class_config = classical_results.get('config', {})
        class_metrics = classical_results.get('metrics', {})
        prompt += f"- Model Type: {class_config.get('model_type', 'N/A')}\n"
        prompt += f"- Sim Time (T): {_format_metric(class_config.get('T'))}, Sim Step (dt): {_format_metric(class_config.get('dt'))}\n"
        ic_state_val = class_config.get('initial_state', 'N/A')
        if isinstance(ic_state_val, (list, np.ndarray)):
            ic_state_str = np.array2string(np.array(ic_state_val), precision=3, separator=', ')
        else:
            ic_state_str = str(ic_state_val)
        prompt += f"- Initial State: {ic_state_str}\n"
        prompt += f"- Driver Coupling: {class_config.get('enable_driver_coupling', False)}\n"
        prompt += f"- LLE (Full Dim, Comp 0): {_format_metric(class_metrics.get('LLE_FullDim'))}\n"
        prompt += f"- LLE (3D Proj, Comp 0): {_format_metric(class_metrics.get('LLE_3D'))}\n"
        prompt += f"- CorrDim (Full Dim): {_format_metric(class_metrics.get('CorrDim_FullDim'))}\n"
        prompt += f"- CorrDim (3D Proj): {_format_metric(class_metrics.get('CorrDim_3D'))}\n"
    elif classical_run_exists:
         prompt += "\n--- Classical Simulation Results ---\n"
         prompt += "- Classical simulation was performed, but representative results summary unavailable or metrics missing.\n"
    elif getattr(base_config, 'enable_classical_simulation', False):
         prompt += "\n--- Classical Simulation Results ---\n"
         prompt += "- Classical simulation was enabled in config but failed or produced no results across runs.\n"


    # Quantum Analysis Methods Performed
    prompt += "\n--- Quantum Analysis Methods Performed ---\n"
    analyses_done = set()
    if validation_results_df is not None and not validation_results_df.empty:
        if 'metric_LLE' in validation_results_df.columns and validation_results_df['metric_LLE'].notna().any(): analyses_done.add(f"LLE on '{getattr(base_config, 'observable_for_lle', 'N/A')}'")
        if 'metric_DET' in validation_results_df.columns and validation_results_df['metric_DET'].notna().any(): analyses_done.add(f"RQA on '{getattr(base_config, 'observable_for_rqa', 'N/A')}' (Metrics: DET, LAM, ENTR, etc.)")
        if 'metric_CWT_coeffs_computed' in validation_results_df.columns and validation_results_df['metric_CWT_coeffs_computed'].any(): analyses_done.add(f"CWT on '{getattr(base_config, 'observable_for_cwt', 'N/A')}' (Scalogram plots)")
        if 'metric_CorrDim' in validation_results_df.columns and validation_results_df['metric_CorrDim'].notna().any(): analyses_done.add(f"Correlation Dimension on '{getattr(base_config, 'observable_for_corr_dim', 'N/A')}'")
        if 'metric_TransferEntropy' in validation_results_df.columns and validation_results_df['metric_TransferEntropy'].notna().any(): analyses_done.add(f"Transfer Entropy (Driver -> '{getattr(base_config, 'info_flow_observable_system', 'N/A')}')")
        if 'observable_final_entropy' in validation_results_df.columns and validation_results_df['observable_final_entropy'].notna().any(): analyses_done.add("Spatial Shannon Entropy S(x) (Final Value)")
        if 'config_driver_type' in validation_results_df.columns and validation_results_df['config_driver_type'].nunique() > 1:
             drivers = validation_results_df['config_driver_type'].unique().tolist()
             analyses_done.add(f"Tested Driver Types: {drivers}")
        if ('config_enable_parameter_scheduling' in validation_results_df.columns and validation_results_df['config_enable_parameter_scheduling'].any()) or not gating_runs.empty or not embedding_runs.empty:
             analyses_done.add("Parameter Scheduling (e.g., Gating/Embedding)")

    if repeatability_summary and pd.notna(repeatability_summary.get('final_state_l2_mean')):
        analyses_done.add("Repeatability testing (incl. Final State L2)")
    elif repeatability_summary:
        analyses_done.add("Repeatability testing (Metrics only)")

    if sensitivity_jacobian is not None and not sensitivity_jacobian.empty:
        analyses_done.add("Sensitivity (Jacobian) analysis")

    if analyses_done:
        prompt += "- " + "\n- ".join(sorted(list(analyses_done))) + "\n"
    else:
        prompt += "- No advanced quantum analysis methods detected or summary data provided.\n"

    # Generated Plots
    prompt += "\n--- Generated Plots (Refer to files in results/plots directory) ---\n"
    plot_files = comparison_metrics.get('plot_files', [])
    plot_files_unique = sorted(list(set(filter(None, plot_files))))
    if plot_files_unique:
         prompt += "\n".join(f"- {pf}" for pf in plot_files_unique) + "\n"
         # Add note about manifest if it was generated
         if getattr(base_config, 'generate_plot_manifest', False):
             prompt += "- See 'plot_manifest.md' for more context.\n"
    else:
         prompt += "- Standard plots (Heatmap, Observables) generated per run.\n"
         prompt += "- Meta-analysis plots (Param Maps, Embeddings, Jacobian, Repeatability) may exist.\n"
         prompt += "- Analysis plots (FFT, RQA, CWT, LLE/CD debug) generated per run if enabled.\n"


    # Analysis Task (Updated numbering and content)
    prompt += "\n--- Analysis Task ---\n"
    prompt += "Based *only* on the quantitative results summaries (DataFrame stats, sweep tables, driver comparisons, scheduling summaries, repeatability, sensitivity), parameter ranges, specific comparisons (if provided), and analysis types mentioned above:\n"
    prompt += "1. Briefly summarize the goal (e.g., exploring parameter space, testing sensitivity, comparing drivers, gating/embedding) and the scope (number of runs, parameters varied, scheduling).\n"
    prompt += "2. Describe the key quantitative findings across the parameter space explored (refer to Summary Statistics, varied params, sweep tables, clustering info if available).\n"
    prompt += "3. Analyze Repeatability (if performed): How consistent are key metrics across identical runs (Repeatability Stats)? How similar are the final quantum states (Final State L2 Distance stats)? Is the system's response reliable?\n" # Updated
    prompt += "4. Analyze Sensitivity (if performed): Which metrics are most sensitive to which parameters (Jacobian)? Compare sensitivity to driver parameters (alpha, Lorenz IC) vs. quantum parameters (x0, k0_psi).\n"
    prompt += "5. Analyze Parametric Sweep results (if available): How do metrics change systematically? Is there evidence of transitions? How does TE correlate with LLE/DET?\n"
    prompt += "6. Compare effects of different driver types (if tested): Do average metrics distinguish driver classes (Driver Comparison table)? Do chaotic drivers induce significantly different dynamics than periodic/zero drivers based on metrics?\n" # Updated
    prompt += "7. Analyze Specific Comparisons (if L2 diffs provided): Summarize Activation vs Control vs Sensitivity runs based on the provided L2 differences. How do these compare to the Jacobian?\n" # Updated
    prompt += "8. Analyze Parameter Scheduling (if performed, e.g., Gating/Embedding): Based on the 'Final Metrics for Scheduled Runs' table, do the metrics for scheduled runs differ significantly from constant ON/OFF runs (if applicable)? Does this suggest control potential or indicate lasting effects of transient modulation?\n" # Updated

    classical_point_num = 9
    discussion_point_num = 10 # Default if classical is skipped
    if classical_results and isinstance(classical_results, dict) and classical_results.get('metrics'):
        prompt += f"{classical_point_num}. Analyze the *classical* simulation results: Summarize LLE/CorrDim (Full & 3D). Does the classical model exhibit chaos/hyperchaos? How does coupling affect it (if tested)? Does it support the 3D chaos from 4D determinism idea?\n"
        discussion_point_num = classical_point_num + 1
    elif classical_run_exists:
        prompt += f"[{classical_point_num}. Classical simulation was performed, but representative results summary unavailable or metrics missing.]\n"
        discussion_point_num = classical_point_num + 1
    else:
        prompt += f"[{classical_point_num}. Classical simulation not run or results unavailable.]\n"
        discussion_point_num = classical_point_num # Use classical point num for discussion start

    prompt += f"{discussion_point_num}. Discuss whether the combined quantitative findings provide evidence **for or against** the core hypothesis. Specifically address:\n"
    prompt += "    - Quantum Complexity Source: Is complexity (high LLE, DET, CorrDim, low predictability) primarily driven by chaotic external signals vs. QIC sensitivity? Use driver comparison, sensitivity analysis (Jacobian, L2 diffs), and TE results.\n"
    prompt += "    - System Reliability & Control: How reliable is the system's response (Repeatability: metrics & L2)? Does sensitivity/gating analysis suggest potential for controlling quantum state evolution via external signals?\n" # Updated
    prompt += "    - Quantum-Classical Analogy: How do the quantum complexity metrics (LLE, CorrDim) compare to the classical ones (especially the 3D projection)? Do they behave similarly under equivalent driving (if tested)? Does the classical analogy hold quantitatively?\n"

    next_point = discussion_point_num + 1
    prompt += f"{next_point}. Conclude on the strength of the evidence based *solely* on the provided numerical summaries. Mention specific qualitative analysis (inspecting listed plots like heatmaps, phase space, RPs) needed for confirmation.\n"
    prompt += f"{next_point+1}. Suggest potential applications (e.g., sensing, encoding, quantum control) based on observed phenomena like sensitivity, repeatability, driver fingerprinting potential, or gating effects.\n"
    prompt += f"{next_point+2}. Based *only* on the provided results, do these findings necessitate a fundamentally new theory or framework? Explain, referencing specific results (LLE ranges, metric comparisons Q vs C, TE, sensitivity, repeatability).\n"
    prompt += f"{next_point+3}. Structure the response like a concise 'Results and Discussion' section. Be objective and quantitative where possible.\n"

    return prompt


def call_openai_api(api_key, model, prompt):
    """
    DEPRECATED: Use llm_interface.get_llm_response instead.
    Calls the OpenAI API and returns the response.
    """
    logger.warning("call_openai_api is deprecated. Use llm_interface.get_llm_response(..., provider='openai') instead.")
    if LLM_INTERFACE_AVAILABLE:
        return llm_interface.get_llm_response(prompt, provider='openai', model_name=model, api_key=api_key)
    else:
        logger.error("Cannot call OpenAI API: llm_interface module is unavailable.")
        return "Error: LLM Interface unavailable."


# --- END OF FILE quantum_chaos_sim/reporting.py ---