# --- START OF FILE quantum_chaos_sim/meta_analysis.py ---
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os # For saving plots and loading data

# Import optional dependencies safely
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier # Example for fingerprinting
    from sklearn.model_selection import train_test_split # For classifier testing
    from sklearn.metrics import accuracy_score, classification_report # For classifier testing
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap # umap-learn
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# --- Import analysis functions and common utilities ---
try:
    # Import functions previously defined separately
    from .analysis.gating_analysis import analyze_gating_transients
    from .analysis.embedding_analysis import decode_embedded_signal
    # Import common helpers
    from .analysis.common_analysis_utils import (get_results_dir, get_plot_dir,
                                                 save_or_show_plot as _save_meta_plot, # Use alias
                                                 load_observable_data, find_config_in_df,
                                                 get_config_value, sanitize_filename)
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError as e:
    ANALYSIS_MODULES_AVAILABLE = False
    # logger will be defined below, log the warning there
    _analysis_import_error = e


logger = logging.getLogger(__name__) # Ensure logger is defined

# Log dependency status
if not SKLEARN_AVAILABLE: logger.warning("scikit-learn not found. Clustering, scaling, and fingerprint classification functionality disabled.")
if not UMAP_AVAILABLE: logger.warning("umap-learn not found. UMAP dimensionality reduction disabled.")
if not ANALYSIS_MODULES_AVAILABLE: logger.error(f"Failed to import analysis sub-modules (gating/embedding/common): {_analysis_import_error}. Detailed analysis functions unavailable.")



def cluster_results(results_df, features, method='kmeans', n_clusters=4, prefix='cluster_'):
    """ Performs clustering on the results DataFrame based on specified features. """
    if not SKLEARN_AVAILABLE:
        logger.error("Cannot perform clustering: scikit-learn not available.")
        return results_df # Return original DF
    if results_df is None or results_df.empty:
        logger.warning("Clustering skipped: Input DataFrame is empty or None.")
        return results_df
    valid_features = [f for f in features if f in results_df.columns]
    if not valid_features:
         logger.error("Clustering failed: No valid feature columns provided.")
         results_df[prefix + 'label'] = np.nan
         return results_df
    logger.info(f"Performing {method} clustering (k={n_clusters}) on features: {valid_features}")
    feature_df = results_df[valid_features].copy()
    original_index = feature_df.index
    rows_before_drop = len(feature_df)
    feature_df.dropna(inplace=True)
    rows_after_drop = len(feature_df)
    for col in feature_df.columns: feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df.dropna(inplace=True)
    rows_after_coercion_drop = len(feature_df)
    if rows_after_coercion_drop < n_clusters:
        logger.error(f"Clustering failed: Not enough valid numeric data points ({rows_after_coercion_drop}) for {n_clusters} clusters after dropping NaNs/non-numerics.")
        results_df[prefix + 'label'] = np.nan
        return results_df
    if rows_after_drop < rows_before_drop or rows_after_coercion_drop < rows_after_drop:
        logger.warning(f"Clustering: Dropped {rows_before_drop - rows_after_coercion_drop} rows due to NaNs or non-numeric types in features.")
    scaler = StandardScaler(); scaled_features = scaler.fit_transform(feature_df)
    cluster_labels_on_valid_indices = np.full(rows_after_coercion_drop, np.nan)
    try:
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels_on_valid_indices = kmeans.fit_predict(scaled_features)
        else:
            logger.error(f"Clustering method '{method}' not implemented.")
            results_df[prefix + 'label'] = np.nan; return results_df
        results_df[prefix + 'label'] = np.nan
        results_df.loc[feature_df.index, prefix + 'label'] = cluster_labels_on_valid_indices
        results_df[prefix + 'label'] = pd.Categorical(results_df[prefix + 'label'])
        logger.info("Clustering complete. Added column: " + prefix + 'label')
    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        results_df[prefix + 'label'] = np.nan
    return results_df

def perform_dimensionality_reduction(results_df, features, method='umap', n_components=2, prefix='embedding_', **kwargs):
    """ Performs dimensionality reduction (UMAP) on specified features. """
    if method == 'umap' and not UMAP_AVAILABLE:
         logger.error("Cannot perform UMAP: umap-learn not available."); return results_df
    if results_df is None or results_df.empty:
        logger.warning("Dimensionality reduction skipped: Input DataFrame is empty or None."); return results_df
    valid_features = [f for f in features if f in results_df.columns]
    if not valid_features:
         logger.error("Dim reduction failed: No valid feature columns provided.")
         for i in range(n_components): results_df[f"{prefix}{chr(ord('X')+i)}"] = np.nan
         return results_df
    logger.info(f"Performing {method.upper()} dimensionality reduction (n_components={n_components}) on features: {valid_features}")
    feature_df = results_df[valid_features].copy(); original_index = feature_df.index
    rows_before_drop = len(feature_df); feature_df.dropna(inplace=True); rows_after_drop = len(feature_df)
    for col in feature_df.columns: feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df.dropna(inplace=True); rows_after_coercion_drop = len(feature_df)
    if rows_after_coercion_drop < 5:
        logger.error(f"Dim reduction failed: Not enough valid numeric data points ({rows_after_coercion_drop}) after dropping NaNs/non-numerics.")
        for i in range(n_components): results_df[f"{prefix}{chr(ord('X')+i)}"] = np.nan
        return results_df
    if rows_after_drop < rows_before_drop or rows_after_coercion_drop < rows_after_drop:
        logger.warning(f"Dim Reduction: Dropped {rows_before_drop - rows_after_coercion_drop} rows due to NaNs or non-numeric types in features.")
    scaler = StandardScaler(); scaled_features = scaler.fit_transform(feature_df)
    for i in range(n_components): results_df[f"{prefix}{chr(ord('X')+i)}"] = np.nan
    try:
        if method == 'umap':
            n_neighbors = kwargs.get('n_neighbors', min(15, rows_after_coercion_drop - 1)); n_neighbors = max(2, n_neighbors)
            min_dist = kwargs.get('min_dist', 0.1); metric = kwargs.get('metric', 'euclidean')
            logger.debug(f" UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
            # Setting random_state ensures reproducibility but forces n_jobs=1 (disables parallelism in UMAP).
            # Set random_state=None if UMAP performance is critical and exact reproducibility isn't needed.
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
            embedding = reducer.fit_transform(scaled_features)
        else: logger.error(f"Dimensionality reduction method '{method}' not implemented."); return results_df
        for i in range(n_components): results_df.loc[feature_df.index, f"{prefix}{chr(ord('X')+i)}"] = embedding[:, i]
        logger.info("Dimensionality reduction complete. Added columns: " + ", ".join([f"{prefix}{chr(ord('X')+i)}" for i in range(n_components)]))
    except Exception as e: logger.error(f"Dimensionality reduction failed: {e}", exc_info=True)
    return results_df

def plot_parameter_map(results_df, param_x, param_y, color_metric, cluster_labels=None, filename="param_map.png", config_obj=None):
    """ Creates a scatter plot of runs based on two config parameters. """
    if results_df is None or results_df.empty: logger.warning("Param map plot skipped: DataFrame empty."); return
    required_cols = [param_x, param_y, color_metric]; missing_cols = [col for col in required_cols if col not in results_df.columns]
    if cluster_labels and cluster_labels not in results_df.columns: missing_cols.append(cluster_labels)
    if missing_cols: logger.warning(f"Param map plot skipped: Missing required columns: {missing_cols}."); return
    logger.info(f"Generating parameter map plot: {param_x} vs {param_y}, colored by {color_metric}.")
    plot_df = results_df.copy(); numeric_cols_check = [param_x, param_y]
    if pd.api.types.is_numeric_dtype(plot_df[color_metric]): numeric_cols_check.append(color_metric)
    for col in numeric_cols_check:
        if col in plot_df.columns: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df.dropna(subset=[param_x, param_y, color_metric], inplace=True)
    if plot_df.empty: logger.warning(f"Param map plot skipped: No non-NaN data available for columns {param_x}, {param_y}, {color_metric}."); return
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
         is_color_numeric = pd.api.types.is_numeric_dtype(plot_df[color_metric])
         palette = 'viridis' if is_color_numeric else 'tab10'
         sns.scatterplot(data=plot_df, x=param_x, y=param_y, hue=color_metric, style=cluster_labels if cluster_labels in plot_df.columns else None, palette=palette, size=color_metric if is_color_numeric else None, sizes=(20, 200) if is_color_numeric else None, legend='auto', ax=ax)
         ax.set_xlabel(param_x.replace('config_', '').replace('classical_', '')); ax.set_ylabel(param_y.replace('config_', '').replace('classical_', ''))
         ax.set_title(f'Parameter Map ({param_x.replace("config_","").replace("classical_","")} vs {param_y.replace("config_","").replace("classical_","")})'); ax.grid(True, alpha=0.3)
         if not is_color_numeric and plot_df[color_metric].nunique() > 15: ax.legend().set_visible(False); logger.warning(f"Hiding legend for '{color_metric}' due to too many categories ({plot_df[color_metric].nunique()}).")
         elif is_color_numeric:
              norm = plt.Normalize(plot_df[color_metric].min(), plot_df[color_metric].max()); sm = plt.cm.ScalarMappable(cmap=palette, norm=norm); sm.set_array([])
              fig.colorbar(sm, ax=ax, label=color_metric.replace('metric_', '').replace('observable_', '').replace('classical_', ''))
         plt.tight_layout(); plot_dir = get_plot_dir(results_df, config_obj); _save_meta_plot(fig, filename, plot_dir) # Use imported helper
    except Exception as e: logger.error(f"Failed to generate parameter map plot: {e}", exc_info=True); plt.close(fig)

def plot_embedding(results_df, embedding_x='embedding_X', embedding_y='embedding_Y', color_metric='metric_LLE', cluster_labels=None, filename="embedding_plot.png", config_obj=None):
    """ Plots the 2D embedding (e.g., from UMAP). """
    if results_df is None or results_df.empty: logger.warning("Embedding plot skipped: DataFrame empty."); return
    required_cols = [embedding_x, embedding_y, color_metric]; missing_cols = [col for col in required_cols if col not in results_df.columns]
    if cluster_labels and cluster_labels not in results_df.columns: missing_cols.append(cluster_labels)
    if missing_cols: logger.warning(f"Embedding plot skipped: Missing required columns: {missing_cols}."); return
    logger.info(f"Generating embedding plot: Colored by {color_metric}.")
    plot_df = results_df.copy(); numeric_cols_check = [embedding_x, embedding_y]
    if pd.api.types.is_numeric_dtype(plot_df[color_metric]): numeric_cols_check.append(color_metric)
    for col in numeric_cols_check:
        if col in plot_df.columns: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df.dropna(subset=[embedding_x, embedding_y, color_metric], inplace=True)
    if plot_df.empty: logger.warning(f"Embedding plot skipped: No non-NaN data available for columns {embedding_x}, {embedding_y}, {color_metric}."); return
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
         is_color_numeric = pd.api.types.is_numeric_dtype(plot_df[color_metric]); palette = 'viridis' if is_color_numeric else 'tab10'
         sns.scatterplot(data=plot_df, x=embedding_x, y=embedding_y, hue=color_metric, style=cluster_labels if cluster_labels in plot_df.columns else None, palette=palette, size=color_metric if is_color_numeric else None, sizes=(20, 150) if is_color_numeric else None, legend='auto', ax=ax)
         ax.set_xlabel(embedding_x); ax.set_ylabel(embedding_y)
         ax.set_title(f'2D Embedding ({embedding_x.split("_")[0].upper()}) - Colored by {color_metric.replace("metric_","").replace("config_","").replace("classical_","")}')
         ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.3)
         if not is_color_numeric and plot_df[color_metric].nunique() > 15: ax.legend().set_visible(False); logger.warning(f"Hiding legend for '{color_metric}' due to too many categories ({plot_df[color_metric].nunique()}).")
         elif is_color_numeric:
              norm = plt.Normalize(plot_df[color_metric].min(), plot_df[color_metric].max()); sm = plt.cm.ScalarMappable(cmap=palette, norm=norm); sm.set_array([])
              if ax.get_legend() is not None: ax.get_legend().remove()
              fig.colorbar(sm, ax=ax, label=color_metric.replace('metric_', '').replace('observable_', '').replace('classical_', ''))
         plt.tight_layout(); plot_dir = get_plot_dir(results_df, config_obj); _save_meta_plot(fig, filename, plot_dir) # Use imported helper
    except Exception as e: logger.error(f"Failed to generate embedding plot: {e}", exc_info=True); plt.close(fig)

def analyze_repeatability(results_df, run_label_prefix, metric_cols, full_results_cache, config_obj=None):
    """
    Analyzes the consistency of results for repeated runs, including final state L2 distance.

    Args:
        results_df (pd.DataFrame): DataFrame containing metadata about runs.
        run_label_prefix (str): Prefix identifying the set of repeatability runs.
        metric_cols (list): List of metric column names to analyze statistically.
        full_results_cache (dict): Dictionary containing the full results (including observables)
                                    for all runs, keyed by run_label. Needed for psi_final.
        config_obj (object, optional): Configuration object for plot settings, dx, etc.

    Returns:
        dict: Dictionary containing summary statistics and L2 distance info.
    """
    logger.info(f"Analyzing repeatability for runs starting with '{run_label_prefix}'...")
    summary_stats = {
        "notes": "",
        "metrics_stats": {},
        "metrics_plot": None,
        "final_state_l2_mean": np.nan,
        "final_state_l2_std": np.nan,
        "num_l2_distances_calculated": 0
    }

    if full_results_cache is None:
         logger.error("Repeatability L2 analysis failed: full_results_cache was not provided.")
         summary_stats["notes"] = "L2 analysis failed (missing results cache)."
         # Continue with metric analysis if possible
    if results_df is None:
         logger.error("Repeatability analysis failed: results_df is None.")
         summary_stats["notes"] += " DataFrame missing."
         return summary_stats


    repeat_df = results_df[results_df['run_label'].str.startswith(run_label_prefix, na=False)].copy()
    if repeat_df.empty:
        logger.warning(f"No runs found with prefix '{run_label_prefix}'. Skipping repeatability analysis.")
        summary_stats["notes"] = f"No runs found with prefix '{run_label_prefix}'."
        return summary_stats

    num_repeats = len(repeat_df)
    successful_repeats_df = repeat_df[repeat_df['success']].copy()
    num_successful = len(successful_repeats_df)
    logger.info(f"Found {num_repeats} runs ({num_successful} successful).")

    if num_successful < 2:
        logger.warning(f"Need at least 2 successful runs for repeatability analysis. Found {num_successful}.")
        summary_stats["notes"] += f" Insufficient successful runs ({num_successful}) for detailed analysis."
        # Still return metric stats if num_successful == 1? Maybe not useful.
        return summary_stats # Exit early if less than 2 successful

    # --- Metric Analysis (Existing Logic) ---
    valid_metric_cols = [col for col in metric_cols if col in successful_repeats_df.columns]
    numeric_metric_cols = successful_repeats_df[valid_metric_cols].select_dtypes(include=np.number).columns.tolist()
    if numeric_metric_cols:
        stats = successful_repeats_df[numeric_metric_cols].agg(['mean', 'std', 'min', 'max'])
        summary_stats['metrics_stats'] = stats.to_dict()
        logger.info("Metric Statistics Across Repeats:\n"+stats.to_string(float_format="%.4g"))
        num_metrics = len(numeric_metric_cols)
        if num_metrics > 0:
             ncols = min(3, num_metrics); nrows = (num_metrics + ncols - 1) // ncols
             fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
             axes = axes.flatten(); plot_count = 0
             for i, col in enumerate(numeric_metric_cols):
                  if successful_repeats_df[col].notna().any():
                      try:
                          sns.histplot(successful_repeats_df[col].dropna(), kde=True, ax=axes[plot_count])
                          axes[plot_count].set_title(f'{col.replace("metric_", "").replace("classical_", "").replace("observable_", "")}') # Clean title
                          axes[plot_count].set_xlabel("Value"); plot_count += 1
                      except Exception as e:
                          logger.error(f"Failed to plot histogram for metric '{col}': {e}", exc_info=True)
                          axes[i].set_title(f'{col.replace("metric_", "").replace("classical_", "").replace("observable_", "")} (Plot Error)')
             for j in range(plot_count, len(axes)): fig.delaxes(axes[j]) # Remove unused axes
             if plot_count > 0:
                  fig.suptitle(f"Repeatability Metrics Distribution ({run_label_prefix})", fontsize=16)
                  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                  plot_dir = get_plot_dir(results_df, config_obj) # Use imported helper
                  plot_filename = f"repeatability_{run_label_prefix}_metrics_hist.png"
                  saved_path = _save_meta_plot(fig, os.path.join(plot_dir, plot_filename), True) # Use imported helper
                  if saved_path: summary_stats['metrics_plot'] = os.path.basename(saved_path) # Store only filename
             else:
                  logger.warning("No valid metrics with data found to plot histograms for repeatability.")
                  plt.close(fig) # Close the empty figure
    else:
        logger.warning("No valid numeric metric columns found for repeatability analysis.")
        summary_stats['metrics_stats'] = {}

    # --- L2 Distance Calculation ---
    logger.info("Attempting final state L2 distance calculation...")
    l2_distances = []
    baseline_label = None
    baseline_psi = None
    dx = None

    # Find the first successful run to use as baseline
    for index, row in successful_repeats_df.iterrows():
        label = row['run_label']
        if label in full_results_cache:
             run_data = full_results_cache[label]
             if run_data and run_data.get('success'):
                 psi_final_key = f'psi_final_{label}'
                 psi = run_data.get('observables', {}).get(psi_final_key)
                 run_config = run_data.get('config') # Get config from cache
                 if psi is not None and isinstance(psi, np.ndarray) and run_config:
                      dx_run = get_config_value(run_config, 'dx', None)
                      if dx_run is not None:
                           baseline_label = label
                           baseline_psi = psi
                           dx = dx_run
                           logger.info(f"Using run '{baseline_label}' as baseline for L2 distance.")
                           break # Found baseline
                      else:
                           logger.warning(f"Could not get dx from config for potential baseline run '{label}'.")
                 else:
                      logger.debug(f"Final state or config missing for potential baseline run '{label}'.")
        else:
             logger.warning(f"Run '{label}' not found in full_results_cache.")

    if baseline_psi is None or dx is None:
         logger.error("Could not find a suitable baseline run with final state and dx for L2 calculation.")
         summary_stats["notes"] += " L2 analysis failed (no baseline found)."
         return summary_stats

    # Compare other successful runs to the baseline
    for index, row in successful_repeats_df.iterrows():
        label = row['run_label']
        if label == baseline_label: continue # Skip comparing baseline to itself

        if label in full_results_cache:
            run_data = full_results_cache[label]
            if run_data and run_data.get('success'):
                 psi_final_key = f'psi_final_{label}'
                 psi_comp = run_data.get('observables', {}).get(psi_final_key)
                 if psi_comp is not None and isinstance(psi_comp, np.ndarray) and psi_comp.shape == baseline_psi.shape:
                     diff = psi_comp - baseline_psi
                     # Calculate L2 norm: ||psi_comp - psi_baseline|| * sqrt(dx)
                     l2_dist = np.linalg.norm(diff) * np.sqrt(dx)
                     if np.isfinite(l2_dist):
                         l2_distances.append(l2_dist)
                         logger.debug(f" L2 distance between '{label}' and '{baseline_label}': {l2_dist:.4e}")
                     else:
                         logger.warning(f"L2 distance calculation resulted in non-finite value for run '{label}'. Skipping.")
                 else:
                      logger.warning(f"Final state missing, invalid, or shape mismatch for comparison run '{label}'. Skipping L2.")
            # else: run failed, already filtered out
        else:
             logger.warning(f"Comparison run '{label}' not found in full_results_cache. Skipping L2.")


    # Calculate mean and std dev of L2 distances
    num_distances = len(l2_distances)
    summary_stats["num_l2_distances_calculated"] = num_distances
    if num_distances > 0:
        summary_stats['final_state_l2_mean'] = np.mean(l2_distances)
        summary_stats['final_state_l2_std'] = np.std(l2_distances) if num_distances > 1 else 0.0 # Std dev is 0 if only one distance
        logger.info(f"Final State L2 Distances (vs '{baseline_label}', N={num_distances}): Mean={summary_stats['final_state_l2_mean']:.4e}, Std={summary_stats['final_state_l2_std']:.4e}")
    else:
         logger.warning(f"Could not calculate any L2 distances (needed at least 1 successful run besides baseline).")
         summary_stats["notes"] += " L2 distance calculation yielded no results."

    return summary_stats

def calculate_sensitivity_jacobian(results_df, base_run_label="Sensitivity_Base", params_perturbed=None, config_obj=None):
    """ Calculates a numerical Jacobian (sensitivity) using finite differences. """
    logger.info(f"Calculating sensitivity Jacobian based on run '{base_run_label}'...")
    base_run = results_df[results_df['run_label'] == base_run_label]
    if base_run.empty: logger.error(f"Sensitivity base run '{base_run_label}' not found."); return pd.DataFrame()
    if not base_run['success'].iloc[0]: logger.error(f"Sensitivity base run '{base_run_label}' was not successful."); return pd.DataFrame()
    base_run_metrics = {col: val for col, val in base_run.iloc[0].items() if col.startswith(('metric_', 'classical_metric_', 'observable_')) and pd.api.types.is_number(val)}
    if params_perturbed is None:
        pert_labels = results_df[results_df['run_label'].str.contains("_Pert_Pos", na=False)]['run_label']
        params_perturbed = sorted(list(set([label.split('_')[1] for label in pert_labels])))
        if not params_perturbed: logger.warning("Could not infer perturbed parameters for Jacobian."); return pd.DataFrame()
        logger.info(f"Inferred perturbed parameters for Jacobian: {params_perturbed}")
    metric_names = sorted([m for m in base_run_metrics.keys()])
    if not metric_names: logger.warning("No numeric metric columns found in the base run."); return pd.DataFrame()
    jacobian = pd.DataFrame(index=metric_names, columns=params_perturbed, dtype=float)
    for param in params_perturbed:
        label_pos = f"Sensitivity_{param}_Pert_Pos"; label_neg = f"Sensitivity_{param}_Pert_Neg"
        run_pos = results_df[results_df['run_label'] == label_pos]; run_neg = results_df[results_df['run_label'] == label_neg]
        if run_pos.empty or run_neg.empty: logger.warning(f"Jacobian: Missing perturbation run for '{param}'. Skipping."); jacobian[param] = np.nan; continue
        if not run_pos['success'].iloc[0] or not run_neg['success'].iloc[0]: logger.warning(f"Jacobian: Perturbation run for '{param}' unsuccessful. Skipping."); jacobian[param] = np.nan; continue
        config_param_name = None
        if param == 'lorenz_initial_state': # Special handling for array parameters
             # Find the first component column that exists and differs
             for i in range(3):
                  col_name = f'config_{param}_{i}'
                  if col_name in base_run.columns and col_name in run_pos.columns and col_name in run_neg.columns:
                       base_val = base_run[col_name].iloc[0]; pos_val = run_pos[col_name].iloc[0]; neg_val = run_neg[col_name].iloc[0]
                       if pd.api.types.is_number(base_val) and pd.api.types.is_number(pos_val) and pd.api.types.is_number(neg_val) and not np.isclose(pos_val - neg_val, 0):
                            config_param_name = col_name
                            logger.debug(f"Jacobian: Using component '{config_param_name}' for Lorenz IC diff.")
                            break
             if config_param_name is None: logger.warning(f"Jacobian: Could not find perturbed component column for '{param}'. Skipping."); jacobian[param] = np.nan; continue
        else: config_param_name = 'config_' + param
        if config_param_name not in run_pos.columns or config_param_name not in run_neg.columns: logger.warning(f"Jacobian: Config param '{config_param_name}' missing for '{param}'. Skipping."); jacobian[param] = np.nan; continue
        param_val_pos = run_pos[config_param_name].iloc[0]; param_val_neg = run_neg[config_param_name].iloc[0]
        if not (pd.api.types.is_number(param_val_pos) and pd.api.types.is_number(param_val_neg)): logger.warning(f"Jacobian: Non-numeric config param values for '{param}'. Skipping."); jacobian[param] = np.nan; continue
        delta_param = param_val_pos - param_val_neg
        if np.isclose(delta_param, 0): logger.warning(f"Jacobian: Perturbation difference is zero for '{param}'. Skipping."); jacobian[param] = np.nan; continue
        for metric in metric_names:
            metric_val_pos = run_pos[metric].iloc[0]; metric_val_neg = run_neg[metric].iloc[0]
            if pd.isna(metric_val_pos) or pd.isna(metric_val_neg): sensitivity = np.nan
            else: delta_metric = metric_val_pos - metric_val_neg; sensitivity = delta_metric / delta_param
            jacobian.loc[metric, param] = sensitivity
    logger.info("Sensitivity Jacobian calculation complete.")
    if not jacobian.empty:
         logger.info("\nSensitivity Jacobian (dMetric / dParam):\n" + jacobian.to_string(float_format="%.3e"))
         try:
              plt.figure(figsize=(max(8, len(params_perturbed)*0.8), max(6, len(metric_names)*0.5)))
              sns.heatmap(jacobian.astype(float).dropna(axis=0, how='all').dropna(axis=1, how='all'), annot=True, fmt=".2e", cmap="vlag", center=0, linewidths=.5)
              plt.title("Sensitivity Jacobian (dMetric / dParam)"); plt.xlabel("Parameter Perturbed"); plt.ylabel("Metric"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
              plot_dir = get_plot_dir(results_df, config_obj); plot_filename = "sensitivity_jacobian_heatmap.png"; _save_meta_plot(plt.gcf(), plot_filename, plot_dir) # Use imported helper
         except Exception as e: logger.error(f"Failed to generate Jacobian heatmap plot: {e}", exc_info=True); plt.close(plt.gcf())
    else: logger.info("Sensitivity Jacobian is empty.")
    return jacobian

def extract_fingerprint(df_row, feature_cols):
    """ Extracts a numerical feature vector (fingerprint) from a DataFrame row. """
    if df_row is None or df_row.empty: return None
    if not all(col in df_row.index for col in feature_cols): return None
    fingerprint_data = []
    for col in feature_cols:
        val = df_row[col]
        if pd.api.types.is_number(val) and not pd.isna(val): fingerprint_data.append(float(val))
        else: return None
    return np.array(fingerprint_data)

def build_fingerprint_database(results_df, feature_cols, target_col='config_driver_type'):
    """ Builds a database (DataFrame) of fingerprints and their corresponding target labels. """
    logger.info(f"Building fingerprint database using features: {feature_cols} and target: {target_col}")
    required_cols = feature_cols + [target_col, 'run_label']
    if not all(col in results_df.columns for col in required_cols): logger.error(f"Cannot build fingerprint DB: Missing required columns."); return pd.DataFrame()
    fingerprints = []; labels = []; run_labels = []
    for index, row in results_df.iterrows():
        fp = extract_fingerprint(row, feature_cols)
        if fp is not None:
            target = row[target_col]
            if pd.notna(target): fingerprints.append(fp); labels.append(target); run_labels.append(row['run_label'])
    if not fingerprints: logger.warning("No valid fingerprints extracted."); return pd.DataFrame()
    fp_db = pd.DataFrame({'run_label': run_labels, 'fingerprint': fingerprints, 'target_label': labels})
    logger.info(f"Fingerprint database built with {len(fp_db)} entries."); return fp_db

def train_driver_classifier(fingerprint_df, test_size=0.3, n_neighbors=3):
    """ Trains and evaluates a simple k-NN classifier to predict driver type from fingerprints. """
    if not SKLEARN_AVAILABLE: logger.error("Cannot train classifier: scikit-learn not available."); return None, np.nan, None
    if fingerprint_df is None or fingerprint_df.empty or len(fingerprint_df) < 10: logger.warning(f"Cannot train classifier: Insufficient data ({len(fingerprint_df) if fingerprint_df is not None else 0} rows)."); return None, np.nan, None
    logger.info(f"Training k-NN classifier (k={n_neighbors}) on fingerprint data...")
    try:
        X = np.vstack(fingerprint_df['fingerprint'].values); y = fingerprint_df['target_label'].values
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors); knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test); accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Classifier trained. Test Accuracy: {accuracy:.4f}")
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info("Classification Report:\n" + report)
        return knn, accuracy, report
    except Exception as e: logger.error(f"Failed to train or evaluate classifier: {e}", exc_info=True); return None, np.nan, None

def compare_quantum_classical(results_df, config_obj=None, q_metric_map=None, c_metric_map=None):
    """
    Generates plots comparing key quantum and classical metrics for runs where both were performed.
    """
    logger.info("Generating Quantum vs. Classical comparison plots...")
    if results_df is None or results_df.empty: logger.warning("Quantum vs. Classical comparison skipped: DataFrame empty."); return
    if q_metric_map is None: q_metric_map = {'LLE': 'metric_LLE', 'CorrDim': 'metric_CorrDim'}
    if c_metric_map is None: c_metric_map = {'LLE': 'classical_metric_LLE_3D', 'CorrDim': 'classical_metric_CorrDim_3D'}
    common_metrics = sorted(list(set(q_metric_map.keys()) & set(c_metric_map.keys())))
    qc_compare_cols = [q_metric_map[m] for m in common_metrics if q_metric_map[m] in results_df.columns]
    qc_compare_cols += [c_metric_map[m] for m in common_metrics if c_metric_map[m] in results_df.columns]
    if 'config_driver_type' in results_df.columns: qc_compare_cols.append('config_driver_type')
    if 'config_alpha' in results_df.columns: qc_compare_cols.append('config_alpha')
    if 'run_label' in results_df.columns: qc_compare_cols.append('run_label')
    qc_compare_cols = [c for c in qc_compare_cols if c in results_df.columns]
    if len(qc_compare_cols) < 3: logger.warning("Quantum vs. Classical comparison skipped: Insufficient common metrics or identifiers found."); return
    compare_df = results_df[qc_compare_cols].dropna().copy()
    if compare_df.empty: logger.warning("Quantum vs. Classical comparison skipped: No runs found with comparable quantum and classical metrics."); return
    plot_dir = get_plot_dir(results_df, config_obj); num_metrics_to_plot = len(common_metrics) # Use imported helper
    if num_metrics_to_plot == 0: logger.warning("Quantum vs. Classical comparison skipped: No common metrics defined in maps were found in DataFrame."); return

    fig, axes = plt.subplots(1, num_metrics_to_plot, figsize=(7 * num_metrics_to_plot, 6.5), squeeze=False); axes = axes.flatten(); plot_count = 0
    for i, metric_name in enumerate(common_metrics):
        q_col = q_metric_map.get(metric_name); c_col = c_metric_map.get(metric_name)
        if q_col not in compare_df.columns or c_col not in compare_df.columns: logger.warning(f"Skipping Q/C comparison for '{metric_name}': Column missing."); continue
        ax = axes[plot_count]
        try:
            hue_col = 'config_driver_type' if 'config_driver_type' in compare_df.columns else None
            size_col = 'config_alpha' if 'config_alpha' in compare_df.columns else None
            sns.scatterplot(data=compare_df, x=c_col, y=q_col, hue=hue_col, size=size_col, sizes=(20, 150) if size_col else None, alpha=0.7, legend='auto', ax=ax)
            ax.set_xlabel(f"Classical {metric_name} ({c_col.replace('classical_metric_','').replace('_3D','')})"); ax.set_ylabel(f"Quantum {metric_name} ({q_col.replace('metric_','')})")
            ax.set_title(f"Quantum vs Classical {metric_name}"); ax.grid(True, alpha=0.4)
            if hue_col and compare_df[hue_col].nunique() > 10: ax.legend().set_visible(False)
            plot_count += 1
        except Exception as e: logger.error(f"Failed to create Q/C comparison plot for {metric_name}: {e}", exc_info=True); ax.set_title(f"Q vs C {metric_name} (Plot Error)")
    for j in range(plot_count, len(axes)): fig.delaxes(axes[j])
    if plot_count > 0:
        fig.suptitle("Quantum vs. Classical Metric Comparison", fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = "comparison_quantum_vs_classical_metrics.png"; _save_meta_plot(fig, plot_filename, plot_dir) # Use imported helper
    else: logger.warning("No Quantum vs. Classical comparison plots generated."); plt.close(fig)

# Note: analyze_gating_transients and decode_embedded_signal are imported from
# quantum_chaos_sim.analysis.gating_analysis and quantum_chaos_sim.analysis.embedding_analysis
# They can be called directly from main.py after run_suite completes, passing the results_df and base_config.

# --- END OF FILE quantum_chaos_sim/meta_analysis.py ---