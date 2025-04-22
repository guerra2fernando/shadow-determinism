# --- START OF FILE classical_mechanics/analysis.py ---

"""
Functions for analyzing the results of classical mechanics simulations,
including chaos quantifiers like LLE and Correlation Dimension.
Uses the standard approach of embedding a single component time series.
"""

import numpy as np
import logging
# Remove direct nolds import here if compute_correlation_dimension handles it
# import nolds
from ..validation.metrics import NOLDS_AVAILABLE, compute_largest_lyapunov_exponent, compute_correlation_dimension

logger = logging.getLogger(__name__)

def analyze_classical_trajectory(t_values, states, config_obj):
    """
    Analyzes a classical trajectory (e.g., from simulate_classical_system).
    Calculates LLE and Correlation Dimension using the first component (index 0)
    of the state vector as a representative time series and applying
    Takens' embedding theorem via the 'nolds' library functions.
    Optionally analyzes the projected 3D subspace (components 0, 1, 2) similarly.

    Args:
        t_values (np.ndarray): Time points.
        states (np.ndarray): State vectors at each time point (shape: N_points, state_dim).
        config_obj: Configuration object containing classical analysis settings.

    Returns:
        dict: A dictionary containing analysis results (metrics).
              Keys like 'LLE_FullDim', 'CorrDim_FullDim', 'LLE_3D', 'CorrDim_3D'.
              Values are np.nan if calculation fails or is disabled.
    """
    analysis_results = {}
    if t_values is None or states is None or len(t_values) < 2 or states.shape[0] < 2:
        logger.warning("Classical analysis skipped: Input trajectory is None or too short.")
        return analysis_results

    if not NOLDS_AVAILABLE:
        logger.warning("Classical analysis skipped: 'nolds' library not available for LLE/CorrDim.")
        return analysis_results

    dt = t_values[1] - t_values[0]
    state_dim = states.shape[1]
    n_points = len(t_values)
    logger.info(f"--- Analyzing Classical Trajectory (Dim={state_dim}, Points={n_points}) ---")

    # --- Revised Default Parameters ---
    # Embedding dimension heuristic (e.g., based on FNN, but use smaller default)
    default_emb_dim = min(7, state_dim + 1) # Cap default embedding dim lower
    # Lag heuristic (e.g., first minimum of mutual info, or fraction of period)
    # Use a much smaller fraction or a fixed small number like 10 or 20.
    # Let's try a fixed default lag initially, as percentage can be problematic.
    default_lag = 10 # Fixed default lag - Often needs tuning!
    # min_tsep: Ensure it's reasonable, maybe based on estimated period or fixed value
    # Avoid lag*(emb_dim-1) if lag can be large. Maybe use lag or a fixed value?
    default_min_tsep = default_lag * 2 # Example: min_tsep = 2 * lag

    lle_emb_dim = getattr(config_obj, 'classical_lle_emb_dim', default_emb_dim)
    lle_lag = getattr(config_obj, 'classical_lle_lag', default_lag)
    # Use the config value for min_tsep if provided, otherwise calculate carefully
    lle_min_tsep = getattr(config_obj, 'classical_lle_min_tsep', None)
    if lle_min_tsep is None:
         # Use a heuristic: max of default or lag * (emb_dim - 1), but capped
         calc_tsep = lle_lag * (lle_emb_dim - 1) if lle_emb_dim > 1 else lle_lag
         lle_min_tsep = max(default_min_tsep, calc_tsep)
         # Cap min_tsep to avoid excessive values, e.g., fraction of data length
         lle_min_tsep = min(lle_min_tsep, int(n_points * 0.1)) # Cap at 10% of data length
         lle_min_tsep = max(1, lle_min_tsep) # Ensure it's at least 1
         logger.debug(f"Using calculated lle_min_tsep = {lle_min_tsep}")
    else:
         logger.debug(f"Using lle_min_tsep from config = {lle_min_tsep}")


    cd_emb_dim = getattr(config_obj, 'classical_cd_emb_dim', lle_emb_dim) # Default CD emb_dim to LLE emb_dim
    cd_lag = getattr(config_obj, 'classical_cd_lag', lle_lag) # Default CD lag to LLE lag
    fit_method = getattr(config_obj, 'classical_fit_method', 'RANSAC') # RANSAC needs sklearn

    # --- Analyze Full State Space (using Component 0 for embedding) ---
    if state_dim >= 1 and n_points > 100: # Need enough points for analysis
        logger.info("Analyzing full state space using Component 0...")
        representative_series_full = states[:, 0]

        # Calculate LLE for Component 0
        lle_full = compute_largest_lyapunov_exponent(
            representative_series_full, dt=dt, emb_dim=lle_emb_dim, lag=lle_lag,
            min_tsep=lle_min_tsep, # Pass the calculated/config min_tsep
            fit_method=fit_method
        )
        analysis_results['LLE_FullDim'] = lle_full
        analysis_results['LLE_FullDim_Emb'] = lle_emb_dim
        analysis_results['LLE_FullDim_Lag'] = lle_lag
        analysis_results['LLE_FullDim_MinTsep'] = lle_min_tsep # Store for reference
        if not np.isnan(lle_full):
            logger.info(f"LLE (Full Dim, Component 0, Emb={lle_emb_dim}, Lag={lle_lag}, TSep={lle_min_tsep}): {lle_full:.4e}")
        else:
            logger.warning("LLE calculation failed for Full Dim (Component 0).")


        # Calculate Correlation Dimension for Component 0
        cd_full = compute_correlation_dimension(
             representative_series_full, emb_dim=cd_emb_dim, lag=cd_lag, fit=fit_method
        )
        analysis_results['CorrDim_FullDim'] = cd_full
        analysis_results['CorrDim_FullDim_Emb'] = cd_emb_dim
        analysis_results['CorrDim_FullDim_Lag'] = cd_lag
        if not np.isnan(cd_full):
            logger.info(f"Correlation Dimension (Full Dim, Component 0, Emb={cd_emb_dim}, Lag={cd_lag}): {cd_full:.4f}")
        else:
            logger.warning("Correlation Dimension calculation failed for Full Dim (Component 0).")

    else:
         logger.info("Skipping full dimension analysis (state_dim < 1 or insufficient points).")


    # --- Analyze Projected 3D Subspace (Components 0, 1, 2) using Component 0 ---
    if getattr(config_obj, 'classical_analyze_3d_subspace', False) and state_dim >= 3 and n_points > 100:
        logger.info("Analyzing projected 3D subspace using Component 0...")
        representative_series_3d = states[:, 0] # Same as representative_series_full

        # Use potentially different/smaller embedding for 3D projection analysis
        lle_emb_dim_3d = min(lle_emb_dim, 5) # Cap 3D embedding dim lower
        cd_emb_dim_3d = min(cd_emb_dim, 5)
        # Recalculate min_tsep for 3D parameters
        calc_tsep_3d = lle_lag * (lle_emb_dim_3d - 1) if lle_emb_dim_3d > 1 else lle_lag
        lle_min_tsep_3d = max(default_min_tsep, calc_tsep_3d)
        lle_min_tsep_3d = min(lle_min_tsep_3d, int(n_points * 0.1))
        lle_min_tsep_3d = max(1, lle_min_tsep_3d)
        logger.debug(f"Using calculated lle_min_tsep_3d = {lle_min_tsep_3d}")


        # Calculate LLE for the 3D projection (using Component 0)
        lle_3d = compute_largest_lyapunov_exponent(
            representative_series_3d, dt=dt, emb_dim=lle_emb_dim_3d, lag=lle_lag,
            min_tsep=lle_min_tsep_3d, # Use 3D specific min_tsep
            fit_method=fit_method
        )
        analysis_results['LLE_3D'] = lle_3d
        analysis_results['LLE_3D_Emb'] = lle_emb_dim_3d
        analysis_results['LLE_3D_Lag'] = lle_lag
        analysis_results['LLE_3D_MinTsep'] = lle_min_tsep_3d # Store for reference
        if not np.isnan(lle_3d):
            logger.info(f"LLE (3D Proj, Component 0, Emb={lle_emb_dim_3d}, Lag={lle_lag}, TSep={lle_min_tsep_3d}): {lle_3d:.4e}")
        else:
            logger.warning("LLE calculation failed for 3D Projection (Component 0).")

        # Calculate Correlation Dimension for the 3D projection (using Component 0)
        cd_3d = compute_correlation_dimension(
            representative_series_3d, emb_dim=cd_emb_dim_3d, lag=cd_lag, fit=fit_method
        )
        analysis_results['CorrDim_3D'] = cd_3d
        analysis_results['CorrDim_3D_Emb'] = cd_emb_dim_3d
        analysis_results['CorrDim_3D_Lag'] = cd_lag
        if not np.isnan(cd_3d):
            logger.info(f"Correlation Dimension (3D Proj, Component 0, Emb={cd_emb_dim_3d}, Lag={cd_lag}): {cd_3d:.4f}")
        else:
             logger.warning("Correlation Dimension calculation failed for 3D Projection (Component 0).")

    else:
        logger.info("Skipping 3D subspace analysis (disabled in config, state_dim < 3, or insufficient points).")

    logger.info("--- Classical Trajectory Analysis Complete ---")
    return analysis_results

# --- END OF FILE classical_mechanics/analysis.py ---