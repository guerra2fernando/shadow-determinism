# --- START OF FILE information_flow.py ---

# quantum_chaos_sim/validation/information_flow.py
"""
Calculates information flow metrics like Transfer Entropy using PyInform.
Requires 'pyinform' library: pip install pyinform
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Dependency Check ---
try:
    import pyinform
    from pyinform import transfer_entropy, mutual_info
    PYINFORM_AVAILABLE = True
except ImportError:
    logger.warning("Library 'pyinform' not found. Information flow analysis (Transfer Entropy, Mutual Info) will not be available.")
    logger.warning("Install it using: pip install pyinform")
    PYINFORM_AVAILABLE = False


def discretize_series(series, num_bins, shared_min=None, shared_max=None):
    """
    Discretizes a continuous series into integer bins (0 to num_bins-1).

    Args:
        series (np.ndarray): The 1D continuous time series.
        num_bins (int): The desired number of bins.
        shared_min (float, optional): Minimum value for binning range (if shared across series).
        shared_max (float, optional): Maximum value for binning range (if shared across series).

    Returns:
        np.ndarray: The discretized series with integer values from 0 to num_bins-1.
                    Returns None if input is invalid.
    """
    if series is None or series.ndim != 1 or series.size == 0:
        logger.error("Discretization Error: Invalid input series.")
        return None
    if not np.all(np.isfinite(series)):
        logger.error("Discretization Error: Series contains non-finite values.")
        # Option: Handle by removing/interpolating NaNs, or fail. Let's fail for now.
        return None

    min_val = shared_min if shared_min is not None else np.min(series)
    max_val = shared_max if shared_max is not None else np.max(series)

    # Handle constant series case to avoid zero range
    if np.isclose(min_val, max_val):
        logger.warning("Discretization Warning: Series is constant. Assigning all points to bin 0.")
        return np.zeros_like(series, dtype=int)

    # Create bin edges - linspace creates num_bins+1 edges
    # Add small epsilon to max_val to ensure the max value falls into the last bin
    bins = np.linspace(min_val, max_val + 1e-9, num_bins + 1)

    # Digitize: returns indices 1 to num_bins+1 (values < bins[0] are 0, but shouldn't happen with linspace)
    # We want indices 0 to num_bins-1
    discretized = np.digitize(series, bins) - 1

    # Clip values just in case (e.g., if max_val was exactly on the edge)
    discretized = np.clip(discretized, 0, num_bins - 1)

    return discretized.astype(int)


def calculate_transfer_entropy(source_series, target_series, k, num_bins=16, lag=1, local=False):
    """
    Calculates the Transfer Entropy (TE) from a source to a target time series
    using pyinform, after discretizing the input series.

    Measures the reduction in uncertainty about the target's next state given
    the history of both the target and the source, compared to knowing only
    the target's history. TE(Source -> Target).

    NOTE: pyinform.transfer_entropy implicitly uses lag=1. The 'lag' parameter
          here is kept for potential future extensions but currently ignored.

    Args:
        source_series (np.ndarray): The 1D source time series (continuous or discrete).
        target_series (np.ndarray): The 1D target time series (continuous or discrete).
        k (int): History length to consider for conditioning (k > 0).
        num_bins (int): Number of bins to use for discretizing continuous data.
                        Ignored if data appears already discrete (all integers).
        lag (int): Time lag (currently ignored by pyinform.transfer_entropy, effectively always 1).
        local (bool): If True, calculate the local TE for each time step.
                      If False (default), calculate the average TE over the series.

    Returns:
        float or np.ndarray or None:
            - Average TE (float) if local=False.
            - Array of local TE values (np.ndarray) if local=True.
            - None if pyinform is not available or input validation/discretization fails.
            - np.nan if calculation within pyinform fails.
    """
    if not PYINFORM_AVAILABLE:
        logger.error("Cannot calculate Transfer Entropy: 'pyinform' library not available.")
        return None

    # --- Input Validation ---
    if source_series is None or target_series is None:
        logger.error("TE Error: Source or target series is None.")
        return None
    if not isinstance(source_series, np.ndarray) or not isinstance(target_series, np.ndarray):
        logger.error("TE Error: Source and target series must be NumPy arrays.")
        return None
    if source_series.ndim != 1 or target_series.ndim != 1:
        logger.error("TE Error: Source and target series must be 1-dimensional.")
        return None
    if len(source_series) != len(target_series):
        logger.error(f"TE Error: Source ({len(source_series)}) and target ({len(target_series)}) series must have the same length.")
        return None
    if not isinstance(k, int) or k <= 0:
        logger.error(f"TE Error: History length k ({k}) must be a positive integer.")
        return None
    if not isinstance(lag, int) or lag <= 0:
        logger.error(f"TE Error: Lag ({lag}) must be a positive integer (though ignored by current pyinform call).")
        return None
    if lag != 1:
        logger.warning(f"TE Warning: Requested lag={lag}, but pyinform.transfer_entropy implicitly uses lag=1. The 'lag' parameter is currently ignored.")

    n = len(source_series)
    effective_lag = 1 # PyInform's implicit lag
    if n < k + effective_lag:
        logger.error(f"TE Error: Time series length ({n}) is too short for k={k}. Needs at least {k+effective_lag} points for pyinform.")
        return None
    if not isinstance(num_bins, int) or num_bins <= 1:
        logger.error(f"TE Error: num_bins ({num_bins}) must be an integer greater than 1.")
        return None

    # --- Discretization ---
    # Check if data looks continuous (contains floats or needs scaling)
    # We'll discretize unless BOTH series look like non-negative integers already.
    needs_discretization = False
    if np.issubdtype(source_series.dtype, np.floating) or np.issubdtype(target_series.dtype, np.floating):
        needs_discretization = True
    elif np.any(source_series < 0) or np.any(target_series < 0):
        needs_discretization = True
    # Optional: Add check for max value if pre-discretized data might exceed C limits

    if needs_discretization:
        logger.info(f"Discretizing source and target series into {num_bins} bins...")
        # Determine shared range for consistent binning
        combined_min = min(np.min(source_series), np.min(target_series))
        combined_max = max(np.max(source_series), np.max(target_series))

        source_discrete = discretize_series(source_series, num_bins, combined_min, combined_max)
        target_discrete = discretize_series(target_series, num_bins, combined_min, combined_max)

        if source_discrete is None or target_discrete is None:
            logger.error("TE Error: Discretization failed.")
            return None # Discretization helper logs specific error
        source_input = source_discrete
        target_input = target_discrete
        logger.info("Discretization complete.")
    else:
        # Data appears to be non-negative integers already
        logger.info("Input series appear discrete and non-negative. Using as is.")
        source_input = source_series.astype(int)
        target_input = target_series.astype(int)

    # --- Calculation ---
    logger.info(f"Calculating Transfer Entropy (Source->Target): k={k}, (implicit lag=1), local={local}...")

    try:
        # Call pyinform with the non-negative integer series
        te_value = transfer_entropy(source_input, target_input, k=k, local=local)

        # Check for NaN result
        if isinstance(te_value, np.ndarray) and np.any(np.isnan(te_value)):
             logger.warning(f"TE calculation resulted in NaN values (local={local}).")
        elif not isinstance(te_value, np.ndarray) and np.isnan(te_value):
             logger.warning(f"TE calculation resulted in NaN (local={local}).")
             return np.nan

        logger.info("Transfer Entropy calculation complete.")
        return te_value

    except ValueError as ve:
        logger.error(f"ValueError during Transfer Entropy calculation: {ve}. Check parameters/data/binning.", exc_info=True)
        return None
    except TypeError as te:
         logger.error(f"TypeError during Transfer Entropy calculation: {te}. Check pyinform function signature.", exc_info=True)
         return None
    except pyinform.error.InformError as ie: # Catch specific Inform errors
        logger.error(f"PyInform Error during TE calculation: {ie}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error calculating Transfer Entropy: {e}", exc_info=True)
        return None


# Example: Add Mutual Information calculation if needed later
def calculate_mutual_information(series1, series2, k=1, lag=0, local=False):
    """
    Calculates the Mutual Information (MI) between two time series.
    (Implementation Placeholder - Adapt based on specific MI needs and discretization)
    """
    if not PYINFORM_AVAILABLE:
        logger.error("Cannot calculate Mutual Information: 'pyinform' library not available.")
        return None
    logger.warning("Mutual Information calculation function is a placeholder and may need discretization.")
    # Add validation and discretization similar to TE
    # ...
    try:
        # mi_value = mutual_info(series1_discrete, series2_discrete, k=k, local=local) # Use discretized
        return np.nan # Return NaN until implemented
    except Exception as e:
        logger.error(f"Error calculating Mutual Information: {e}", exc_info=True)
        return None

# --- END OF FILE information_flow.py ---