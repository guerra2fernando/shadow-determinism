# --- START OF FILE metrics.py ---

# quantum_chaos_sim/validation/metrics.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Dependency Checks ---

# nolds (for LLE, Correlation Dimension)
try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    logger.warning("Library 'nolds' not found. Lyapunov exponent (LLE) and Correlation Dimension calculation will not be available.")
    logger.warning("Install it using: pip install nolds")
    NOLDS_AVAILABLE = False

# pyRQA (for RQA) - Replaces pyunicorn
_RQA_IMPORT_ERROR = None # Store potential import error
try:
    # Imports based on pyRQA documentation examples
    from pyrqa.settings import Settings
    from pyrqa.computation import RQAComputation, RPComputation # Need RPComputation to get the matrix easily
    from pyrqa.time_series import TimeSeries # For data handling
    from pyrqa.neighbourhood import FixedRadius # Main one needed for our logic
    from pyrqa.metric import EuclideanMetric, MaximumMetric, TaxicabMetric # For mapping
    from pyrqa.analysis_type import Classic # Default, but good to have accessible

    RQA_AVAILABLE = True
    # Perform a more robust check
    try:
        _test_ts = TimeSeries([1,2,3], embedding_dimension=1, time_delay=1)
        _test_settings = Settings(_test_ts, neighbourhood=FixedRadius(0.1), similarity_measure=EuclideanMetric)
        # Check if computation can be created (doesn't require OpenCL init usually)
        _test_comp = RQAComputation.create(_test_settings, verbose=False)
        logger.debug("pyRQA imports and basic object creation successful.")
    except Exception as e_test: # Catch potential errors during test instantiation/creation
         logger.warning(f"Library 'pyrqa' found, but failed basic test: {e_test}")
         logger.warning("This might indicate issues with OpenCL setup or other dependencies.")
         RQA_AVAILABLE = False # Mark as unavailable if basic setup fails
         _RQA_IMPORT_ERROR = e_test # Store the error

except ImportError as e:
    logger.warning("Library 'pyrqa' or one of its core components not found. Recurrence Quantification Analysis (RQA) will not be available.")
    logger.warning(f"Import Error: {e}") # Log the specific import error
    logger.warning("Install it using: pip install pyRQA")
    RQA_AVAILABLE = False
    _RQA_IMPORT_ERROR = e # Store the error


# PyWavelets (for CWT)
try:
    import pywt
    CWT_AVAILABLE = True
    _USE_SCIPY_CWT = False # Prefer PyWavelets
except ImportError:
    # Fallback check for scipy.signal.cwt (less common, but possible)
    try:
        from scipy import signal
        if hasattr(signal, 'cwt'):
            logger.info("Using scipy.signal.cwt for Wavelet Analysis (PyWavelets not found).")
            CWT_AVAILABLE = True
            _USE_SCIPY_CWT = True
        else:
             raise ImportError # Scipy installed but no CWT function
    except ImportError:
        logger.warning("Library 'PyWavelets' not found, and scipy.signal.cwt is unavailable. Continuous Wavelet Transform (CWT) analysis will not be available.")
        logger.warning("Install PyWavelets using: pip install PyWavelets")
        CWT_AVAILABLE = False
        _USE_SCIPY_CWT = False

# --- Existing Metrics Functions ---

def compute_largest_lyapunov_exponent(time_series, dt,
                                       emb_dim=None, lag=None,
                                       min_tsep=None, tau=1,
                                       fit_method='RANSAC',
                                       default_emb_dim=5, default_lag_factor=0.1,
                                       debug_plot=False, plot_file=None):
    """
    Computes the largest Lyapunov exponent (LLE) from a time series using 'nolds'.
    Requires the 'nolds' library to be installed (`pip install nolds`).
    'RANSAC' fit method also requires `pip install scikit-learn`.
    """
    if not NOLDS_AVAILABLE:
        logger.error("Cannot compute Lyapunov exponent: 'nolds' library not installed.")
        return np.nan

    if time_series is None or len(time_series) < 100:
        logger.warning(f"Time series too short ({len(time_series) if time_series is not None else 0} points) for reliable LLE calculation.")
        return np.nan

    estimated_lag = False
    if lag is None:
        lag = max(1, int(default_lag_factor * len(time_series)))
        logger.warning(f"Lyapunov 'lag' not provided. Using heuristic value: {lag}. Result may be inaccurate.")
        estimated_lag = True

    estimated_emb_dim = False
    if emb_dim is None:
        emb_dim = default_emb_dim
        logger.warning(f"Lyapunov 'emb_dim' not provided. Using default value: {emb_dim}. Result may be inaccurate.")
        estimated_emb_dim = True

    if min_tsep is None:
        min_tsep = lag * (emb_dim - 1) if emb_dim > 1 else lag
        min_tsep = max(1, min_tsep)
        logger.debug(f"Using min_tsep = {min_tsep} (based on lag * (emb_dim - 1))")

    traj_len = max(20, int(len(time_series) * 0.1))

    logger.info(f"Computing LLE with: emb_dim={emb_dim}, lag={lag}, min_tsep={min_tsep}, tau={tau}, fit='{fit_method}', traj_len={traj_len}")

    try:
        data = np.asarray(time_series, dtype=float)

        if not isinstance(emb_dim, int) or emb_dim < 1:
             logger.error(f"LLE Error: emb_dim must be a positive integer, got {emb_dim}")
             return np.nan
        if not isinstance(lag, int) or lag < 1:
             logger.error(f"LLE Error: lag must be a positive integer, got {lag}")
             return np.nan
        if not isinstance(min_tsep, int) or min_tsep < 1:
              logger.error(f"LLE Error: min_tsep must be a positive integer, got {min_tsep}")
              return np.nan
        if not isinstance(tau, (int, float)) or tau <= 0:
               logger.error(f"LLE Error: tau must be a positive number, got {tau}")
               return np.nan

        required_len_nolds = (emb_dim - 1) * lag + traj_len + min_tsep
        if len(data) < required_len_nolds:
             logger.error(f"LLE Error: Data length ({len(data)}) is too short for the given parameters. Needs at least {required_len_nolds}.")
             return np.nan

        lle = nolds.lyap_r(
            data=data,
            emb_dim=emb_dim,
            lag=lag,
            min_tsep=min_tsep,
            tau=tau,
            min_neighbors='auto',
            trajectory_len=traj_len,
            fit=fit_method,
            debug_plot=debug_plot,
            plot_file=plot_file
        )

        if np.isnan(lle):
             logger.warning(f"LLE calculation returned NaN. Fitting might have failed (check debug plot if enabled).")
             if np.std(data) < 1e-10:
                  logger.warning("LLE Warning: Input time series has near-zero standard deviation.")
             return np.nan

        logger.info(f"Estimated Largest Lyapunov Exponent (per step, tau={tau}): {lle:.4e}")
        if estimated_emb_dim or estimated_lag:
             logger.warning("LLE calculated using estimated/default embedding parameters. Result may be inaccurate.")
        return lle

    except ValueError as ve:
        logger.error(f"ValueError during LLE calculation: {ve}. Check parameters/data.", exc_info=True)
        if "fit method RANSAC requires scikit-learn" in str(ve):
             logger.error("RANSAC fit method requires scikit-learn. Install it: pip install scikit-learn")
        return np.nan
    except Exception as e:
        logger.error(f"Unexpected error computing Lyapunov exponent: {e}", exc_info=True)
        return np.nan

def perform_fft_analysis(time_series, dt):
    """
    Performs FFT on a time series to look for frequency signatures.
    Returns positive frequencies and corresponding magnitudes (scaled).
    """
    logger.info("Performing FFT analysis on time series...")
    n = len(time_series)
    if n < 2:
        logger.warning(f"Time series too short ({n} points) for FFT analysis.")
        return np.array([]), np.array([])

    data = np.asarray(time_series)
    if not np.all(np.isfinite(data)):
        logger.error("FFT Error: Input data contains non-finite values (NaN or Inf).")
        return np.array([]), np.array([])
    if data.dtype.kind not in 'fc':
        data = data.astype(float)

    data_detrended = data - np.mean(data)

    try:
        if np.isrealobj(data_detrended):
            fft_result = np.fft.rfft(data_detrended)
            freqs = np.fft.rfftfreq(n, d=dt)
            fft_magnitude = np.abs(fft_result)
            fft_magnitude = fft_magnitude / n
            if n % 2 == 0:
                fft_magnitude[1:-1] *= 2.0
            else:
                fft_magnitude[1:] *= 2.0
            positive_freqs = freqs
            positive_fft_magnitude = fft_magnitude
        else:
            logger.info("FFT input data is complex. Using standard fft.")
            fft_result = np.fft.fft(data_detrended)
            freqs = np.fft.fftfreq(n, d=dt)
            positive_freq_indices = np.where(freqs >= 0)[0]
            positive_freqs = freqs[positive_freq_indices]
            fft_magnitude = np.abs(fft_result[positive_freq_indices])
            positive_fft_magnitude = fft_magnitude / n

        logger.info("FFT analysis complete.")
        return positive_freqs, positive_fft_magnitude

    except Exception as e:
        logger.error(f"Error during FFT analysis: {e}", exc_info=True)
        return np.array([]), np.array([])


# --- NEW Metrics Functions (Using pyRQA) ---

def compute_rqa_metrics(time_series, emb_dim, time_delay, threshold, threshold_type='fixed', similarity_measure='euclidean', theiler_corrector=1, min_diag_len=2, min_vert_len=2, min_white_vert_len=2, normalize=False, analysis_class=Classic):
    """
    Computes Recurrence Quantification Analysis (RQA) metrics using pyRQA.
    Requires `pip install pyRQA`. Needs functional OpenCL drivers/runtime.

    Args:
        time_series (np.ndarray): The 1D time series data.
        emb_dim (int): Embedding dimension (pyRQA: embedding_dimension).
        time_delay (int): Time delay (lag) (pyRQA: time_delay).
        threshold (float): Neighbourhood radius ('fixed') or fraction of std dev ('adaptive').
                            (pyRQA: radius parameter of FixedRadius neighbourhood).
        threshold_type (str): 'fixed' or 'adaptive'. 'adaptive' calculates radius based on std dev.
        similarity_measure (str): Metric name ('euclidean', 'manhattan', 'supremum'). pyRQA uses classes.
        theiler_corrector (int): Correction for autocorrelation. Points within this temporal distance
                                 are excluded from recurrence calculation (pyRQA: theiler_corrector).
                                 Default 1 excludes the main diagonal.
        min_diag_len (int): Minimum diagonal line length for DET (pyRQA: min_diagonal_line_length).
        min_vert_len (int): Minimum vertical line length for LAM (pyRQA: min_vertical_line_length).
        min_white_vert_len (int): Minimum white vertical line length for TT (pyRQA: min_white_vertical_line_length).
        normalize (bool): Whether to normalize the time series (z-score) *before* analysis.
                          NOTE: pyRQA's Settings object doesn't take 'normalize'. Apply manually if needed.
        analysis_class (Type): The pyRQA analysis type class (e.g., Classic, Cross). Default: Classic.

    Returns:
        dict: Dictionary containing RQA metrics (mapped to common names like RR, DET, LAM, TT, ENTR)
              and the Recurrence Matrix itself ('RP_Matrix'). Also includes 'threshold_value'.
              Returns an empty dictionary if calculation fails or pyRQA is unavailable.
    """
    if not RQA_AVAILABLE:
        logger.error(f"Cannot compute RQA metrics: 'pyrqa' library or dependencies not available/functional. Import error: {_RQA_IMPORT_ERROR}")
        return {}

    logger.info(f"Computing RQA metrics using pyRQA: emb_dim={emb_dim}, time_delay={time_delay}, "
                f"threshold={threshold} (type: {threshold_type}), theiler={theiler_corrector}, "
                f"l_min={min_diag_len}, v_min={min_vert_len}, normalize={normalize}...")

    # --- Input Validation ---
    if time_series is None or len(time_series) == 0:
         logger.warning("RQA (pyRQA): Time series is empty.")
         return {}
    if not isinstance(emb_dim, int) or emb_dim < 1:
         logger.error("RQA Error: emb_dim must be a positive integer.")
         return {}
    if not isinstance(time_delay, int) or time_delay < 1:
         logger.error("RQA Error: time_delay must be a positive integer.")
         return {}
    if not isinstance(threshold, (int, float)) or threshold < 0:
         logger.error("RQA Error: threshold must be a non-negative number.")
         return {}
    if threshold_type not in ['fixed', 'adaptive']:
         logger.error("RQA Error: threshold_type must be 'fixed' or 'adaptive'.")
         return {}
    if not isinstance(theiler_corrector, int) or theiler_corrector < 0:
          logger.error("RQA Error: theiler_corrector must be a non-negative integer.")
          return {}

    data = np.asarray(time_series)
    if not np.all(np.isfinite(data)):
         logger.error("RQA Error: time_series contains non-finite values (NaN/Inf).")
         return {}

    required_len = (emb_dim - 1) * time_delay + 1
    if len(data) < required_len:
         logger.error(f"RQA (pyRQA) Error: Data length ({len(data)}) is too short. Needs at least {required_len} points.")
         return {}

    # --- Computation ---
    try:
        # Manual Normalization if requested
        if normalize:
             logger.info("Normalizing (z-score) time series before RQA.")
             data_std = np.std(data)
             if data_std > 1e-9:
                 data = (data - np.mean(data)) / data_std
             else:
                 logger.warning("Skipping normalization: Data standard deviation is near zero.")
                 # Data is effectively constant, RQA might yield trivial results.

        # Map similarity measure string to pyRQA class
        metric_classes = {
            'euclidean': EuclideanMetric, 'maximum': MaximumMetric, 'supremum': MaximumMetric,
            'taxicab': TaxicabMetric, 'manhattan': TaxicabMetric
        }
        similarity_measure_class = metric_classes.get(similarity_measure.lower())
        if similarity_measure_class is None:
             logger.error(f"RQA (pyRQA): Invalid similarity_measure '{similarity_measure}'.")
             return {}

        # Determine threshold radius
        if threshold_type == 'adaptive':
            # Use standard deviation of potentially normalized data
            ts_std_dev = np.std(data) # std dev of (potentially normalized) data
            if ts_std_dev < 1e-9:
                 logger.warning("RQA (pyRQA): Data standard deviation is near zero. Using fixed threshold 0.01.")
                 current_threshold = 0.01
            else:
                current_threshold = threshold * ts_std_dev
                logger.info(f"RQA (pyRQA): Adaptive threshold calculated: {current_threshold:.3f} ({threshold} * std_dev {ts_std_dev:.3f})")
        else: # threshold_type == 'fixed'
             current_threshold = threshold

        # Create TimeSeries object
        ts_object = TimeSeries(data, embedding_dimension=emb_dim, time_delay=time_delay)

        # Define Settings (REMOVED 'normalize' argument)
        settings = Settings(
            time_series=ts_object,
            analysis_type=analysis_class,
            neighbourhood=FixedRadius(current_threshold),
            similarity_measure=similarity_measure_class,
            theiler_corrector=theiler_corrector
            # Note: normalize argument removed here
        )

        # Run RQA Computation
        logger.debug("Creating RQAComputation...")
        computation = RQAComputation.create(settings, verbose=False)
        logger.debug("Running RQA computation...")
        rqa_result = computation.run()
        logger.debug("RQA computation finished.")

        # Set analysis parameters on the result object
        rqa_result.min_diagonal_line_length = min_diag_len
        rqa_result.min_vertical_line_length = min_vert_len
        rqa_result.min_white_vertical_line_length = min_white_vert_len

        # Get the Recurrence Matrix
        logger.debug("Creating RPComputation for matrix...")
        rp_computation = RPComputation.create(settings, verbose=False)
        logger.debug("Running RP computation...")
        rp_result = rp_computation.run()
        rp_matrix = rp_result.recurrence_matrix_reverse.copy()
        logger.debug("RP computation finished.")

        # Extract results
        metric_map = {
            'recurrence_rate': 'RR', 'determinism': 'DET', 'average_diagonal_line': 'L',
            'longest_diagonal_line': 'L_max', 'divergence': 'DIV', 'entropy_diagonal_lines': 'ENTR',
            'laminarity': 'LAM', 'trapping_time': 'TT', 'longest_vertical_line': 'V_max',
            'entropy_vertical_lines': 'V_entr', 'average_white_vertical_line': 'W',
            'longest_white_vertical_line': 'W_max', 'longest_white_vertical_line_inverse': 'W_div',
            'entropy_white_vertical_lines': 'W_entr'
        }
        rqa_results_mapped = {}
        for pyrqa_attr, standard_name in metric_map.items():
            if hasattr(rqa_result, pyrqa_attr):
                value = getattr(rqa_result, pyrqa_attr)
                if isinstance(value, (np.number, np.bool_)):
                    rqa_results_mapped[standard_name] = value.item()
                else:
                    rqa_results_mapped[standard_name] = value
            else:
                logger.warning(f"RQA (pyRQA): Metric attribute '{pyrqa_attr}' not found. Setting to NaN.")
                rqa_results_mapped[standard_name] = np.nan

        # Store the matrix and key parameters
        rqa_results_mapped['RP_Matrix'] = rp_matrix
        rqa_results_mapped['threshold_value'] = current_threshold
        rqa_results_mapped['embedding_dimension'] = emb_dim
        rqa_results_mapped['time_delay'] = time_delay

        logger.info(f"RQA (pyRQA) Results: RR={rqa_results_mapped.get('RR', np.nan):.4f}, DET={rqa_results_mapped.get('DET', np.nan):.4f}, LAM={rqa_results_mapped.get('LAM', np.nan):.4f}, L={rqa_results_mapped.get('L', np.nan):.4f}, ENTR={rqa_results_mapped.get('ENTR', np.nan):.4f} (Threshold={current_threshold:.3f})")
        return rqa_results_mapped

    except Exception as e:
        logger.error(f"Error computing RQA metrics with pyRQA: {e}", exc_info=True)
        err_str = str(e).lower()
        if "platform_not_found" in err_str or "clgetplatformids failed" in err_str \
           or "clbuildprogram failed" in err_str or "could not find platform" in err_str \
           or "invalid kernel name" in err_str:
            logger.error("***************************************************************")
            logger.error("PyRQA Error likely due to OpenCL setup issues.")
            logger.error("Check OpenCL drivers/SDK. See PyRQA/PyOpenCL docs.")
            logger.error("***************************************************************")
        return {}


def perform_cwt(time_series, dt, wavelet_type='cmor1.5-1.0', scales=np.arange(1, 64)):
    """
    Performs Continuous Wavelet Transform (CWT) using PyWavelets.
    Requires `pip install PyWavelets`.
    """
    if not CWT_AVAILABLE:
        logger.error("Cannot perform CWT: 'PyWavelets' library not available.")
        return None, None
    if _USE_SCIPY_CWT:
         logger.warning("Using scipy.signal.cwt - function not implemented here. Skipping CWT.")
         return None, None

    logger.info(f"Performing CWT: wavelet='{wavelet_type}', {len(scales)} scales, dt={dt:.4e}...")

    if time_series is None or len(time_series) < 2:
        logger.warning(f"CWT: Time series is empty or too short ({len(time_series) if time_series is not None else 0} points).")
        return None, None

    data = np.asarray(time_series)
    if not np.all(np.isfinite(data)):
        logger.error("CWT Error: Input data contains non-finite values (NaN or Inf).")
        return None, None
    if data.dtype.kind not in 'fc':
        data = data.astype(float)

    try:
        coefficients, freqs_pywt = pywt.cwt(data, scales, wavelet_type, sampling_period=dt)
        logger.info(f"CWT calculation complete. Coefficients shape: {coefficients.shape}")
        return coefficients, freqs_pywt

    except ValueError as ve:
         logger.error(f"ValueError during CWT calculation: {ve}. Check wavelet type ('{wavelet_type}'), scales, or data.", exc_info=True)
         return None, None
    except Exception as e:
        logger.error(f"Unexpected error performing CWT with PyWavelets: {e}", exc_info=True)
        return None, None


def compute_correlation_dimension(time_series, emb_dim, lag, rvals_count=20, fit='poly', debug_plot=False, plot_file=None):
    """
    Computes the correlation dimension using the Grassberger-Procaccia algorithm
    via the 'nolds' library.
    Requires `pip install nolds`.

    Args:
        time_series (np.ndarray): The 1D time series data.
        emb_dim (int): Embedding dimension.
        lag (int): Time delay (lag).
        rvals_count (int): Approximate number of radii 'r' nolds should use. (Note: nolds handles selection).
        fit (str): Method to fit the log-log plot ('poly' or 'RANSAC').
        debug_plot (bool): If True, nolds generates a log-log plot of C(r) vs r.
        plot_file (str, optional): File path to save the debug plot.

    Returns:
        float: The estimated correlation dimension for the given emb_dim.
               Returns np.nan if 'nolds' is not available, data is too short,
               or calculation/fitting fails.
    """
    if not NOLDS_AVAILABLE:
        logger.error("Cannot compute Correlation Dimension: 'nolds' library not installed.")
        return np.nan

    logger.info(f"Computing Correlation Dimension: emb_dim={emb_dim}, lag={lag}, approx_rvals={rvals_count}, fit='{fit}'...")

    if time_series is None or len(time_series) < 100:
        logger.warning(f"Time series too short ({len(time_series) if time_series is not None else 0} points) for reliable CD calc.")
        return np.nan

    if not isinstance(emb_dim, int) or emb_dim < 1:
         logger.error("CorrDim Error: emb_dim must be a positive integer.")
         return np.nan
    if not isinstance(lag, int) or lag < 1:
         logger.error("CorrDim Error: lag must be a positive integer.")
         return np.nan
    if fit not in ['poly', 'RANSAC']:
         logger.error("CorrDim Error: fit must be 'poly' or 'RANSAC'.")
         return np.nan

    data = np.asarray(time_series)
    if not np.all(np.isfinite(data)):
        logger.error("CorrDim Error: Input data contains non-finite values (NaN or Inf).")
        return np.nan
    if data.dtype.kind not in 'fc':
        data = data.astype(float)

    required_len = (emb_dim - 1) * lag + 1 # Basic check
    if len(data) < required_len:
         logger.error(f"CorrDim Error: Data length ({len(data)}) too short for emb_dim={emb_dim}, lag={lag}. Needs {required_len}.")
         return np.nan

    try:
        logger.debug(f"Letting nolds choose the radii (approx {rvals_count}).")
        # Call nolds.corr_dim WITHOUT rvals_n
        corr_dim_est = nolds.corr_dim(
            data=data,
            emb_dim=emb_dim,
            lag=lag,
            # rvals_n=rvals_count, # REMOVED this argument
            fit=fit,
            debug_plot=debug_plot,
            plot_file=plot_file
            # nolds will use its default number of radii (~100) if rvals/rvals_n not given
        )

        if np.isnan(corr_dim_est):
             logger.warning(f"Correlation Dimension calculation returned NaN. Fitting might have failed (check debug plot if enabled).")
             if np.std(data) < 1e-10:
                  logger.warning("CorrDim Warning: Input time series has near-zero standard deviation.")
             return np.nan

        logger.info(f"Estimated Correlation Dimension for emb_dim={emb_dim}: {corr_dim_est:.4f}")
        return corr_dim_est

    except ValueError as ve:
        # Catch specific errors if needed, e.g., RANSAC dependency
        logger.error(f"ValueError during Correlation Dimension calculation: {ve}. Check parameters/data.", exc_info=True)
        if "fit method RANSAC requires scikit-learn" in str(ve):
             logger.error("RANSAC fit method requires scikit-learn. Install it: pip install scikit-learn")
        return np.nan
    except TypeError as te:
        # Catch potential argument errors if nolds API changes
         logger.error(f"TypeError during Correlation Dimension calculation: {te}. Check nolds function signature.", exc_info=True)
         return np.nan
    except Exception as e:
        logger.error(f"Unexpected error computing Correlation Dimension: {e}", exc_info=True)
        return np.nan

# --- END OF FILE metrics.py ---