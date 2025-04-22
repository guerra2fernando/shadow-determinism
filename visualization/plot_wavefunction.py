# --- START OF FILE quantum_chaos_sim/visualization/plot_wavefunction.py ---

# quantum_chaos_sim/visualization/plot_wavefunction.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec # For composite plots
import numpy as np
import os
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable # For colorbars
import pandas as pd # Import pandas for checking types
from types import SimpleNamespace # Import SimpleNamespace

logger = logging.getLogger(__name__)

# --- Dependency Check Result Imports ---
# Import flags from metrics to avoid plotting if library missing
try:
    # Use relative import from sibling directory
    from ..validation.metrics import RQA_AVAILABLE, CWT_AVAILABLE, NOLDS_AVAILABLE
except ImportError:
    logger.warning("Could not import dependency flags from metrics.py. Plotting functions might attempt to run even if libraries are missing.")
    RQA_AVAILABLE, CWT_AVAILABLE, NOLDS_AVAILABLE = False, False, False


# --- Helper Functions ---
def _get_config_value(config_obj, attr, default):
    """Safely gets an attribute from config_obj or returns default."""
    if isinstance(config_obj, dict):
        return config_obj.get(attr, default)
    if isinstance(config_obj, SimpleNamespace):
        return getattr(config_obj, attr, default)
    # Fallback for other potential types (e.g., module object)
    if hasattr(config_obj, attr):
        return getattr(config_obj, attr)
    return default


def _sanitize_filename(name):
    """Removes or replaces characters potentially problematic in filenames."""
    if not isinstance(name, str):
         name = str(name) # Convert potential non-strings
    chars_to_remove = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '[', ']', ','] # Added brackets, comma
    sanitized = name
    for char in chars_to_remove:
        sanitized = sanitized.replace(char, '')
    sanitized = sanitized.replace(' ', '_') # Replace spaces with underscores
    sanitized = sanitized.replace('(', '').replace(')', '') # Remove parentheses too
    sanitized = sanitized.replace('=', 'eq').replace('.', 'p') # Replace equals, period
    # Limit length
    max_len = 100
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "_trunc"
    return sanitized

def _ensure_plot_dir(config_obj):
    """Creates the plot directory if it doesn't exist."""
    plot_dir = _get_config_value(config_obj, 'plot_dir', None)
    if plot_dir is None:
        try:
             pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             plot_dir = os.path.join(pkg_root, "results", "plots")
        except NameError:
             plot_dir = os.path.join(os.getcwd(), "results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def _save_or_show_plot(fig, filepath, save_flag):
    """Saves or shows the current figure based on save_flag. Returns filepath if saved."""
    saved_path = None
    if save_flag:
        try:
            fig.savefig(filepath, bbox_inches='tight', dpi=150) # Add dpi
            logger.info(f"Plot saved to {filepath}")
            saved_path = filepath # Store path if saved
        except Exception as e:
            logger.error(f"Failed to save plot to {filepath}: {e}", exc_info=True)
    else:
        try:
            plt.show()
        except Exception as e:
            if "TclError" in str(e) or "no display name" in str(e).lower() or "cannot connect to X server" in str(e).lower():
                 logger.warning(f"Cannot display plot interactively (likely no GUI): {e}")
            else:
                 logger.warning(f"Could not display plot interactively: {e}", exc_info=True)
    plt.close(fig) # Close figure to free memory
    return saved_path # Return path if saved, else None


# --- Plotting Functions (Refined with flags and return values) ---

def plot_probability_heatmap(psi_history, x_grid, t_grid, filename="probability_heatmap.png", config_obj=None):
    """ Plots a heatmap of the probability density |psi(x, t)|^2. Returns filepath if saved."""
    # Check flags first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_per_run = _get_config_value(config_obj, 'save_plots_per_run', False)
    save_heatmap_flag = _get_config_value(config_obj, 'save_heatmap', True)
    if not (save_results and save_plots_per_run and save_heatmap_flag):
        return None

    plot_dir = _ensure_plot_dir(config_obj)
    filepath = os.path.join(plot_dir, filename)

    logger.info(f"Generating probability density heatmap ({filename})...")
    if psi_history is None or psi_history.size == 0:
        logger.warning(f"Skipping heatmap generation for {filename}: psi_history is empty.")
        return None

    prob_density_history = np.abs(psi_history)**2

    fig, ax = plt.subplots(figsize=(10, 6))
    if t_grid is None or x_grid is None or len(t_grid) < 2 or len(x_grid) < 2:
         logger.warning(f"Skipping heatmap for {filename}: Invalid t_grid or x_grid.")
         plt.close(fig); return None

    # Determine extent based on input data dimensions vs grid dimensions
    history_steps, history_points = prob_density_history.shape
    grid_steps, grid_points = len(t_grid), len(x_grid)
    t_extent_max = t_grid[-1]; x_extent_min = x_grid[0]; x_extent_max = x_grid[-1]

    if history_steps != grid_steps:
         logger.warning(f"Time step mismatch for heatmap {filename}: History ({history_steps}) vs t_grid ({grid_steps}). Plotting available history.")
         if history_steps < grid_steps:
              t_extent_max = t_grid[history_steps-1]
         else:
              prob_density_history = prob_density_history[:grid_steps, :]
    if history_points != grid_points:
         logger.warning(f"Spatial point mismatch for heatmap {filename}: History ({history_points}) vs x_grid ({grid_points}). Plotting might be distorted.")

    try:
        im = ax.imshow(
            prob_density_history.T,
            extent=[t_grid[0], t_extent_max, x_extent_min, x_extent_max],
            aspect='auto', origin='lower', cmap='viridis', interpolation='nearest'
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label=r'$|\psi(x, t)|^2$')
        ax.set_xlabel('Time (t)'); ax.set_ylabel('Position (x)')
        ax.set_title(fr'Probability Density Evolution $|\psi(x, t)|^2$' + f'\n({os.path.splitext(filename)[0]})')
        plt.tight_layout()
        saved_path = _save_or_show_plot(fig, filepath, save_results)
        return saved_path # Return filepath if saved
    except Exception as e:
         logger.error(f"Failed to generate heatmap plot '{filename}': {e}", exc_info=True)
         plt.close(fig)
         return None


def plot_chaos_trajectory(t_chaos, states_chaos, z_signal, filename="lorenz_trajectory.png", config_obj=None):
    """ Plots the chaos driver trajectory and z(t) signal. Returns filepath if saved."""
    # Check flags first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_per_run = _get_config_value(config_obj, 'save_plots_per_run', False)
    if not (save_results and save_plots_per_run):
        return None

    plot_dir = _ensure_plot_dir(config_obj)
    filepath = os.path.join(plot_dir, filename)

    logger.info(f"Plotting chaos driver trajectory ({filename})...")
    fig = plt.figure(figsize=(12, 5))
    is_3d_plot_possible = states_chaos is not None and isinstance(states_chaos, np.ndarray) and states_chaos.ndim == 2 and states_chaos.shape[1] >= 3 and states_chaos.shape[0] > 0
    ax1 = fig.add_subplot(1, 2, 1, projection='3d' if is_3d_plot_possible else None)

    if is_3d_plot_possible:
        try:
            ax1.plot(states_chaos[:, 0], states_chaos[:, 1], states_chaos[:, 2], lw=0.5, alpha=0.8)
            ax1.set_xlabel("X Axis"); ax1.set_ylabel("Y Axis"); ax1.set_zlabel("Z Axis")
            ax1.set_title("Driver Trajectory (3D)")
            ax1.scatter(states_chaos[0,0], states_chaos[0,1], states_chaos[0,2], c='r', marker='o', s=50, label='Start', zorder=10)
            ax1.legend()
        except Exception as e:
            logger.error(f"Error plotting 3D trajectory for {filename}: {e}")
            ax1.remove(); ax1 = fig.add_subplot(1, 2, 1)
            ax1.text(0.5, 0.5, "3D Plot Error", ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Driver Trajectory (Error)")
    else:
        txt = "No 3D Trajectory Data" if states_chaos is not None else "No Trajectory Data"
        ax1.text(0.5, 0.5, txt, ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Driver Trajectory")

    ax2 = fig.add_subplot(1, 2, 2)
    driver_component_name = getattr(config_obj, 'info_flow_observable_driver', 'z')
    if t_chaos is not None and z_signal is not None and len(t_chaos) == len(z_signal) and len(t_chaos) > 0:
        ax2.plot(t_chaos, z_signal); ax2.set_xlabel("Time (t)"); ax2.set_ylabel(f"{driver_component_name}(t)")
        ax2.set_title(f"Driver Signal {driver_component_name}(t) used in Potential"); ax2.grid(True)
    elif z_signal is not None and len(z_signal) > 0:
        logger.warning(f"Length mismatch or missing t_chaos in {filename}. Plotting z vs index.")
        ax2.plot(z_signal); ax2.set_xlabel("Time index"); ax2.set_ylabel(f"{driver_component_name}(t)")
        ax2.set_title(f"Driver Signal {driver_component_name}(t) (Index Based)"); ax2.grid(True)
    else:
        logger.warning(f"No z_signal data to plot in {filename}.")
        ax2.text(0.5, 0.5, "No Signal Data", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f"Driver Signal {driver_component_name}(t) (No Data)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Chaos Driver ({os.path.splitext(filename)[0]})", fontsize=14)
    saved_path = _save_or_show_plot(fig, filepath, save_results)
    return saved_path


def plot_observables(t_grid, observables, filename="observables.png", config_obj=None):
    """ Plots the calculated observables over time. Returns filepath if saved."""
    # Check flags first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_per_run = _get_config_value(config_obj, 'save_plots_per_run', False)
    if not (save_results and save_plots_per_run):
        return None

    plot_dir = _ensure_plot_dir(config_obj)
    filepath = os.path.join(plot_dir, filename)

    logger.info(f"Plotting observables ({filename})...")
    if not observables: logger.warning(f"No observables data provided for plotting {filename}."); return None

    plot_keys = [k for k, v in observables.items()
                 if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1
                 and not k.startswith('psi_final_') and not k.startswith('driver_')
                 and k not in ['RP_Matrix', 'CWT_coeffs_computed']] # Exclude non-time series

    num_plots = len(plot_keys)
    if num_plots == 0: logger.warning(f"No valid time series observables found for plotting {filename}."); return None

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True, squeeze=False)
    axes = axes.flatten()
    plot_any_success = False

    for i, name in enumerate(sorted(plot_keys)):
        data = observables.get(name); ax = axes[i]
        if not isinstance(data, np.ndarray):
             logger.warning(f"Observable '{name}' not NumPy array in {filename}. Skipping."); ax.set_title(f"{name} (Data Error)"); continue
        if t_grid is None or len(data) != len(t_grid):
             data_len = len(data) if data is not None else 'None'; tgrid_len = len(t_grid) if t_grid is not None else 'None'
             logger.warning(f"Length mismatch '{name}' in {filename}: data ({data_len}), t_grid ({tgrid_len}). Skipping."); ax.set_title(f"{name} (Length Mismatch)"); continue
        if np.all(np.isnan(data)):
              logger.warning(f"Observable '{name}' contains only NaNs in {filename}. Skipping plot."); ax.set_title(f"{name} (All NaN)"); continue

        try:
            valid_indices = ~np.isnan(data); t_valid = t_grid[valid_indices]; data_valid = data[valid_indices]
            if len(data_valid) == 0:
                 logger.warning(f"No valid (non-NaN) data points for observable '{name}' in {filename}. Skipping plot."); ax.set_title(f"{name} (No Valid Data)"); continue
            label_text = name.replace('<', r'\langle ').replace('>', r'\rangle')
            ax.plot(t_valid, data_valid, label=name.split(" (")[0], lw=1.5)
            ax.set_ylabel(rf'${label_text}$'); ax.grid(True)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,4), useMathText=True)
            if name == "Norm Conservation":
                ax.plot(t_valid, data_valid - 1.0, label=r'Norm$^2 - 1$', lw=1.5)
                ax.set_ylabel(r'Norm$^2$ Deviation')
                ax.axhline(0.0, color='r', linestyle='--', lw=0.8, label=r'Ideal Norm$^2=1$')
                ax.legend(loc='best'); ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3), useMathText=True)
            elif "Entropy" in name: ax.ticklabel_format(axis='y', style='plain'); ax.legend(loc='best')
            else: ax.legend(loc='best')
            plot_any_success = True
        except Exception as e: logger.error(f"Error plotting observable '{name}' in {filename}: {e}", exc_info=True); ax.set_title(f"{name} (Plotting Error)")

    if not plot_any_success: logger.error(f"Failed to plot any observables for {filename}."); plt.close(fig); return None
    axes[-1].set_xlabel("Time (t)")
    fig.suptitle(f"System Observables vs Time\n({os.path.splitext(filename)[0]})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    saved_path = _save_or_show_plot(fig, filepath, save_results)
    return saved_path


def animate_wavefunction(psi_history, x_grid, t_grid, potential_history=None, interval=50, filename="wavefunction_animation.mp4", config_obj=None):
    """ Animates the wavefunction probability density |psi(x,t)|^2. Returns filepath if saved."""
    save_results = _get_config_value(config_obj, 'save_results', False)
    animate_flag = _get_config_value(config_obj, 'animate', False)
    if not (save_results and animate_flag): return None

    plot_dir = _ensure_plot_dir(config_obj)
    filepath = os.path.join(plot_dir, filename)
    frame_step = max(1, int(_get_config_value(config_obj, 'plot_interval', 50)))

    logger.info(f"Attempting to generate wavefunction animation ({filename})...")
    if psi_history is None or psi_history.size == 0: logger.warning(f"Skipping animation {filename}: psi_history is empty."); return None
    if x_grid is None or t_grid is None: logger.warning(f"Skipping animation {filename}: x_grid or t_grid is missing."); return None
    if len(psi_history) != len(t_grid):
        logger.warning(f"Length mismatch for animation {filename}: psi_history ({len(psi_history)}) vs t_grid ({len(t_grid)}). Using shorter length.")
        num_frames_total = min(len(psi_history), len(t_grid))
        psi_history = psi_history[:num_frames_total]; t_grid = t_grid[:num_frames_total]
        if potential_history is not None and len(potential_history) > num_frames_total: potential_history = potential_history[:num_frames_total]
    else: num_frames_total = len(psi_history)

    fig, ax = plt.subplots(figsize=(10, 6))
    prob_density_history = np.abs(psi_history)**2
    max_prob = np.max(prob_density_history) if prob_density_history.size > 0 else 1.0
    if max_prob <= 1e-9: max_prob = 1.0

    line, = ax.plot(x_grid, prob_density_history[0, :], label=r'$|\psi(x, t)|^2$')
    ax.set_ylim(0, max_prob * 1.1); ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_xlabel("Position (x)"); ax.set_ylabel(r"Probability Density $|\psi|^2$")
    ax.set_title(f"Wavefunction Evolution\n({os.path.splitext(filename)[0]})")
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    potential_line, ax2, plot_potential = None, None, False
    if potential_history is not None and potential_history.shape == psi_history.shape and potential_history.size > 0:
        plot_potential = True; logger.debug(f"Potential history found for animation {filename}, shape: {potential_history.shape}")
    elif potential_history is not None: logger.warning(f"Potential history shape mismatch or empty in animation {filename}. Potential will not be plotted.")

    if plot_potential:
        ax2 = ax.twinx(); min_V = np.min(potential_history); max_V = np.max(potential_history)
        if np.isnan(min_V) or np.isinf(min_V) or np.isnan(max_V) or np.isinf(max_V): min_V_lim, max_V_lim = -5, 5
        else: v_range = max_V - min_V; v_buffer = max(0.1 * abs(min_V), 0.1 * abs(max_V), 0.1) if v_range > 1e-6 else 0.5; min_V_lim, max_V_lim = min_V - v_buffer, max_V + v_buffer
        potential_line, = ax2.plot(x_grid, potential_history[0,:], 'r--', lw=0.8, label='Potential V(x,t)', alpha=0.7)
        ax2.set_ylim(min_V_lim, max_V_lim); ax2.set_ylabel("Potential V(x,t)", color='r'); ax2.tick_params(axis='y', labelcolor='r')
        lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else: ax.legend(loc='upper right')

    def update(frame_index_actual):
        line.set_ydata(prob_density_history[frame_index_actual, :])
        time_text.set_text(f'Time = {t_grid[frame_index_actual]:.2f} (Step {frame_index_actual})')
        lines_to_return = [line, time_text]
        if potential_line is not None and plot_potential: potential_line.set_ydata(potential_history[frame_index_actual,:]); lines_to_return.append(potential_line)
        return lines_to_return

    frames_to_render = list(range(0, num_frames_total, frame_step))
    if num_frames_total > 0 and (num_frames_total - 1) not in frames_to_render: frames_to_render.append(num_frames_total - 1)
    if not frames_to_render: logger.warning(f"No frames to render for animation {filename}. Skipping."); plt.close(fig); return None

    logger.info(f"Creating animation with {len(frames_to_render)} frames (step={frame_step}, interval={interval}ms)...")
    anim = animation.FuncAnimation(fig, update, frames=frames_to_render, blit=True, interval=interval, repeat=False)
    saved_path = None

    if save_results:
        try:
            writer = animation.writers['ffmpeg'](fps=max(1,int(1000/interval)), metadata=dict(artist='QuantumChaosSim'), bitrate=1800)
            anim.save(filepath, writer=writer, dpi=150); logger.info(f"Animation saved to {filepath}"); saved_path = filepath
        except (FileNotFoundError, KeyError):
             logger.warning(f"FFmpeg writer not found or available. Trying Pillow (GIF)...")
             try:
                 writer = animation.PillowWriter(fps=max(1,int(1000/interval))); gif_filepath = os.path.splitext(filepath)[0] + ".gif"
                 anim.save(gif_filepath, writer=writer, dpi=150); logger.info(f"Animation saved as GIF to {gif_filepath}"); saved_path = gif_filepath
             except Exception as e_gif: logger.error(f"Failed to save animation using FFmpeg or Pillow: {e_gif}", exc_info=True)
        except Exception as e: logger.error(f"Failed to save animation to {filepath}. Error: {e}", exc_info=True)
    else:
        try: plt.show()
        except Exception as e: logger.warning(f"Could not display animation interactively: {e}")
    plt.close(fig)
    return saved_path


def plot_comparison(t_grid, observables_run1, observables_run2, run1_label="Run1", run2_label="Run2", filename_prefix="comparison", config_obj=None):
    """ Compares observables and final state between two runs. Returns list of generated filepaths."""
    # Check flag first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_comparison = _get_config_value(config_obj, 'save_plots_comparison', False)
    if not (save_results and save_plots_comparison):
        return None # Return None (or empty list) if disabled

    plot_dir = _ensure_plot_dir(config_obj)
    x_grid = _get_config_value(config_obj, 'x_grid', None)
    generated_files = []

    logger.info(f"Plotting comparison: {run1_label} vs {run2_label} (prefix: {filename_prefix})...")
    keys1 = set(k for k, v in observables_run1.items() if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1 and not k.startswith('psi_final_') and not k.startswith('driver_') and k not in ['RP_Matrix', 'CWT_coeffs_computed'])
    keys2 = set(k for k, v in observables_run2.items() if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1 and not k.startswith('psi_final_') and not k.startswith('driver_') and k not in ['RP_Matrix', 'CWT_coeffs_computed'])
    common_keys = sorted(list(keys1 & keys2))
    num_plots_obs = len(common_keys)

    if num_plots_obs > 0:
        fig_obs, axes_obs = plt.subplots(num_plots_obs, 1, figsize=(10, 2.5 * num_plots_obs), sharex=True, squeeze=False); axes_obs = axes_obs.flatten()
        plot_any_success_obs = False
        for i, name in enumerate(common_keys):
            data1 = observables_run1.get(name); data2 = observables_run2.get(name); ax = axes_obs[i]
            if data1 is None or data2 is None: logger.warning(f"Missing data for '{name}' in comparison '{filename_prefix}'. Skipping."); continue
            if t_grid is None or len(data1) != len(t_grid) or len(data2) != len(t_grid): logger.warning(f"Length mismatch for '{name}' in comparison '{filename_prefix}'. Skipping."); ax.set_title(f"{name} (Length Mismatch)"); continue
            try:
                clean_label1 = run1_label.split('_', 1)[1] if '_' in run1_label else run1_label; clean_label2 = run2_label.split('_', 1)[1] if '_' in run2_label else run2_label
                valid1 = ~np.isnan(data1); valid2 = ~np.isnan(data2); t_valid1 = t_grid[valid1]; data_valid1 = data1[valid1]; t_valid2 = t_grid[valid2]; data_valid2 = data2[valid2]
                if len(data_valid1) > 0: ax.plot(t_valid1, data_valid1, label=f'{clean_label1}', lw=1.5, alpha=0.9)
                if len(data_valid2) > 0: ax.plot(t_valid2, data_valid2, '--', label=f'{clean_label2}', lw=1.5, alpha=0.9)
                label_text = name.replace('<', r'\langle ').replace('>', r'\rangle'); ax.set_ylabel(rf'${label_text}$'); ax.grid(True); ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,4), useMathText=True)
                if name == "Norm Conservation":
                    ax.clear(); ax.plot(t_valid1, data_valid1 - 1.0, label=f'{clean_label1} (Dev)', lw=1.5, alpha=0.9); ax.plot(t_valid2, data_valid2 - 1.0, '--', label=f'{clean_label2} (Dev)', lw=1.5, alpha=0.9)
                    ax.set_ylabel(r'Norm$^2$ Deviation'); ax.axhline(0.0, color='r', linestyle=':', lw=0.8, label='Ideal Norm=1'); ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3), useMathText=True)
                elif "Entropy" in name: ax.ticklabel_format(axis='y', style='plain')
                if len(data_valid1) > 0 or len(data_valid2) > 0: ax.legend(loc='best'); plot_any_success_obs = True
                else: ax.set_title(f"{name} (All NaN Data)")
            except Exception as e: logger.error(f"Error plotting comparison for observable '{name}': {e}", exc_info=True); ax.set_title(f"{name} (Plotting Error)")

        if plot_any_success_obs:
            axes_obs[-1].set_xlabel("Time (t)"); clean_label1_title = run1_label.split('_', 1)[1] if '_' in run1_label else run1_label; clean_label2_title = run2_label.split('_', 1)[1] if '_' in run2_label else run2_label
            fig_obs.suptitle(f"Observables Comparison: {clean_label1_title} vs {clean_label2_title}", fontsize=14); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filepath_obs = os.path.join(plot_dir, f"{_sanitize_filename(filename_prefix)}_observables.png")
            saved_path = _save_or_show_plot(fig_obs, filepath_obs, save_results)
            if saved_path: generated_files.append(os.path.basename(saved_path))
        else: logger.info(f"No common time series observables plotted for comparison '{filename_prefix}'."); plt.close(fig_obs)
    else: logger.warning(f"No common time series observables found for comparison '{filename_prefix}'.")

    psi_final_key1 = f'psi_final_{run1_label}'; psi_final_key2 = f'psi_final_{run2_label}'; psi_final1 = observables_run1.get(psi_final_key1); psi_final2 = observables_run2.get(psi_final_key2)
    if psi_final1 is not None and psi_final2 is not None and x_grid is not None and len(psi_final1) == len(x_grid) and len(psi_final2) == len(x_grid):
        logger.info(f"Plotting final state comparison for {run1_label} vs {run2_label}...")
        prob1 = np.abs(psi_final1)**2; prob2 = np.abs(psi_final2)**2; fig_final, ax_final = plt.subplots(figsize=(10, 6))
        clean_label1 = run1_label.split('_', 1)[1] if '_' in run1_label else run1_label; clean_label2 = run2_label.split('_', 1)[1] if '_' in run2_label else run2_label
        ax_final.plot(x_grid, prob1, label=rf'$|\psi_{{{clean_label1}}}(x, T)|^2$', lw=1.5, alpha=0.9); ax_final.plot(x_grid, prob2, '--', label=rf'$|\psi_{{{clean_label2}}}(x, T)|^2$', lw=1.5, alpha=0.9)
        ax_final.set_xlabel("Position (x)"); ax_final.set_ylabel(r"Final Probability Density $|\psi(x,T)|^2$"); ax_final.grid(True, axis='both', linestyle='--', alpha=0.6)
        ax_diff = ax_final.twinx(); diff = prob1 - prob2; ax_diff.plot(x_grid, diff, ':', color='gray', alpha=0.7, lw=1.0, label='Difference ($Run1 - Run2$)')
        ax_diff.set_ylabel("Probability Density Difference", color='gray'); ax_diff.tick_params(axis='y', labelcolor='gray'); ax_diff.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        max_abs_diff = np.max(np.abs(diff)) if diff.size > 0 else 1.0; ax_diff.set_ylim(-max_abs_diff * 1.1, max_abs_diff * 1.1) if max_abs_diff > 1e-9 else ax_diff.set_ylim(-0.1, 0.1)
        ax_final.set_title(f"Comparison of Final Probability Densities\n({clean_label1} vs {clean_label2})"); lines, labels = ax_final.get_legend_handles_labels(); lines2, labels2 = ax_diff.get_legend_handles_labels()
        ax_final.legend(lines + lines2, labels + labels2, loc='best'); plt.tight_layout()
        filepath_final = os.path.join(plot_dir, f"{_sanitize_filename(filename_prefix)}_final_state_comparison.png")
        saved_path = _save_or_show_plot(fig_final, filepath_final, save_results)
        if saved_path: generated_files.append(os.path.basename(saved_path))
    else: logger.warning(f"Final state data or x_grid missing/invalid, cannot plot final state comparison for '{filename_prefix}'.")

    return generated_files if generated_files else None


def plot_recurrence(rqa_metrics_dict, filename="recurrence_plot.png", config_obj=None):
    """ Plots a Recurrence Plot (RP). Returns filepath if saved."""
    # Check flags first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_per_run = _get_config_value(config_obj, 'save_plots_per_run', False)
    if not (save_results and save_plots_per_run):
        return None

    plot_dir = _ensure_plot_dir(config_obj)
    filepath = os.path.join(plot_dir, filename)

    if not RQA_AVAILABLE: logger.warning(f"Cannot plot recurrence plot {filename}: pyRQA not available."); return None
    if not rqa_metrics_dict or not isinstance(rqa_metrics_dict, dict): logger.warning(f"Cannot plot recurrence plot {filename}: Invalid rqa_metrics_dict provided."); return None
    rp_matrix = rqa_metrics_dict.get('RP_Matrix')
    if rp_matrix is None: logger.warning(f"Cannot plot recurrence plot {filename}: 'RP_Matrix' not found in rqa_metrics_dict."); return None

    logger.info(f"Generating recurrence plot ({filename})...")
    try:
        threshold_value = rqa_metrics_dict.get('threshold_value', np.nan)
        if not isinstance(rp_matrix, np.ndarray): logger.warning(f"Recurrence matrix is not a NumPy array for {filename}. Skipping plot."); return None
        N = rp_matrix.shape[0]
        if N == 0: logger.warning(f"Recurrence matrix is empty for {filename}. Skipping plot."); return None
        fig, ax = plt.subplots(figsize=(8, 8)); rp_plot_matrix = rp_matrix.astype(bool) if rp_matrix.dtype != bool else rp_matrix
        ax.imshow(rp_plot_matrix, cmap='binary', origin='lower', interpolation='nearest')
        ax.set_xlabel(f"Time Index i (0 to {N-1})"); ax.set_ylabel(f"Time Index j (0 to {N-1})")
        thresh_str = f"{threshold_value:.2e}" if not np.isnan(threshold_value) and (abs(threshold_value)<1e-2 or abs(threshold_value)>1e3) else f"{threshold_value:.2f}"
        ax.set_title(f"Recurrence Plot (Threshold={thresh_str})\n({os.path.splitext(filename)[0]})"); plt.tight_layout()
        saved_path = _save_or_show_plot(fig, filepath, save_results)
        return saved_path
    except Exception as e: logger.error(f"Failed to generate recurrence plot {filename}: {e}", exc_info=True); plt.close(fig); return None


def plot_cwt_scalogram(coeffs, times, freqs, filename="cwt_scalogram.png", config_obj=None):
    """ Plots a CWT scalogram heatmap. Returns filepath if saved."""
    # Check flags first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_per_run = _get_config_value(config_obj, 'save_plots_per_run', False)
    if not (save_results and save_plots_per_run):
        return None

    plot_dir = _ensure_plot_dir(config_obj)
    filepath = os.path.join(plot_dir, filename)

    if not CWT_AVAILABLE: logger.warning(f"Cannot plot CWT scalogram {filename}: Wavelet library not available."); return None
    if coeffs is None or times is None or freqs is None: logger.warning(f"Cannot plot CWT scalogram {filename}: Input data missing."); return None
    if coeffs.shape[1] != len(times) or coeffs.shape[0] != len(freqs): logger.warning(f"Cannot plot CWT scalogram {filename}: Data shape mismatch."); return None

    logger.info(f"Generating CWT scalogram ({filename})...")
    try:
        fig, ax = plt.subplots(figsize=(12, 6)); magnitude = np.abs(coeffs)
        T, F = np.meshgrid(times, freqs); from matplotlib.colors import LogNorm
        min_mag = np.min(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1e-9; max_mag = np.max(magnitude)
        norm = LogNorm(vmin=max(1e-9, min_mag), vmax=max_mag) if max_mag > 0 and min_mag > 0 and max_mag / min_mag > 100 else None
        im = ax.pcolormesh(T, F, magnitude, cmap='viridis', shading='gouraud', norm=norm)
        ax.set_xlabel("Time (t)"); ax.set_ylabel("Frequency (Hz)")
        min_freq = freqs[freqs > 0].min() if np.any(freqs > 0) else 1e-3; max_freq = freqs.max()
        if max_freq > min_freq: ax.set_ylim(min_freq, max_freq); ax.set_yscale('log') if max_freq / min_freq > 10 else None
        ax.set_title(f"CWT Scalogram (Magnitude)\n({os.path.splitext(filename)[0]})")
        divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.1); plt.colorbar(im, cax=cax, label='Magnitude |CWT|')
        plt.tight_layout()
        saved_path = _save_or_show_plot(fig, filepath, save_results)
        return saved_path
    except Exception as e: logger.error(f"Failed to generate CWT scalogram {filename}: {e}", exc_info=True); plt.close(fig); return None


def plot_correlation_dimension_fit(log_r, log_c, fit_coeffs, fit_indices=None, filename="corr_dim_fit.png", config_obj=None):
    """ Plots the log-log correlation sum C(r) vs r fit. Returns filepath if saved."""
    # Check debug flag first
    save_results = _get_config_value(config_obj, 'save_results', False)
    debug_plot = _get_config_value(config_obj, 'corr_dim_debug_plot', False)
    if not (save_results and debug_plot): return None

    plot_dir = _ensure_plot_dir(config_obj)
    filepath = os.path.join(plot_dir, filename)

    if not NOLDS_AVAILABLE: logger.warning(f"Cannot plot Corr Dim fit {filename}: nolds not available."); return None
    if log_r is None or log_c is None or fit_coeffs is None: logger.warning(f"Cannot plot Corr Dim fit {filename}: Missing input data."); return None

    logger.info(f"Generating Correlation Dimension fit plot ({filename})...")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(log_r, log_c, 'bo', markersize=4, label='Log(C(r)) vs Log(r) (Data)')
        if len(fit_coeffs) >= 2:
            slope, intercept = fit_coeffs[0], fit_coeffs[1]; r_fit = log_r # Default to full range
            if fit_indices is not None:
                 if isinstance(fit_indices, slice): r_fit = log_r[fit_indices]
                 elif isinstance(fit_indices, np.ndarray) and fit_indices.dtype == bool and len(fit_indices) == len(log_r): r_fit = log_r[fit_indices]
                 elif isinstance(fit_indices, np.ndarray) and fit_indices.dtype == int: valid_fit_indices = fit_indices[(fit_indices >= 0) & (fit_indices < len(log_r))]; r_fit = log_r[valid_fit_indices]
                 else: logger.warning(f"Unsupported fit_indices type: {type(fit_indices)}. Plotting fit over full range.")
            if r_fit.size > 0: ax.plot(r_fit, slope * r_fit + intercept, 'r--', lw=2, label=f'Fit (Slope={slope:.3f})')
            else: logger.warning(f"No points selected by fit_indices for Corr Dim fit plot {filename}.")
        else: logger.warning(f"Fit coefficients ({fit_coeffs}) insufficient for plotting linear fit.")
        ax.set_xlabel("Log(r)"); ax.set_ylabel("Log(C(r))"); ax.set_title(f"Correlation Dimension Fit\n({os.path.splitext(filename)[0]})")
        ax.legend(); ax.grid(True); plt.tight_layout()
        saved_path = _save_or_show_plot(fig, filepath, save_results)
        return saved_path
    except Exception as e: logger.error(f"Failed to generate Correlation Dimension fit plot {filename}: {e}", exc_info=True); plt.close(fig); return None


def plot_composite_final_states(results_dict, labels, filename="composite_final_states.png", config_obj=None):
    """ Compares final probability densities. Returns filepath if saved."""
    # Check flags first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_comparison = _get_config_value(config_obj, 'save_plots_comparison', False)
    if not (save_results and save_plots_comparison): return None

    plot_dir = _ensure_plot_dir(config_obj); filepath = os.path.join(plot_dir, filename)
    x_grid = _get_config_value(config_obj, 'x_grid', None)
    if x_grid is None and results_dict and labels:
         first_valid_label = next((l for l in labels if l in results_dict and results_dict[l]), None)
         if first_valid_label: x_grid = _get_config_value(results_dict[first_valid_label].get('config'), 'x_grid', None)
    if x_grid is None: logger.error(f"Cannot create composite final state plot {filename}: x_grid missing."); return None
    if not results_dict or not labels: logger.warning(f"Skipping composite final state plot {filename}: Missing results data or labels."); return None

    logger.info(f"Generating composite final state plot ({filename}) for labels: {labels}...")
    fig, ax = plt.subplots(figsize=(12, 7)); num_lines = len(labels); colors = plt.cm.viridis(np.linspace(0, 0.9, num_lines)); linestyles = ['-', '--', ':', '-.'] * (num_lines // 4 + 1)
    plotted_any = False
    for i, label in enumerate(labels):
        if label not in results_dict: logger.warning(f"Label '{label}' not found in results for composite plot {filename}. Skipping."); continue
        run_data = results_dict.get(label); obs = run_data.get('observables', {})
        if run_data is None or not run_data.get('success', False): logger.warning(f"Run '{label}' failed or missing data, skipping in composite plot {filename}."); continue
        psi_final_key = f'psi_final_{label}'; psi_final = obs.get(psi_final_key)
        if psi_final is not None and len(psi_final) == len(x_grid):
            prob_final = np.abs(psi_final)**2; clean_label = label.split('_', 1)[1] if '_' in label else label
            ax.plot(x_grid, prob_final, label=f'{clean_label}', color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], lw=1.5, alpha=0.8); plotted_any = True
        else: logger.warning(f"Final state '{psi_final_key}' missing or invalid for label '{label}' in composite plot {filename}. Skipping.")
    if not plotted_any: logger.error(f"No valid final states found to plot for {filename}."); plt.close(fig); return None
    ax.set_xlabel("Position (x)"); ax.set_ylabel(r"Final Probability Density $|\psi(x,T)|^2$"); ax.set_title("Comparison of Final Probability Densities")
    ax.legend(loc='best'); ax.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    saved_path = _save_or_show_plot(fig, filepath, save_results)
    return saved_path


def plot_composite_rqa(results_dict, labels, filename="composite_rqa.png", config_obj=None):
    """ Compares Recurrence Plots side-by-side. Returns filepath if saved."""
    # Check flags first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_comparison = _get_config_value(config_obj, 'save_plots_comparison', False)
    if not (save_results and save_plots_comparison): return None

    plot_dir = _ensure_plot_dir(config_obj)
    filepath = os.path.join(plot_dir, filename)
    if not RQA_AVAILABLE:
        logger.warning(f"Cannot create composite RQA plot {filename}: pyRQA not available.")
        return None

    # Filter labels for runs that exist, succeeded, and have RQA results
    valid_labels_data = {}
    for l in labels:
        if l in results_dict:
             rd = results_dict[l]
             # Check if run succeeded AND has the 'rqa_full_output' key
             if rd and rd.get('success') and 'rqa_full_output' in rd and rd['rqa_full_output']:
                 valid_labels_data[l] = rd
             else:
                 logger.debug(f"Skipping label '{l}' for composite RQA: Missing, failed, or no RQA results found in cache.")
        else:
            logger.debug(f"Skipping label '{l}' for composite RQA: Not found in results_dict.")


    if not valid_labels_data:
        logger.warning(f"Skipping composite RQA plot {filename}: No successful runs with valid RQA results found for labels: {labels}.")
        return None

    valid_labels = list(valid_labels_data.keys())
    logger.info(f"Generating composite RQA plot ({filename}) using pyRQA results for labels: {valid_labels}...")
    num_plots = len(valid_labels)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6.5), squeeze=False)
    axes = axes.flatten()
    plotted_any = False

    for i, label in enumerate(valid_labels):
        ax = axes[i]
        # --- Access data from 'rqa_full_output' ---
        rqa_output = valid_labels_data[label].get('rqa_full_output', {})
        rp_matrix = rqa_output.get('RP_Matrix')
        threshold_value = rqa_output.get('threshold_value', np.nan)
        det = rqa_output.get('DET', np.nan)
        # --- End Access ---

        if rp_matrix is None:
            logger.warning(f"RP_Matrix not found for label '{label}' in composite RQA plot {filename} (looked in rqa_full_output). Cannot plot matrix.")
            ax.set_title(f"{label}\n(RP Matrix Unavailable)")
            ax.text(0.5, 0.5, "Matrix Unavailable", ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if rp_matrix is not None and isinstance(rp_matrix, np.ndarray) and rp_matrix.shape[0] > 0:
            try:
                N = rp_matrix.shape[0]
                rp_plot_matrix = rp_matrix.astype(bool) if rp_matrix.dtype != bool else rp_matrix
                ax.imshow(rp_plot_matrix, cmap='binary', origin='lower', interpolation='nearest')
                clean_label = label.split('_', 1)[1] if '_' in label else label
                thresh_str = f"{threshold_value:.2e}" if not np.isnan(threshold_value) and (abs(threshold_value)<1e-2 or abs(threshold_value)>1e3) else f"{threshold_value:.2f}"
                det_str = f"{det:.3f}" if not np.isnan(det) else "N/A"
                ax.set_title(f"{clean_label}\n(Thr={thresh_str}, DET={det_str})")
                if i == 0:
                    ax.set_xlabel(f"Time Index i (0 to {N-1})")
                    ax.set_ylabel(f"Time Index j (0 to {N-1})")
                else:
                    ax.set_xlabel(f"Time Index i")
                    ax.set_yticklabels([])
                plotted_any = True
            except Exception as e:
                logger.error(f"Failed to plot recurrence matrix for label '{label}': {e}", exc_info=True)
                ax.set_title(f"{label}\n(Plotting Error)")
                ax.text(0.5, 0.5, "Plot Error", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            logger.warning(f"Recurrence Matrix invalid or empty for label '{label}' in composite plot {filename}.")
            clean_label = label.split('_', 1)[1] if '_' in label else label
            ax.set_title(f"{clean_label}\n(Invalid/Empty RP Matrix)")
            ax.text(0.5, 0.5, "Invalid/Empty Matrix", ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    if not plotted_any:
        logger.error(f"No valid recurrence plots found to plot for {filename}.")
        plt.close(fig)
        return None

    fig.suptitle("Comparison of Recurrence Plots (using pyRQA)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    saved_path = _save_or_show_plot(fig, filepath, save_results)
    return saved_path


def plot_composite_fft(results_dict, labels, observable_name, config_obj=None, filename="composite_fft.png"):
    """ Compares FFT amplitude spectra. Returns filepath if saved."""
    # Check flags first
    save_results = _get_config_value(config_obj, 'save_results', False)
    save_plots_comparison = _get_config_value(config_obj, 'save_plots_comparison', False)
    if not (save_results and save_plots_comparison): return None

    plot_dir = _ensure_plot_dir(config_obj); filepath = os.path.join(plot_dir, filename)
    dt = _get_config_value(config_obj, 'dt_quantum', None)
    if dt is None and results_dict and labels:
         first_valid_label = next((l for l in labels if l in results_dict and results_dict[l]), None)
         if first_valid_label: dt = _get_config_value(results_dict[first_valid_label].get('config'), 'dt_quantum', None)
    if dt is None: logger.error(f"Cannot create composite FFT plot {filename}: dt_quantum missing."); return None
    valid_labels_data = {l: rd for l, rd in results_dict.items() if l in labels and rd and rd.get('success')}
    if not results_dict or not valid_labels_data: logger.warning(f"Skipping composite FFT plot {filename}: Missing results data or valid labels."); return None

    valid_labels = list(valid_labels_data.keys()); logger.info(f"Generating composite FFT plot ({filename}) for observable '{observable_name}' and labels: {valid_labels}...")
    fig, ax = plt.subplots(figsize=(12, 7)); num_lines = len(valid_labels); colors = plt.cm.viridis(np.linspace(0, 0.9, num_lines)); linestyles = ['-', '--', ':', '-.'] * (num_lines // 4 + 1)
    plotted_any = False; max_freq_all = 0; min_freq_all = np.inf

    try: from ..validation.metrics import perform_fft_analysis # Import locally for use
    except ImportError: logger.error("Cannot import perform_fft_analysis. Skipping composite FFT plot."); return None

    for i, label in enumerate(valid_labels):
        obs = valid_labels_data[label].get('observables', {}); time_series = obs.get(observable_name)
        if time_series is not None and isinstance(time_series, np.ndarray) and len(time_series) > 1:
            freqs, spec = perform_fft_analysis(time_series, dt)
            if freqs is not None and spec is not None and len(freqs) > 1:
                clean_label = label.split('_', 1)[1] if '_' in label else label; non_dc_indices = freqs > 1e-9
                if np.any(non_dc_indices):
                    freqs_plot = freqs[non_dc_indices]; spec_plot = spec[non_dc_indices]
                    ax.plot(freqs_plot, spec_plot, label=f'{clean_label}', color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], lw=1.5, alpha=0.8)
                    plotted_any = True; max_freq_all = max(max_freq_all, freqs_plot.max()); min_freq_all = min(min_freq_all, freqs_plot.min())
                else: logger.warning(f"FFT for label '{label}' resulted only in DC component. Cannot plot on log scale.")
            else: logger.warning(f"FFT calculation failed or yielded insufficient components for label '{label}' in composite plot {filename}.")
        else: logger.warning(f"Time series for observable '{observable_name}' missing or invalid for label '{label}' in composite plot {filename}. Skipping.")

    if not plotted_any: logger.error(f"No valid FFT spectra found to plot for observable '{observable_name}' in {filename}."); plt.close(fig); return None
    ax.set_xlabel("Frequency (Hz or 1/time_unit)"); ax.set_ylabel("Amplitude Spectrum |FFT|")
    ax.set_title(f"Comparison of FFT Spectra for Observable: {observable_name}"); ax.set_yscale('log'); ax.set_xscale('log'); ax.grid(True, which='both', linestyle='--', alpha=0.6)
    if max_freq_all > 0 and min_freq_all < np.inf and max_freq_all > min_freq_all : ax.set_xlim(min_freq_all * 0.9, max_freq_all * 1.1)
    ax.legend(loc='best'); plt.tight_layout()
    saved_path = _save_or_show_plot(fig, filepath, save_results)
    return saved_path


# --- END OF FILE quantum_chaos_sim/visualization/plot_wavefunction.py ---