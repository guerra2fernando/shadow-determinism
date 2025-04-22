# quantum_chaos_sim/core/schrodinger.py
import numpy as np
import logging
# operators module doesn't need config access directly
from .operators import kinetic_operator_momentum_space

logger = logging.getLogger(__name__)

class SplitOperatorSolver:
    """
    Solves the Time-Dependent Schrödinger Equation using the Split-Operator FFT method.
    Assumes units where hbar=1, m=1 unless specified otherwise in the config object.
    Accepts a config object (e.g., SimpleNamespace) during initialization.
    """
    def __init__(self, config_obj):
        """
        Initializes the solver with parameters from the config object.

        Args:
            config_obj: An object (like SimpleNamespace) holding configuration
                        attributes (e.g., x_grid, k_grid, dt_quantum, hbar, m, N, dx).
        """
        # Validate necessary attributes exist in config_obj
        required_attrs = ['x_grid', 'k_grid', 'dt_quantum', 'hbar', 'm', 'N', 'dx']
        for attr in required_attrs:
            if not hasattr(config_obj, attr):
                raise AttributeError(f"SplitOperatorSolver requires attribute '{attr}' in the provided config object.")

        # Access attributes from the passed object
        self.x_grid = config_obj.x_grid
        self.k_grid = config_obj.k_grid
        self.dt = config_obj.dt_quantum
        self.hbar = config_obj.hbar
        self.m = config_obj.m
        self.N = config_obj.N
        self.dx = config_obj.dx # Store dx if needed for checks later

        # Pre-compute the kinetic energy propagator in momentum space for half a time step
        # This is exp(-i * T * (dt/2) / hbar)
        self.kinetic_propagator_half = self._calculate_kinetic_propagator(self.dt / 2.0)
        logger.info("SplitOperatorSolver initialized.")
        logger.debug(f"Solver params: N={self.N}, dx={self.dx:.4f}, dt={self.dt:.4e}")

        # Stability check heuristic (CFL-like condition for potential)
        # Vmax * dt / hbar << 1. This is a rough guide.
        # Actual stability depends complexly, but split-op is often quite stable.
        # We cannot check Vmax here as V is time-dependent. Check can be done in evolve loop if needed.

    def _calculate_kinetic_propagator(self, time_step):
        """
        Computes the kinetic energy propagator exp(-i * T * time_step / hbar)
        in momentum space.

        Args:
            time_step (float): The time duration for the propagation (e.g., dt/2).

        Returns:
            np.ndarray: The kinetic propagator array in k-space.
        """
        # T = p^2 / (2m) = (hbar*k)^2 / (2m)
        T_k = kinetic_operator_momentum_space(self.k_grid, self.hbar, self.m)
        propagator = np.exp(-1j * T_k * time_step / self.hbar)
        return propagator

    def _potential_propagator(self, V_xt, time_step):
        """
        Computes the potential energy propagator exp(-i * V(x, t) * time_step / hbar)
        in position space.

        Args:
            V_xt (np.ndarray): Potential V(x) at the current time t.
            time_step (float): The time duration for the propagation (e.g., dt/2).

        Returns:
            np.ndarray: The potential propagator array in x-space.
        """
        propagator = np.exp(-1j * V_xt * time_step / self.hbar)
        return propagator

    def evolve(self, psi, V_xt):
        """
        Performs one time step evolution using the Symmetric Split-Operator method:
        psi(t + dt) = exp(-iV dt/2) * exp(-iT dt) * exp(-iV dt/2) * psi(t)

        Args:
            psi (np.ndarray): Wavefunction at current time t (length N).
            V_xt (np.ndarray): Potential V(x, t) at current time t (length N).

        Returns:
            np.ndarray: Wavefunction at time t + dt.
        """
        # --- Step 1: Half step in potential ---
        # psi -> exp(-i * V * dt / (2*hbar)) * psi
        psi = self._potential_propagator(V_xt, self.dt / 2.0) * psi

        # --- Step 2: Full step in kinetic energy (in momentum space) ---
        # psi -> FFT(psi)
        psi_k = np.fft.fft(psi)

        # psi_k -> exp(-i * T_k * dt / hbar) * psi_k
        # We use two applications of the pre-computed half-step propagator:
        # exp(-i*T*dt/hbar) = exp(-i*T*dt/(2*hbar)) * exp(-i*T*dt/(2*hbar))
        psi_k = self.kinetic_propagator_half * psi_k # First half dt/2
        psi_k = self.kinetic_propagator_half * psi_k # Second half dt/2

        # psi_k -> IFFT(psi_k)
        psi = np.fft.ifft(psi_k)

        # --- Step 3: Half step in potential ---
        # psi -> exp(-i * V * dt / (2*hbar)) * psi
        psi = self._potential_propagator(V_xt, self.dt / 2.0) * psi

        # Optional: Re-normalize at each step (can help stability but violates exact unitarity)
        # norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        # if norm > 1e-10:
        #    psi /= norm

        return psi

    def initial_wavefunction(self, config_obj):
        """
        Generates the initial Gaussian wavepacket based on parameters
        in the provided config object.

        Args:
            config_obj: An object holding configuration attributes
                       (e.g., x0, sigma_psi, k0_psi).

        Returns:
            np.ndarray: The normalized initial wavefunction array.
        """
        # Validate necessary attributes
        required_attrs = ['x0', 'sigma_psi', 'k0_psi']
        for attr in required_attrs:
            if not hasattr(config_obj, attr):
                raise AttributeError(f"initial_wavefunction requires attribute '{attr}' in the config object.")

        # Access attributes from the passed object
        x = self.x_grid # Use grid from solver instance
        x0 = config_obj.x0
        sigma_psi = config_obj.sigma_psi
        k0_psi = config_obj.k0_psi

        # Create Gaussian profile centered at x0 with width sigma_psi
        gaussian_part = np.exp(-(x - x0)**2 / (2 * sigma_psi**2))
        # Add momentum component exp(i*k0*x)
        momentum_part = np.exp(1j * k0_psi * x)

        psi0 = gaussian_part * momentum_part

        # Normalize the initial wavefunction: integral |psi0|^2 dx = 1
        # Numerical integral: sum |psi0_i|^2 * dx = 1
        prob_density = np.abs(psi0)**2
        norm_sq = np.sum(prob_density) * self.dx # Use dx from solver instance
        if norm_sq < 1e-15: # Avoid division by zero or near-zero
             logger.warning("Initial wavefunction norm is close to zero before normalization.")
             # Return zero array or handle as error depending on desired behavior
             return np.zeros_like(psi0)

        norm = np.sqrt(norm_sq)
        psi0 /= norm

        # Verify norm after normalization (should be very close to 1)
        final_norm_sq = np.sum(np.abs(psi0)**2) * self.dx
        logger.info(f"Initial wavefunction created. Gaussian params: x0={x0}, sigma={sigma_psi}, k0={k0_psi}.")
        logger.info(f"Wavefunction normalized. Initial Norm^2 = {final_norm_sq:.6f}")

        return psi0