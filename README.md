# 👁️‍🗨️ Shadow Determinism - A Quantum Chaos Simulation Suite 

## 🔬 Overview

This project simulates a one-dimensional (1D) quantum particle subjected to various time-dependent potentials, including those modulated by classical chaotic systems (Lorenz, Rössler, Logistic Map), periodic signals, noise, and scheduled parameter changes. The primary goal is to investigate the emergence and characteristics of complexity and chaotic behavior in the quantum system under different driving conditions.

The simulation employs the accurate and stable **Split-Operator Fast Fourier Transform (FFT)** method to solve the Time-Dependent Schrödinger Equation (TDSE). It includes a comprehensive suite of analysis tools to quantify the system's dynamics using metrics from chaos theory and information theory.

Furthermore, the project explores the potential for controlling the quantum system and encoding information through parameter modulation (gating and embedding experiments). It also features an optional parallel simulation of a classical high-dimensional system (e.g., 4D coupled oscillators) to study quantum-classical correspondence. An experimental AI Orchestration layer allows for automated experiment planning and analysis using Large Language Models (LLMs).

## 👓 Features

*   **Quantum Simulation Core:**
    *   Solves the 1D TDSE using the Split-Operator FFT method.
    *   Models a Gaussian wavepacket initial state.
    *   Implements a dynamic potential `V(x,t)` with static, intrinsic oscillatory, and externally driven components.
*   **External Driving Signals:**
    *   **Chaotic:** Lorenz attractor, Rössler attractor, Logistic map.
    *   **Periodic/Quasi-periodic:** Sine wave, sum of two incommensurate sine waves.
    *   **Stochastic:** Filtered Gaussian or uniform noise (lowpass, highpass, bandpass).
    *   **Control:** Zero signal (constant potential), Constant signal.
*   **Classical Mechanics Simulation:**
    *   Optional parallel simulation of classical systems using `scipy.integrate.solve_ivp`.
    *   Models implemented: Coupled 4D Harmonic Oscillators, 4D Torus Flow, Rössler Hyperchaos.
    *   Capability to couple the classical system to the *quantum* driver signal `z(t)`.
*   **Parameter Scheduling:**
    *   Ability to define time-dependent schedules for key coupling parameters (`alpha`, `epsilon`).
    *   Includes pre-defined experiment generators for:
        *   **Gating:** Switching coupling parameters ON/OFF during simulation.
        *   **Embedding:** Modulating coupling parameters based on a binary message string.
*   **Observables & Analysis Metrics:**
    *   **Standard Quantum Observables:** Position `<x>`, Momentum `<p>`, Energy `<H>`, Norm Conservation, Spatial Variance `Var(x)`, Shannon Entropy `S(x)`.
    *   **Complexity/Chaos Metrics (Quantum & Classical):**
        *   Largest Lyapunov Exponent (LLE) using `nolds`.
        *   Recurrence Quantification Analysis (RQA) metrics (DET, LAM, ENTR, etc.) using `pyRQA`.
        *   Correlation Dimension (CorrDim) using `nolds`.
        *   Transfer Entropy (TE) using `pyinform` (Driver -> Quantum System).
    *   **Signal Analysis:** Fast Fourier Transform (FFT), Continuous Wavelet Transform (CWT) using `PyWavelets`.
    *   **State Comparison:** L2 norm distance between final quantum states of different runs.
*   **Experiment Management:**
    *   Systematically generate configurations for various experiments:
        *   Standard activation/control/sensitivity tests.
        *   Parametric sweeps (e.g., coupling strength `alpha`, `epsilon`).
        *   Lorenz Initial Condition (IC) sweeps (LHS, random, grid).
        *   Driver family comparisons.
        *   Repeatability tests (with optional noise injection).
        *   Sensitivity analysis (numerical Jacobian via finite differencing).
        *   Gating and Embedding schedule generation.
*   **Results Handling & Visualization:**
    *   Saves summarized run results (configuration, metrics) to HDF5 or CSV DataFrames using `pandas`.
    *   Optionally saves full time-series observable data to `.npy` files.
    *   Generates a wide range of plots using `matplotlib` and `seaborn`:
        *   *Per-Run:* Probability density heatmap, observable time series, driver trajectory, FFT spectrum, RQA plot, CWT scalogram, LLE/CorrDim debug fits (optional).
        *   *Meta-Analysis:* Parameter maps, UMAP/PCA embedding plots, Quantum vs. Classical metric comparisons, Repeatability metric distributions, Sensitivity Jacobian heatmap.
        *   *Comparison:* Composite final state plots, composite FFTs, composite RQA plots for selected runs.
        *   *Gating/Embedding:* Transient analysis plots, decoding results plots.
    *   Generates a Markdown plot manifest for easy navigation of saved figures.
*   **AI Orchestration (Experimental):**
    *   An `Orchestrator` class manages an automated research cycle: Plan -> Execute -> Analyze -> Report -> Refine.
    *   Uses LLMs (OpenAI GPT or Google Gemini via `llm_interface`) for:
        *   **Planning:** Suggesting the next logical experiment based on the goal and past results.
        *   **Analysis:** Interpreting quantitative results and generating textual summaries.
        *   **Reporting:** Drafting sections of a scientific report (Abstract, Intro, Methods, Results, Discussion, Conclusion).
    *   Saves and loads orchestration state for resuming runs.
*   **Configuration:**
    *   Centralized configuration file (`config.py`) using `SimpleNamespace`.
    *   Supports loading API keys from a `.env` file.

## Core Concepts Explored

*   **Quantum Chaos:** Investigating signatures of chaos (sensitivity, complexity metrics) in a quantum system driven by classically chaotic signals.
*   **Quantum-Classical Correspondence:** Comparing dynamics and complexity metrics between the quantum simulation and an optional classical analogue.
*   **Driven Quantum Systems:** Understanding the response of a quantum system to different types of time-dependent external forces.
*   **Information Flow:** Quantifying the influence of the external driver on the quantum system using Transfer Entropy.
*   **Sensitivity and Control:** Assessing the system's stability and potential for control via parameter modulation.
*   **Physical Layer Information Encoding:** Exploring the feasibility of embedding information into quantum dynamics.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd shadow-determinism
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `requirements.txt` should list packages like `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm`, `h5py`, `python-dotenv`.*

4.  **Install Optional Dependencies** (needed for full functionality):
    *   **Chaos Metrics (`nolds`):**
        ```bash
        pip install nolds
        ```
    *   **Recurrence Quantification Analysis (`pyRQA`):**
        ```bash
        pip install pyRQA
        ```
        *   **Important:** `pyRQA` often requires functional OpenCL drivers and potentially the `pyopencl` library (`pip install pyopencl`). Installation can be platform-dependent. Refer to [pyRQA](https://github.com/pik-copan/pyrq) and [PyOpenCL](https://documen.tician.de/pyopencl/) documentation if you encounter issues.
    *   **Information Flow (`pyinform`):**
        ```bash
        pip install pyinform
        ```
    *   **Wavelet Analysis (`PyWavelets`):**
        ```bash
        pip install PyWavelets
        ```
    *   **UMAP Embedding (`umap-learn`):**
        ```bash
        pip install umap-learn
        # umap-learn might have other dependencies like pynndescent
        ```
    *   **Clustering/RANSAC (`scikit-learn`):**
        ```bash
        pip install scikit-learn
        ```
    *   **Latin Hypercube Sampling (`pyDOE3`):** (Note: `pyDOE2` is older)
        ```bash
        pip install pyDOE3
        ```
    *   **LLM Interfaces:**
        ```bash
        pip install openai google-generativeai
        ```
    *   **Markdown Tables (`tabulate`):** (Used for nicer report formatting)
        ```bash
        pip install tabulate
        ```

5.  **API Keys (Optional - for LLM features):**
    *   Create a file named `.env` in the `quantum_chaos_sim` directory (alongside `config.py`).
    *   Add your API keys to this file:
        ```dotenv
        OPENAI_API_KEY="your_openai_key_here"
        GEMINI_API_KEY="your_gemini_key_here"
        ```
    *   The `config.py` file will automatically load these environment variables.

## Usage

1.  **Configure Simulation:**
    *   Edit `config.py` to set simulation parameters (grid size, time, initial state, potential parameters, driver type, analysis flags, LLM choices, etc.).
    *   Pay special attention to flags like `enable_classical_simulation`, `enable_parameter_scheduling`, `enable_openai_reporting`, `save_observable_arrays`, `enable_orchestration`, and plotting flags (`save_plots_per_run`, etc.).

2.  **Run the Simulation Suite:**
    *   Execute the main script from the parent directory (the one containing the `shadow_determinism` folder):
        ```bash
        python -m shadow-determinism.main
        ```
    *   This will:
        *   Generate experiment configurations based on settings in `config.py` and `experiment_manager.py`.
        *   Run the quantum simulations (and classical simulations if enabled).
        *   Perform enabled analyses (LLE, RQA, FFT, CWT, CorrDim, TE).
        *   Save results (DataFrame, plots, logs, optional observable arrays) to the `results/` directory.
        *   Generate meta-analysis plots and comparison plots if enabled.
        *   Optionally generate an AI-powered report summary using the configured LLM.
        *   Generate a plot manifest (`plot_manifest.md`).

3.  **Run AI Orchestration (Experimental):**
    *   In `config.py`, set `cfg.enable_orchestration = True`.
    *   Define the `cfg.orchestration_goal`.
    *   Configure LLM providers and API keys (`cfg.llm_choice`, `cfg.openai_api_key`, `cfg.gemini_api_key`).
    *   Run the main script as above:
        ```bash
        python -m shadow-determinism.main
        ```
    *   The orchestrator will run multiple iterations:
        *   **Plan:** LLM suggests experiments.
        *   **Execute:** Runs the suggested simulations.
        *   **Analyze:** Performs analysis and LLM interprets results.
        *   **Report:** LLM drafts report sections.
    *   Orchestrator state is saved (`orchestrator_state.json`) allowing resumption. Final combined results and reports are saved.

4.  **Explore Results:**
    *   Check the `results/` directory for:
        *   `simulation_results.h5` (or `.csv`): DataFrame with configuration and metrics for each run.
        *   `plots/`: Directory containing all generated plots. Use `plot_manifest.md` in `results/reports/` for guidance.
        *   `logs/`: Contains `simulation_suite.log` with detailed run information.
        *   `reports/`: Contains AI-generated reports (if enabled) and the plot manifest.
        *   `observables_data/`: Contains `.npy` files for saved observables (if enabled).
        *   `orchestrator_state/`: Contains the orchestrator state file (if orchestration used).


## Contributing

Contributions, bug reports, and feature requests are welcome! Please feel free to open an issue or submit a pull request on the GitHub repository.

## License

Apache 2.0

## Acknowledgements

*   This project utilizes libraries such as NumPy, SciPy, Matplotlib, Pandas, nolds, pyRQA, pyinform, PyWavelets, umap-learn, scikit-learn, and potentially OpenAI/Google Generative AI APIs.

---
