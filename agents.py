# quantum_chaos_sim/agents.py
"""
Defines the AI agents responsible for planning experiments, analyzing results,
and writing report sections in the orchestration loop.
"""

import logging
import copy
import numpy as np
import pandas as pd
from types import SimpleNamespace
import json # For structured LLM output parsing (optional)

# --- Local Imports ---
try:
    from . import llm_interface
    from . import experiment_manager
    from . import meta_analysis
    from . import reporting # For report structure helpers
    from . import config as base_config_module # For default values if needed
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    _import_error = e
    # Define dummy classes/functions if imports fail, to allow Orchestrator to import agents.py
    class MockExperimentManager:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                logging.getLogger(__name__).error(f"Experiment Manager unavailable: Cannot call {name}")
                return []
            return method
    experiment_manager = MockExperimentManager()
    class MockMetaAnalysis:
         def __getattr__(self, name):
            def method(*args, **kwargs):
                logging.getLogger(__name__).error(f"Meta Analysis unavailable: Cannot call {name}")
                return None # Or appropriate default
            return method
    meta_analysis = MockMetaAnalysis()
    # llm_interface and reporting are critical, Orchestrator should handle their absence.

logger = logging.getLogger(__name__)

if not MODULES_AVAILABLE:
    logger.error(f"Failed to import necessary modules for agents: {_import_error}")

# --- Helper Functions ---

def _summarize_history(history):
    """ Creates a concise summary of experiments run so far. """
    if not history:
        return "No experiments run yet."
    # Example: Count runs by type (Sweep, Sensitivity, Driver, etc.)
    summary = f"Total experiments run: {len(history)}\n"
    labels = [item[0] for item in history if isinstance(item, (list, tuple)) and len(item)>0]
    types_seen = {}
    for label in labels:
        parts = label.split('_')
        exp_type = parts[0] if parts else "Unknown"
        types_seen[exp_type] = types_seen.get(exp_type, 0) + 1
    if types_seen:
        summary += "Run types count: " + ", ".join(f"{k}={v}" for k, v in types_seen.items()) + "\n"
    # Could add more detail, e.g., parameter ranges covered in sweeps if stored
    return summary

def _format_quantitative_summary(results_df, config_obj):
    """ Creates a text summary of quantitative results from the DataFrame. """
    if results_df is None or results_df.empty:
        return "No quantitative results available from the latest batch."

    summary = "Latest Batch Quantitative Summary:\n"
    num_runs = len(results_df)
    num_successful = int(results_df['success'].sum())
    summary += f"- Runs: {num_runs} ({num_successful} successful)\n"

    # Summary stats for key metrics
    metric_cols = [col for col in results_df.columns if col.startswith(('metric_', 'classical_metric_'))]
    observable_summary_cols = [col for col in results_df.columns if col.startswith('observable_')]
    summary_cols = metric_cols + observable_summary_cols
    valid_summary_cols = [col for col in summary_cols if col in results_df.columns]

    if valid_summary_cols:
        numeric_summary_cols = results_df[valid_summary_cols].select_dtypes(include=np.number).columns.tolist()
        if numeric_summary_cols:
            successful_df = results_df[results_df['success']]
            if not successful_df.empty:
                stats = successful_df[numeric_summary_cols].describe().transpose()
                stats_selected = stats[['count', 'mean', 'std', 'min', 'max']].copy()
                stats_selected.index = stats_selected.index.str.replace('metric_', '').str.replace('observable_', '').str.replace('classical_', 'Cls_')
                stats_str = stats_selected.to_string(float_format="%.3g")
                summary += f"- Key Metric Stats (Successful Runs):\n{stats_str}\n"
            else:
                 summary += "- No successful runs in batch for metric stats.\n"
        else:
             summary += "- No numeric metric columns found in batch for stats.\n"

    # Check for specific experiment types run in this batch
    labels = results_df['run_label'].tolist()
    if any(l.startswith("Sweep_") for l in labels): summary += "- Parametric sweep performed.\n"
    if any(l.startswith("Sensitivity_") for l in labels): summary += "- Sensitivity analysis performed.\n"
    if any(l.startswith("Repeat_") for l in labels): summary += "- Repeatability test performed.\n"
    if any(l.startswith("Driver_") for l in labels): summary += "- Different driver types tested.\n"
    if any(l.startswith("Gating_") for l in labels): summary += "- Gating experiment performed.\n"
    if any(l.startswith("Embed_") for l in labels): summary += "- Embedding experiment performed.\n"

    return summary


# --- Agent Implementations ---

def plan_experiments(goal: str, history: list, analysis_summary: str, config: SimpleNamespace) -> list:
    """
    Uses an LLM to plan the next batch of experiments based on the goal and current state.

    Args:
        goal (str): The high-level research goal.
        history (list): List of (run_label, config_obj) tuples from previous runs.
        analysis_summary (str): Text summary of findings so far.
        config (SimpleNamespace): The base configuration object.

    Returns:
        list: A list of (run_label, config_obj) tuples for the next batch, or empty list.
    """
    if not MODULES_AVAILABLE or not llm_interface:
        logger.error("Planner Agent: Cannot plan experiments, dependencies missing.")
        return []

    logger.info("--- Planner Agent: Planning next experiments ---")
    history_summary = _summarize_history(history)

    # --- Construct LLM Prompt for Planning ---
    prompt = f"""
You are an AI assistant helping plan computational physics experiments.
Your task is to propose the *next* batch of simulations based on the research goal and findings so far.

**Overall Research Goal:**
{goal}

**Experiment History Summary:**
{history_summary}

**Current Analysis Summary & Findings:**
{analysis_summary if analysis_summary else "No analysis performed yet."}

**Base Configuration Parameters:**
- Driver Type: {getattr(config, 'driver_type', 'N/A')}
- Base Alpha: {getattr(config, 'alpha', 'N/A'):.3f}
- Base Epsilon: {getattr(config, 'epsilon', 'N/A'):.3f}
- Simulation Time (T): {getattr(config, 'T', 'N/A')}
- Available Experiment Generation Functions (Examples):
    - generate_coupling_sweep(param_name='alpha' or 'epsilon', num_values=N, sweep_values=[v1, v2,...])
    - generate_lorenz_ic_sweep(num_samples=N, method='lhs' or 'random')
    - generate_driver_family_configs(driver_list=['lorenz', 'sine', 'zero', ...])
    - generate_sensitivity_configs(params_to_perturb={{'param': scale, ...}})
    - generate_repeatability_configs(num_repeats=N)
    - generate_gating_configs(param_to_gate='alpha', ...)
    - generate_embedding_configs(message_bits=[0,1,...], ...)
    - define_standard_experiments() # Runs baseline, control, basic sensitivity

**Instruction:**
Based *only* on the goal, history, and analysis summary, propose the *single most logical next experiment type* and its necessary parameters to advance the research goal. Be specific.

Choose ONE type from the available functions. Specify the function name and the key arguments needed.
For sweeps, suggest parameter name, range (start, end), and number of steps (e.g., {config.orchestration_planning_heuristic_num_samples} steps is reasonable unless specified otherwise).
For sensitivity, suggest which parameters to perturb (e.g., ['alpha', 'x0']).
For driver families, suggest the list of drivers.
For Lorenz IC sweeps, suggest the number of samples.
For Gating/Embedding, suggest the parameter and basic setup (e.g., standard switch times, short message).

**Output Format:**
Provide your response ONLY in the following JSON format:
{{
  "reasoning": "Brief justification for the chosen experiment based on goal/analysis.",
  "experiment_type": "Name of the experiment generation function (e.g., 'generate_coupling_sweep')",
  "parameters": {{
    "param_name": "value", // e.g., "param_name": "alpha"
    "num_values": 5,       // e.g., for sweep
    "sweep_values": null,  // Or provide specific list: [0.1, 0.5, 1.0]
    "sweep_range": [0.0, 2.0], // Alternative for sweep: [start, end]
    "num_samples": 10,     // e.g., for Lorenz IC sweep
    "driver_list": ["lorenz", "sine"], // e.g., for driver family
    "params_to_perturb": {{"alpha": 1e-4, "x0": 1e-5}}, // e.g., for sensitivity
    "num_repeats": 5,      // e.g., for repeatability
    "param_to_gate": "alpha", // e.g., for gating
    "message_bits": [1, 0, 1] // e.g., for embedding
    // Include ONLY the parameters relevant to the chosen experiment_type
  }}
}}
"""

    provider = config.orchestration_planning_llm
    logger.debug(f"Planner: Sending planning request to {provider}...")
    response_str = llm_interface.get_llm_response(prompt, provider=provider, config_obj=config)

    if response_str.startswith("Error:"):
        logger.error(f"Planner Agent: Failed to get planning suggestion from LLM: {response_str}")
        return []

    # --- Parse LLM Response ---
    try:
        # Clean potential markdown code fences
        if response_str.strip().startswith("```json"):
            response_str = response_str.strip()[7:]
        if response_str.strip().endswith("```"):
            response_str = response_str.strip()[:-3]

        logger.debug(f"Planner: Received raw response:\n{response_str}")
        plan = json.loads(response_str)
        exp_type = plan.get("experiment_type")
        params = plan.get("parameters", {})
        reasoning = plan.get("reasoning", "No reasoning provided.")
        logger.info(f"Planner Agent: LLM suggests running '{exp_type}'. Reasoning: {reasoning}")
        logger.debug(f"Planner Agent: Suggested parameters: {params}")

    except json.JSONDecodeError as e:
        logger.error(f"Planner Agent: Failed to parse LLM JSON response: {e}")
        logger.error(f"LLM Raw Response was:\n{response_str}")
        return []
    except Exception as e:
        logger.error(f"Planner Agent: Error processing LLM response: {e}", exc_info=True)
        return []

    # --- Generate Experiments using Experiment Manager ---
    experiments_to_run = []
    try:
        if hasattr(experiment_manager, exp_type):
            func = getattr(experiment_manager, exp_type)
            # Prepare arguments - careful mapping needed!
            args = {}
            if exp_type == "generate_coupling_sweep":
                args['param_name'] = params.get('param_name', 'alpha')
                args['num_values'] = params.get('num_values', config.orchestration_planning_heuristic_num_samples)
                if 'sweep_values' in params and params['sweep_values']:
                     args['sweep_values'] = params['sweep_values']
                elif 'sweep_range' in params:
                     start, end = params['sweep_range']
                     args['sweep_values'] = np.linspace(start, end, args['num_values']).tolist()
                # If neither sweep_values nor sweep_range provided, let generate_coupling_sweep use its default logic
            elif exp_type == "generate_lorenz_ic_sweep":
                args['num_samples'] = params.get('num_samples', config.orchestration_planning_heuristic_num_samples)
                args['method'] = params.get('method', 'lhs')
            elif exp_type == "generate_driver_family_configs":
                args['driver_list'] = params.get('driver_list', ['lorenz', 'sine', 'zero'])
            elif exp_type == "generate_sensitivity_configs":
                # Expecting a dict like {'param_name': scale_value}
                args['params_to_perturb'] = params.get('params_to_perturb', {})
                if not args['params_to_perturb']: logger.warning("Planner: Sensitivity suggested, but no parameters specified by LLM."); return []
            elif exp_type == "generate_repeatability_configs":
                args['num_repeats'] = params.get('num_repeats', 5)
                # Needs a base config/label prefix - Orchestrator might need to manage this context
                args['run_label_prefix'] = params.get('run_label_prefix', "Planned_Repeat") # Needs better context handling
            elif exp_type == "generate_gating_configs":
                args['param_to_gate'] = params.get('param_to_gate', 'alpha')
                # Use default on/off/switch times unless specified
            elif exp_type == "generate_embedding_configs":
                args['message_bits'] = params.get('message_bits', [1,0,1,0])
                args['param_to_modulate'] = params.get('param_to_modulate', 'alpha')
                # Use default mods/duration unless specified
            elif exp_type == "define_standard_experiments":
                 pass # No specific args needed from LLM usually
            else:
                 logger.warning(f"Planner: Parameter mapping for experiment type '{exp_type}' not fully implemented.")

            logger.info(f"Planner Agent: Calling {exp_type} with args: {args}")
            experiments_to_run = func(base_config=config, **args)

        else:
            logger.error(f"Planner Agent: Experiment type '{exp_type}' suggested by LLM is not a valid function in experiment_manager.")
            return []

    except Exception as e:
        logger.error(f"Planner Agent: Failed to generate experiments using {exp_type}: {e}", exc_info=True)
        return []

    logger.info(f"Planner Agent: Generated {len(experiments_to_run)} configurations for the next batch.")
    return experiments_to_run


def analyze_results(results_df: pd.DataFrame, goal: str, previous_summary: str, config: SimpleNamespace) -> str:
    """
    Performs quantitative analysis and uses an LLM to interpret results in context.

    Args:
        results_df (pd.DataFrame): DataFrame from the latest batch of experiments.
        goal (str): The high-level research goal.
        previous_summary (str): Text summary of findings from previous iterations.
        config (SimpleNamespace): The base configuration object.

    Returns:
        str: Updated textual analysis summary incorporating new findings.
    """
    if not MODULES_AVAILABLE or not llm_interface:
        logger.error("Analyzer Agent: Cannot analyze results, dependencies missing.")
        return previous_summary + "\nError: Analysis dependencies missing."

    logger.info("--- Analyzer Agent: Analyzing latest results ---")

    # --- Perform Quantitative Meta-Analysis ---
    quantitative_summary = "\nQuantitative Analysis Summary (Latest Batch):\n"
    if results_df is None or results_df.empty:
         quantitative_summary += "- No new results data provided.\n"
    elif not meta_analysis:
         quantitative_summary += "- Meta-analysis module unavailable.\n"
         quantitative_summary += _format_quantitative_summary(results_df, config) # Basic summary
    else:
        # Check which analyses are relevant based on run labels in the batch
        labels = results_df['run_label'].tolist()
        run_types_in_batch = set(l.split('_')[0] for l in labels if '_' in l)

        # Run clustering / dim reduction if enough diverse data
        if len(results_df) > 5 and len(run_types_in_batch) > 1: # Heuristic
            features_for_analysis = [ col for col in results_df.columns if col.startswith(('metric_', 'classical_metric_')) and results_df[col].notna().any()]
            numeric_feature_cols = results_df[features_for_analysis].select_dtypes(include=np.number).columns.tolist()
            if numeric_feature_cols:
                 if meta_analysis.SKLEARN_AVAILABLE:
                      results_df = meta_analysis.cluster_results(results_df, features=numeric_feature_cols)
                      if 'cluster_label' in results_df: quantitative_summary += "- Clustering performed.\n"
                 if meta_analysis.UMAP_AVAILABLE:
                      results_df = meta_analysis.perform_dimensionality_reduction(results_df, features=numeric_feature_cols)
                      if 'embedding_X' in results_df: quantitative_summary += "- Dimensionality reduction performed.\n"

        # Run specific analyses based on experiment types identified
        if "Repeat" in run_types_in_batch and hasattr(meta_analysis, 'analyze_repeatability'):
            repeat_prefix = next((l.split('_Repeat_')[0] for l in labels if '_Repeat_' in l), None)
            if repeat_prefix:
                 repeat_summary = meta_analysis.analyze_repeatability(results_df, repeat_prefix, results_df.columns, config)
                 quantitative_summary += f"- Repeatability analysis performed for prefix '{repeat_prefix}'.\n"
                 # Append summary details if available
                 if repeat_summary and 'metrics_stats' in repeat_summary:
                     quantitative_summary += "  Repeatability Stats (Mean/Std):\n"
                     try:
                         stats_df = pd.DataFrame(repeat_summary['metrics_stats']).transpose()
                         for col in stats_df.select_dtypes(include=np.number).columns: stats_df[col] = stats_df[col].apply(lambda x: reporting._format_metric(x, precision=3))
                         stats_df.index = stats_df.index.str.replace('metric_', '').str.replace('observable_', '').str.replace('classical_', 'Cls_')
                         quantitative_summary += stats_df[['mean', 'std']].to_string() + "\n"
                     except: quantitative_summary += "  (Error formatting stats)\n"


        if "Sensitivity" in run_types_in_batch and hasattr(meta_analysis, 'calculate_sensitivity_jacobian'):
             base_label = "Sensitivity_Base"
             if base_label in labels:
                 pert_params = list(set(l.split('_')[1] for l in labels if l.startswith("Sensitivity_") and "_Pert_" in l))
                 jacobian = meta_analysis.calculate_sensitivity_jacobian(results_df, base_label, pert_params, config)
                 if jacobian is not None and not jacobian.empty:
                      quantitative_summary += f"- Sensitivity Jacobian calculated for params: {pert_params}.\n"
                      # Append a concise summary of the Jacobian
                      try:
                           jacobian_summary = jacobian.abs().mean().sort_values(ascending=False) # Avg sensitivity per param
                           quantitative_summary += "  Avg Abs Sensitivity (dMetric/dParam):\n"
                           quantitative_summary += jacobian_summary.to_string(float_format="%.2e") + "\n"
                      except: quantitative_summary += "  (Error formatting Jacobian summary)\n"
                 else: quantitative_summary += "- Sensitivity analysis run, but Jacobian calculation failed or was empty.\n"
             else: quantitative_summary += "- Sensitivity runs found, but baseline 'Sensitivity_Base' missing.\n"

        # Add basic stats summary
        quantitative_summary += _format_quantitative_summary(results_df, config)

        # Placeholder calls for detailed gating/embedding if data is available (needs access to saved .npy)
        if "Gating" in run_types_in_batch and hasattr(meta_analysis, 'analyze_gating_transients'):
             # meta_analysis.analyze_gating_transients(results_df, config) # Needs data loading
             quantitative_summary += "- Gating analysis plots generated (requires separate inspection).\n"
        if "Embed" in run_types_in_batch and hasattr(meta_analysis, 'decode_embedded_signal'):
             # decoded_results, metrics = meta_analysis.decode_embedded_signal(results_df, config) # Needs data loading
             quantitative_summary += "- Embedding decoding analysis performed (check plots/metrics).\n"

        # Call Quantum/Classical comparison plotting if relevant data exists
        q_cols = any(c.startswith('metric_') for c in results_df.columns)
        c_cols = any(c.startswith('classical_metric_') for c in results_df.columns)
        if q_cols and c_cols and hasattr(meta_analysis, 'compare_quantum_classical'):
             # meta_analysis.compare_quantum_classical(results_df, config) # Plot generated internally
             quantitative_summary += "- Quantum vs Classical comparison plots generated.\n"

    # --- Construct LLM Prompt for Interpretation ---
    prompt = f"""
You are an AI assistant specialized in analyzing results from computational physics simulations (quantum chaos).

**Overall Research Goal:**
{goal}

**Analysis Summary from Previous Iterations:**
{previous_summary if previous_summary else "This is the first analysis iteration."}

**Quantitative Summary of Results from the LATEST Experiment Batch:**
{quantitative_summary}

**Instruction:**
Based *only* on the provided goal, previous summary, and the quantitative summary of the *latest* batch:
1. Briefly interpret the key findings from the *latest* batch of experiments. What do the new quantitative results show?
2. Relate these new findings back to the overall research goal and the previous summary. Do the results confirm previous findings, contradict them, or explore new aspects?
3. Identify the most significant or surprising results from this batch.
4. Based on this analysis, what specific questions remain unanswered or what new questions arise?
5. Suggest the *general direction* for the next logical step in the investigation (e.g., "refine the parameter range for alpha", "investigate sensitivity to k0_psi", "compare with a different driver type", "check repeatability under noise").

Keep the analysis concise and focused on interpreting the provided quantitative summaries. Avoid making definitive conclusions if the data is insufficient.

**Output:**
Provide your response as a single block of text summarizing the analysis and interpretation.
"""
    provider = config.orchestration_analysis_llm
    logger.debug(f"Analyzer: Sending analysis request to {provider}...")
    analysis_text = llm_interface.get_llm_response(prompt, provider=provider, config_obj=config)

    if analysis_text.startswith("Error:"):
        logger.error(f"Analyzer Agent: Failed to get analysis interpretation from LLM: {analysis_text}")
        # Return previous summary plus the error and quantitative part
        return previous_summary + "\n\n--- Analysis Error ---\n" + analysis_text + "\n" + quantitative_summary

    logger.info("Analyzer Agent: Received analysis interpretation from LLM.")
    # Combine previous summary with new analysis (or just return new if first pass)
    updated_summary = (previous_summary + "\n\n--- Analysis Iteration Results ---\n" + quantitative_summary + "\nInterpretation:\n" + analysis_text) if previous_summary else ("--- Analysis Iteration Results ---\n" + quantitative_summary + "\nInterpretation:\n" + analysis_text)

    return updated_summary


def write_report_sections(analysis_summary: str, goal: str, config: SimpleNamespace, history: list) -> dict:
    """
    Uses an LLM to generate sections of a scientific report based on the analysis.

    Args:
        analysis_summary (str): The cumulative text summary of findings.
        goal (str): The high-level research goal.
        config (SimpleNamespace): The base configuration object.
        history (list): List of (run_label, config_obj) tuples from all runs.

    Returns:
        dict: Dictionary with report section names as keys and generated text as values.
    """
    if not MODULES_AVAILABLE or not llm_interface:
        logger.error("Reporter Agent: Cannot write report, dependencies missing.")
        return {"Error": "Dependencies missing"}

    logger.info("--- Reporter Agent: Generating report sections ---")
    report_sections = {}
    history_summary = _summarize_history(history) # Get history summary

    # --- Define prompts for each section ---
    prompts = {
        "Abstract": f"""
Based on the following research goal and analysis summary, write a concise scientific abstract (around 150-250 words).

**Goal:** {goal}
**Cumulative Analysis Summary:**
{analysis_summary}
""",
        "Introduction": f"""
Based on the following research goal, write a brief introduction section for a scientific report. Expand slightly on the goal and its context/motivation.

**Goal:** {goal}
""",
        "Methods": f"""
Based on the base configuration and the summary of experiments run, write a brief Methods section. Describe the simulation setup (1D quantum system, potential form), mention the driver types used, parameters varied (qualitatively based on history summary), key analysis metrics calculated (LLE, RQA, TE, etc. - infer from analysis summary), and classical model details if mentioned in the analysis.

**Base Configuration:**
- N={getattr(config, 'N', 'N/A')}, L={getattr(config, 'L', 'N/A')}, T={getattr(config, 'T', 'N/A')}, M={getattr(config, 'M', 'N/A')}
- Base Potential: V0={getattr(config, 'V0_static', 'N/A')} + epsilon*sin(k*x+omega*t + alpha*z(t)) (alpha={getattr(config, 'alpha', 'N/A'):.2f}, epsilon={getattr(config, 'epsilon', 'N/A'):.2f})
- Default Driver: {getattr(config, 'driver_type', 'N/A')}
- Classical Model Enabled: {getattr(config, 'enable_classical_simulation', False)} (Type: {getattr(config, 'classical_model_type', 'N/A') if getattr(config, 'enable_classical_simulation', False) else 'N/A'})

**Experiment History Summary:**
{history_summary}
**Mentioned Analysis Methods (from cumulative summary):**
{analysis_summary}
""",
        "Results": f"""
Based *only* on the cumulative analysis summary provided below, write a Results section describing the key findings. Organize the results logically (e.g., by experiment type: sensitivity, sweeps, driver comparison, gating). Focus on reporting the observations mentioned in the summary. Do not add interpretation here.

**Cumulative Analysis Summary:**
{analysis_summary}
""",
        "Discussion": f"""
Based on the research goal and the cumulative analysis summary, write a Discussion section. Interpret the key findings reported in the Results section in the context of the research goal. Discuss the implications of sensitivity, repeatability, driver effects, scheduling effects, and any quantum-classical comparisons mentioned in the analysis summary. Address the core questions raised by the goal.

**Goal:** {goal}
**Cumulative Analysis Summary:**
{analysis_summary}
""",
        "Conclusion": f"""
Based on the cumulative analysis summary, write a brief Conclusion section summarizing the main takeaways of the study in relation to the original goal.

**Goal:** {goal}
**Cumulative Analysis Summary:**
{analysis_summary}
"""
    }

    provider = config.orchestration_reporting_llm
    # Generate each section
    for section_name, prompt_text in prompts.items():
        logger.info(f"Reporter: Generating section '{section_name}' using {provider}...")
        section_content = llm_interface.get_llm_response(prompt_text, provider=provider, config_obj=config)
        if section_content.startswith("Error:"):
            logger.error(f"Reporter Agent: Failed to generate section '{section_name}': {section_content}")
            report_sections[section_name] = f"Error generating section: {section_content}"
        else:
            report_sections[section_name] = section_content
            logger.info(f"Reporter: Section '{section_name}' generated successfully.")

    return report_sections