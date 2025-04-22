# quantum_chaos_sim/orchestrator.py
"""
Defines the Orchestrator class that manages the AI-driven research cycle:
Plan -> Execute -> Analyze -> Report -> Refine.
"""

import logging
import os
import json
import time
import pandas as pd
from types import SimpleNamespace
import copy

# --- Local Imports ---
try:
    from . import agents
    from . import results_handler
    from . import main as main_module # Import main to call run_suite or similar
    from . import config as base_config_module
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    _import_error = e
    # Define dummy classes/functions if imports fail
    class MockAgents:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                logging.getLogger(__name__).error(f"Agents module unavailable: Cannot call {name}")
                if name == 'plan_experiments': return []
                if name == 'analyze_results': return "Error: Analysis Agent Unavailable"
                if name == 'write_report_sections': return {"Error": "Reporting Agent Unavailable"}
                return None
            return method
    agents = MockAgents()
    class MockMain:
         # Define a placeholder for run_suite if needed by orchestrator's execute logic
         def run_suite(*args, **kwargs):
              logging.getLogger(__name__).error(f"Main module unavailable: Cannot run suite")
              return None, {}
    main_module = MockMain()


logger = logging.getLogger(__name__)

if not MODULES_AVAILABLE:
    logger.critical(f"Orchestrator cannot function: Failed to import critical modules: {_import_error}")


class Orchestrator:
    """ Manages the AI-driven research workflow. """

    def __init__(self, config_obj):
        """
        Initializes the Orchestrator.

        Args:
            config_obj (SimpleNamespace): The base configuration object.
        """
        if not MODULES_AVAILABLE:
            raise ImportError("Orchestrator cannot initialize due to missing dependencies.")

        self.base_config = config_obj
        self.goal = getattr(config_obj, 'orchestration_goal', "Investigate quantum chaos.")
        self.max_iterations = getattr(config_obj, 'orchestration_max_iterations', 3)
        self.state_file = getattr(config_obj, 'orchestrator_state_file', None)

        # Initialize state (or load if possible)
        self.current_iteration = 0
        self.experiment_history = [] # List of (label, config_dict) tuples
        self.analysis_summary = "" # Cumulative text summary
        self.report_sections = {} # Dict of report sections
        self.results_df = pd.DataFrame() # Combined results DataFrame

        if self.state_file and os.path.exists(self.state_file):
            logger.info(f"Attempting to load orchestrator state from {self.state_file}")
            self.load_state()
        else:
            logger.info("Initializing new orchestrator state.")
            # Ensure state directory exists if we plan to save later
            if self.state_file:
                 os.makedirs(os.path.dirname(self.state_file), exist_ok=True)


    def run_main_loop(self):
        """ Executes the main Plan -> Execute -> Analyze -> Report -> Decide loop. """
        logger.info(f"--- Starting AI Orchestration Loop ---")
        logger.info(f"Goal: {self.goal}")
        logger.info(f"Max Iterations: {self.max_iterations}")
        logger.info(f"Current Iteration: {self.current_iteration}")

        while self.current_iteration < self.max_iterations:
            iteration_start_time = time.time()
            self.current_iteration += 1
            logger.info(f"\n===== Orchestration Iteration {self.current_iteration}/{self.max_iterations} =====")

            # 1. Plan Experiments
            logger.info("--- Phase: Planning Experiments ---")
            planned_experiments = agents.plan_experiments(
                goal=self.goal,
                history=self.experiment_history,
                analysis_summary=self.analysis_summary,
                config=self.base_config
            )

            if not planned_experiments:
                logger.warning("Planner Agent returned no experiments. Stopping loop.")
                break

            # 2. Execute Experiments
            logger.info(f"--- Phase: Executing {len(planned_experiments)} Experiments ---")
            batch_results_df, batch_full_results = self._execute_experiments(planned_experiments)

            if batch_results_df is None or batch_results_df.empty:
                logger.error("Execution phase failed or yielded no results. Stopping loop.")
                # Optionally save state here before breaking
                self.save_state()
                break

            # Update overall results DataFrame
            self.results_df = pd.concat([self.results_df, batch_results_df], ignore_index=True)

            # Update history (store config as dict for JSON serialization)
            for label, config_ns in planned_experiments:
                self.experiment_history.append((label, vars(config_ns)))

            # 3. Analyze Results
            logger.info("--- Phase: Analyzing Results ---")
            # Pass the DataFrame from the *current batch* for focused analysis
            self.analysis_summary = agents.analyze_results(
                results_df=batch_results_df, # Analyze only the latest batch
                goal=self.goal,
                previous_summary=self.analysis_summary, # Provide cumulative context
                config=self.base_config
            )
            logger.debug(f"Updated Analysis Summary:\n{self.analysis_summary[-500:]}...") # Log tail

            # 4. Write/Update Report
            logger.info("--- Phase: Writing Report Sections ---")
            self.report_sections = agents.write_report_sections(
                analysis_summary=self.analysis_summary, # Use cumulative summary
                goal=self.goal,
                config=self.base_config,
                history=self.experiment_history
            )
            # Optionally save the full report after each iteration
            self.save_report()

            # 5. Decide Next Step / Refine Goal (Basic Implementation)
            logger.info("--- Phase: Deciding Next Step ---")
            # Basic: Just check iteration count. Future: Use LLM to refine goal?
            if self.current_iteration >= self.max_iterations:
                logger.info("Maximum iterations reached.")
            else:
                logger.info("Proceeding to next iteration.")
                # Potential future logic:
                # refined_goal = agents.refine_goal(self.goal, self.analysis_summary)
                # if refined_goal: self.goal = refined_goal

            # Save state after each successful iteration
            self.save_state()
            iteration_end_time = time.time()
            logger.info(f"Iteration {self.current_iteration} finished ({iteration_end_time - iteration_start_time:.2f} seconds).")


        logger.info("--- AI Orchestration Loop Finished ---")
        # Final save of results DF (potentially redundant if saved in _execute)
        if not self.results_df.empty:
             logger.info("Saving final combined results DataFrame...")
             results_handler.save_results_df(
                 self.results_df,
                 self.base_config.results_dataframe_path, # Overwrite with combined results
                 format=self.base_config.results_storage_format,
                 key=self.base_config.hdf_key
             )
        self.save_report(final=True) # Save final report


    def _execute_experiments(self, experiments_to_run):
        """
        Executes a batch of experiments using main_module.run_suite.
        Handles saving/appending results to the main DataFrame file.

        Args:
            experiments_to_run (list): List of (label, config_obj) tuples.

        Returns:
            tuple: (pd.DataFrame, dict) - The DataFrame for the executed batch,
                   and the dictionary containing full results (if configured).
                   Returns (None, {}) on failure.
        """
        if not experiments_to_run:
            logger.warning("Orchestrator._execute_experiments: No experiments provided.")
            return None, {}

        # Define a temporary file path for this batch's results DF to avoid conflicts
        # We will merge it into the main DF later.
        batch_df_path = os.path.join(
            self.base_config.results_dir,
            f"batch_{self.current_iteration}_temp_results.{self.base_config.results_storage_format}"
        )
        batch_hdf_key = f"batch_{self.current_iteration}"

        logger.info(f"Executing batch of {len(experiments_to_run)} experiments (Iteration {self.current_iteration}). Temp save to: {batch_df_path}")

        # Call run_suite from main_module (or a dedicated execution function if preferred)
        # run_suite saves the DF of the runs it executes.
        # It returns the generated DataFrame and the full results dictionary.
        batch_df, full_results_dict = main_module.run_suite(
            experiment_configs_list=experiments_to_run,
            results_filepath=batch_df_path, # Save batch results separately first
            results_format=self.base_config.results_storage_format,
            hdf_key=batch_hdf_key # Use a batch-specific key if HDF5
        )

        # --- Optional: Clean up the temporary batch file ---
        # We already have the DataFrame in memory (batch_df), so we can remove the temp file.
        # Keep it for debugging if needed.
        # if os.path.exists(batch_df_path):
        #     try:
        #         os.remove(batch_df_path)
        #         logger.debug(f"Removed temporary batch DataFrame file: {batch_df_path}")
        #     except Exception as e:
        #         logger.warning(f"Could not remove temporary batch file {batch_df_path}: {e}")

        if batch_df is None:
            logger.error(f"run_suite failed for batch in iteration {self.current_iteration}.")
            return None, {}

        logger.info(f"Batch execution complete. Results obtained for {len(batch_df)} runs.")
        return batch_df, full_results_dict

    def save_report(self, final=False):
        """ Saves the current report sections to a file. """
        if not self.report_sections:
            logger.warning("No report sections generated yet to save.")
            return

        report_dir = self.base_config.report_dir
        os.makedirs(report_dir, exist_ok=True)
        filename = f"orchestrator_final_report.md" if final else f"orchestrator_report_iter_{self.current_iteration}.md"
        filepath = os.path.join(report_dir, filename)

        logger.info(f"Saving {'final' if final else 'intermediate'} report to {filepath}")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Orchestrated Simulation Report (Iteration {self.current_iteration if not final else 'Final'})\n\n")
                f.write(f"**Goal:** {self.goal}\n\n")
                # Order sections commonly found in reports
                order = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "Error"]
                for section in order:
                    if section in self.report_sections:
                        f.write(f"## {section}\n\n")
                        f.write(self.report_sections[section] + "\n\n")
                # Add any sections not in the standard order
                for section, content in self.report_sections.items():
                    if section not in order:
                         f.write(f"## {section}\n\n")
                         f.write(content + "\n\n")

        except Exception as e:
            logger.error(f"Failed to save report to {filepath}: {e}", exc_info=True)

    def save_state(self):
        """ Saves the orchestrator's current state to a JSON file. """
        if not self.state_file:
            logger.debug("Orchestrator state file not configured. Skipping state save.")
            return

        logger.info(f"Saving orchestrator state to {self.state_file}")
        state_data = {
            "goal": self.goal,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "experiment_history": self.experiment_history, # Stored as (label, config_dict)
            "analysis_summary": self.analysis_summary,
            "report_sections": self.report_sections,
            # Note: Saving the full results_df here might be large.
            # Consider saving only essential info or relying on the main results file.
            # For now, let's skip saving the DataFrame in the state JSON.
        }
        try:
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2) # Use indent for readability
            logger.info("Orchestrator state saved successfully.")
        except TypeError as e:
             logger.error(f"Failed to serialize orchestrator state to JSON: {e}. Check for non-serializable types.", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to save orchestrator state: {e}", exc_info=True)

    def load_state(self):
        """ Loads orchestrator state from a JSON file. """
        if not self.state_file or not os.path.exists(self.state_file):
            logger.warning(f"Orchestrator state file not found ({self.state_file}). Cannot load state.")
            return False

        logger.info(f"Loading orchestrator state from {self.state_file}")
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            self.goal = state_data.get("goal", self.goal)
            self.current_iteration = state_data.get("current_iteration", self.current_iteration)
            self.max_iterations = state_data.get("max_iterations", self.max_iterations)
            self.experiment_history = state_data.get("experiment_history", self.experiment_history)
            self.analysis_summary = state_data.get("analysis_summary", self.analysis_summary)
            self.report_sections = state_data.get("report_sections", self.report_sections)

            # Load the main results DataFrame associated with the saved state
            results_df_path = self.base_config.results_dataframe_path
            if os.path.exists(results_df_path):
                 logger.info(f"Loading associated results DataFrame from {results_df_path}")
                 self.results_df = results_handler.load_results_df(
                     results_df_path,
                     format=self.base_config.results_storage_format,
                     key=self.base_config.hdf_key
                 )
                 if self.results_df is None: self.results_df = pd.DataFrame() # Ensure it's a DF even if load fails
            else:
                 logger.warning(f"Results DataFrame file {results_df_path} not found. Initializing empty DataFrame.")
                 self.results_df = pd.DataFrame()

            logger.info(f"Orchestrator state loaded successfully. Resuming from iteration {self.current_iteration + 1}.")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from state file {self.state_file}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load orchestrator state: {e}", exc_info=True)
            return False