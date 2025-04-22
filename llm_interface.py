# quantum_chaos_sim/llm_interface.py
"""
Provides an abstracted interface for interacting with different Large Language Models (LLMs).
Supports OpenAI and Google Gemini models.
"""
import logging
import os
import time

# --- Optional Imports for LLM Providers ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- Local Config Import ---
# It's generally better practice to pass config values rather than importing the whole module
# but for simplicity in this context, we'll import it. Be mindful of potential circular dependencies.
try:
    from . import config as config_module
except ImportError:
    # Fallback if run standalone or config isn't available (less robust)
    class MockConfig:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_model = "gpt-4-turbo-preview"
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        gemini_model = "gemini-2.0-flash-lite"
    config_module = MockConfig()


logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _get_llm_config(provider, config_obj=None):
    """ Safely retrieves API key and model name for the given provider. """
    if config_obj is None:
         config_obj = config_module # Use imported config if none provided
    # Handle config_obj being a SimpleNamespace or a module-like object
    cfg_dict = vars(config_obj) if hasattr(config_obj, '__dict__') else config_obj.__dict__

    if provider == 'openai':
        api_key = cfg_dict.get('openai_api_key', os.environ.get("OPENAI_API_KEY"))
        model_name = cfg_dict.get('openai_model', "gpt-4-turbo-preview")
        if not api_key or api_key == "YOUR_API_KEY_NOT_FOUND":
            logger.error("OpenAI API key is missing or placeholder.")
            return None, None
        return api_key, model_name
    elif provider == 'gemini':
        api_key = cfg_dict.get('gemini_api_key', os.environ.get("GEMINI_API_KEY"))
        model_name = cfg_dict.get('gemini_model', "gemini-1.5-pro-latest")
        if not api_key or api_key == "YOUR_API_KEY_NOT_FOUND":
            logger.error("Gemini API key is missing or placeholder.")
            return None, None
        return api_key, model_name
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        return None, None

def _handle_llm_exception(e, provider):
    """ Logs and categorizes common LLM API errors. """
    if provider == 'openai' and OPENAI_AVAILABLE:
        if isinstance(e, openai.AuthenticationError):
             logger.error(f"OpenAI Authentication Failed: {e}")
             return "Error: OpenAI Authentication Failed. Check API key."
        elif isinstance(e, openai.RateLimitError):
             logger.error(f"OpenAI Rate Limit Exceeded: {e}. Consider adding retries with backoff.")
             return "Error: OpenAI Rate Limit Exceeded."
        elif isinstance(e, openai.BadRequestError):
             logger.error(f"OpenAI Bad Request Error: {e}. Check prompt length/content or model parameters.")
             return f"Error: OpenAI Bad Request. {e}"
        elif isinstance(e, openai.APITimeoutError):
             logger.error(f"OpenAI API Timeout Error: {e}. Consider increasing timeout or retrying.")
             return "Error: OpenAI API Timeout."
        elif isinstance(e, openai.APIConnectionError):
             logger.error(f"OpenAI API Connection Error: {e}. Check network connectivity.")
             return "Error: OpenAI API Connection Error."
        elif isinstance(e, openai.APIError):
             logger.error(f"General OpenAI API Error: {e}")
             return f"Error: OpenAI API Error. {e}"
    elif provider == 'gemini' and GEMINI_AVAILABLE:
         # Add specific Gemini error handling if library provides distinct types
         # Example (check actual exception types from google-generativeai):
         # from google.api_core import exceptions as google_exceptions
         # if isinstance(e, google_exceptions.Unauthenticated): ...
         # if isinstance(e, google_exceptions.ResourceExhausted): ... # Rate limit?
         # For now, use generic error message
         logger.error(f"Gemini API Error: {e}", exc_info=True)
         return f"Error: Gemini API Error. {e}"

    # Generic error
    logger.error(f"An unexpected error occurred with {provider} API: {e}", exc_info=True)
    return f"Error: Failed to get response from {provider} API. {e}"


# --- Main Interface Function ---

def get_llm_response(prompt: str, provider: str, model_name: str = None, api_key: str = None, config_obj=None, max_retries=1, retry_delay=5) -> str:
    """
    Sends a prompt to the specified LLM provider and returns the response.

    Args:
        prompt (str): The input prompt for the LLM.
        provider (str): The LLM provider ('openai' or 'gemini').
        model_name (str, optional): Specific model name to use. Defaults to config.
        api_key (str, optional): API key for the provider. Defaults to config/env.
        config_obj (object, optional): Configuration object/namespace. Defaults to imported config.
        max_retries (int): Maximum number of retries on potentially transient errors.
        retry_delay (int): Delay in seconds between retries.

    Returns:
        str: The text response from the LLM or an error message string.
    """
    if config_obj is None:
         config_obj = config_module # Use imported config if none provided

    config_api_key, config_model_name = _get_llm_config(provider, config_obj)

    key_to_use = api_key if api_key else config_api_key
    model_to_use = model_name if model_name else config_model_name

    if not key_to_use:
        return f"Error: API key for {provider} not found or configured."

    logger.info(f"Requesting LLM response from {provider} (Model: {model_to_use}).")
    logger.debug(f"Prompt length: {len(prompt)} chars.")

    # --- Safety Check for Prompt Length (Generic) ---
    # Provides a basic warning, specific models might have different limits.
    MAX_CHARS_WARN = 100000 # Warn if prompt is very long
    if len(prompt) > MAX_CHARS_WARN:
        logger.warning(f"Prompt length ({len(prompt)} chars) is large, potential risk of exceeding model context limit.")

    attempts = 0
    while attempts <= max_retries:
        attempts += 1
        try:
            # --- OpenAI ---
            if provider == 'openai':
                if not OPENAI_AVAILABLE: return "Error: OpenAI provider selected but 'openai' library not installed."
                client = openai.OpenAI(api_key=key_to_use)
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant analyzing physics simulation data."}, # Basic system prompt
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3, # Default temperature, consider making configurable
                    # max_tokens=... # Optional: Set max response length
                )
                result_text = response.choices[0].message.content
                # Log usage if available
                if hasattr(response, 'usage') and response.usage:
                    logger.info(f"OpenAI API Usage: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens} tokens.")
                elif hasattr(response, 'model_extra') and 'usage' in response.model_extra: # Pydantic v2 style
                     usage_info = response.model_extra['usage']
                     if usage_info and hasattr(usage_info, 'prompt_tokens'):
                         logger.info(f"OpenAI API Usage: Prompt={usage_info.prompt_tokens}, Completion={usage_info.completion_tokens}, Total={usage_info.total_tokens} tokens.")
                return result_text.strip() if result_text else "Error: Empty response received from OpenAI."

            # --- Gemini ---
            elif provider == 'gemini':
                if not GEMINI_AVAILABLE: return "Error: Gemini provider selected but 'google-generativeai' library not installed."
                genai.configure(api_key=key_to_use)
                # Adjust generation config as needed
                generation_config = {
                    "temperature": 0.3,
                    # "top_p": 1,
                    # "top_k": 1,
                    # "max_output_tokens": 2048, # Example
                }
                # Adjust safety settings if necessary (be cautious)
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
                model = genai.GenerativeModel(model_name=model_to_use,
                                              generation_config=generation_config,
                                              safety_settings=safety_settings)
                response = model.generate_content(prompt)

                # Handle potential blocks or errors in Gemini response
                if not response.candidates:
                     # Log response details if available
                     logger.error(f"Gemini response blocked or empty. Response details: {response}")
                     block_reason = "Unknown"
                     try: # Try to get block reason if structure allows
                          if response.prompt_feedback and response.prompt_feedback.block_reason:
                               block_reason = response.prompt_feedback.block_reason.name
                     except Exception: pass
                     return f"Error: Gemini response blocked or empty (Reason: {block_reason}). Check safety settings or prompt content."

                result_text = response.text
                # Gemini API doesn't directly expose token usage in the same way as OpenAI's ChatCompletion response object AFAIK
                # Usage might need separate tracking if required.
                logger.info(f"Gemini API call successful.")
                return result_text.strip() if result_text else "Error: Empty response received from Gemini."

            # --- Unknown Provider ---
            else:
                # Should have been caught by _get_llm_config, but double-check
                return f"Error: Unsupported LLM provider '{provider}'."

        except Exception as e:
            error_message = _handle_llm_exception(e, provider)
            # Check if error is potentially transient (e.g., rate limit, timeout, connection error)
            is_transient = any(keyword in error_message.lower() for keyword in ["rate limit", "timeout", "connection"])

            if is_transient and attempts <= max_retries:
                logger.warning(f"LLM API call failed (Attempt {attempts}/{max_retries+1}): {error_message}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue # Retry the loop
            else:
                logger.error(f"LLM API call failed permanently after {attempts} attempt(s): {error_message}")
                return error_message # Return the final error message

    # Should not be reached if loop logic is correct, but return error just in case
    return f"Error: LLM call failed after {max_retries+1} attempts."