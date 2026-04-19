from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

from .validation import (
    validate_and_default_float_range,
    validate_and_default_positive_int,
    validate_and_default_string,
    validate_and_default_iterable,
    log_parameter_correction
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentConfig:
    temperature: float = 0.0
    max_context_chunks: int = 5
    
    def __post_init__(self) -> None:
        # Validate temperature with default substitution
        validated_temperature = validate_and_default_float_range(
            self.temperature, 0.0, 0.0, 2.0, "temperature", "AgentConfig"
        )
        if validated_temperature != self.temperature:
            object.__setattr__(self, 'temperature', validated_temperature)
            log_parameter_correction(self.temperature, validated_temperature, "temperature", "AgentConfig", "out of range")
        
        # Validate max_context_chunks with default substitution
        validated_max_chunks = validate_and_default_positive_int(
            self.max_context_chunks, 5, "max_context_chunks", "AgentConfig"
        )
        if validated_max_chunks != self.max_context_chunks:
            object.__setattr__(self, 'max_context_chunks', validated_max_chunks)
            log_parameter_correction(self.max_context_chunks, validated_max_chunks, "max_context_chunks", "AgentConfig", "invalid value")


def build_prompt(question: str, contexts: Iterable[str]) -> str:
    """
    Build prompt with parameter validation and default substitution.
    
    Args:
        question: Question string
        contexts: Iterable of context strings
        
    Returns:
        Formatted prompt string
    """
    # Validate question parameter
    original_question = question
    validated_question = validate_and_default_string(
        question, "What is the answer?", "question", "Agent.build_prompt", allow_empty=False
    )
    if validated_question != original_question:
        log_parameter_correction(original_question, validated_question, "question", "Agent.build_prompt", "invalid question")
    
    # Validate contexts parameter
    original_contexts = contexts
    validated_contexts = validate_and_default_iterable(
        contexts, [], "contexts", "Agent.build_prompt", allow_empty=True
    )
    if validated_contexts != list(original_contexts):
        log_parameter_correction(original_contexts, validated_contexts, "contexts", "Agent.build_prompt", "invalid contexts")
    
    # Validate each context string
    clean_contexts = []
    for i, context in enumerate(validated_contexts):
        validated_context = validate_and_default_string(
            context, f"No context available {i}", f"context[{i}]", "Agent.build_prompt", allow_empty=True
        )
        if validated_context != context:
            log_parameter_correction(context, validated_context, f"context[{i}]", "Agent.build_prompt", "invalid context string")
        clean_contexts.append(validated_context)
    
    context_block = "\n\n".join(clean_contexts)
    return (
        "You are a legal reasoning agent.\n"
        "Use only the provided context to answer the question.\n"
        "Return a concise answer.\n\n"
        f"Question:\n{validated_question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )


def parse_model_output(text: str | None) -> str:
    """
    Parse and clean model output for consistent formatting.
    
    Handles:
    - None or non-string inputs gracefully
    - Removes extra whitespace (leading, trailing, multiple spaces)
    - Normalizes line breaks and special characters
    - Ensures consistent text formatting
    
    Args:
        text: Raw model output text (can be None or non-string)
        
    Returns:
        Cleaned and normalized text string
    """
    # Handle None or non-string inputs
    if text is None:
        return ""
    
    # Convert to string if not already
    text_str = str(text)
    
    # Handle empty or whitespace-only strings
    if not text_str.strip():
        return ""
    
    # Normalize whitespace: replace multiple spaces, tabs, newlines with single spaces
    # Split and rejoin to remove all extra whitespace
    normalized = " ".join(text_str.split())
    
    # Strip leading and trailing whitespace
    cleaned = normalized.strip()
    
    return cleaned


def run_agent(question: str, contexts: Iterable[str], llm, config: AgentConfig | None = None) -> str:
    """
    Run the legal reasoning agent with context management and error handling.
    
    This function orchestrates question-answering using retrieved context with:
    - Context window management through chunk selection
    - LLM API failure handling with graceful fallbacks
    - Configurable temperature and context limits
    - Robust error handling and recovery
    - Parameter validation and default substitution
    
    Args:
        question: The question to answer
        contexts: Iterable of context strings from retrieval
        llm: Language model instance with invoke() or __call__ method
        config: Agent configuration (uses defaults if None)
        
    Returns:
        Generated answer string, or fallback response on errors
    """
    # Validate question parameter
    original_question = question
    validated_question = validate_and_default_string(
        question, "What is the answer?", "question", "Agent.run_agent", allow_empty=False
    )
    if validated_question != original_question:
        log_parameter_correction(original_question, validated_question, "question", "Agent.run_agent", "invalid question")
    
    # Validate contexts parameter
    original_contexts = contexts
    validated_contexts = validate_and_default_iterable(
        contexts, [], "contexts", "Agent.run_agent", allow_empty=True
    )
    if validated_contexts != list(original_contexts):
        log_parameter_correction(original_contexts, validated_contexts, "contexts", "Agent.run_agent", "invalid contexts")
    
    # Use default config if none provided or validate existing config
    if config is None:
        config = AgentConfig()
        logger.info("Agent.run_agent: No config provided, using default AgentConfig")
    elif not isinstance(config, AgentConfig):
        logger.warning(f"Agent.run_agent: Invalid config type {type(config)}, using default AgentConfig")
        config = AgentConfig()
    
    try:
        # Convert contexts to list for indexing and length operations
        context_list = list(validated_contexts)
        
        # Manage context window limitations by selecting appropriate chunks
        selected_contexts = _select_context_chunks(context_list, config.max_context_chunks)
        
        # Build prompt with selected contexts
        prompt = build_prompt(validated_question, selected_contexts)
        
        # Handle LLM API calls with retry and exception handling
        try:
            response = _invoke_llm_with_retry(
                llm=llm,
                prompt=prompt,
                temperature=config.temperature,
                max_retries=3,
                base_delay_seconds=0.5,
            )
        except Exception as api_error:
            # Handle LLM API failures with fallback response
            return _handle_llm_failure(validated_question, selected_contexts, api_error)
        
        # Parse and clean the response
        parsed_response = parse_model_output(response)
        
        # Return fallback if parsing resulted in empty response
        if not parsed_response:
            return _generate_fallback_response(validated_question, selected_contexts)
            
        return parsed_response
        
    except Exception as general_error:
        # Handle any other unexpected errors
        return _handle_general_failure(validated_question, general_error)


def _select_context_chunks(contexts: list[str], max_chunks: int) -> list[str]:
    """
    Select appropriate context chunks respecting the maximum limit.
    
    Args:
        contexts: List of context strings
        max_chunks: Maximum number of chunks to select
        
    Returns:
        Selected context chunks (up to max_chunks)
    """
    if not contexts:
        return []
    
    # If we have fewer contexts than the limit, return all
    if len(contexts) <= max_chunks:
        return contexts
    
    # Select the first max_chunks contexts (assuming they're ranked by relevance)
    return contexts[:max_chunks]


def _invoke_llm_with_retry(
    llm,
    prompt: str,
    temperature: float,
    max_retries: int = 3,
    base_delay_seconds: float = 0.5,
) -> str:
    """
    Invoke an LLM with bounded exponential backoff retry.

    Args:
        llm: LLM object with invoke() method or callable interface
        prompt: Prompt string to send
        temperature: Sampling temperature (used when supported)
        max_retries: Maximum retry attempts after the initial call
        base_delay_seconds: Base delay used for exponential backoff

    Returns:
        Raw LLM response

    Raises:
        Exception: Re-raises the last API exception after retries are exhausted
    """
    if llm is None:
        raise ValueError("LLM object cannot be None")

    attempt = 0
    max_attempts = max_retries + 1
    last_error: Exception | None = None

    while attempt < max_attempts:
        try:
            if hasattr(llm, "invoke"):
                if hasattr(llm, "temperature") or "temperature" in getattr(llm, "__dict__", {}):
                    return llm.invoke(prompt, temperature=temperature)
                return llm.invoke(prompt)

            if callable(llm):
                return llm(prompt)

            raise AttributeError("LLM object is not callable and has no invoke method")

        except Exception as error:
            last_error = error
            attempt += 1

            if attempt >= max_attempts:
                logger.error(
                    "Agent._invoke_llm_with_retry: LLM call failed after %s attempts. "
                    "Error type: %s, message: %s",
                    max_attempts,
                    type(error).__name__,
                    str(error),
                )
                raise

            delay = min(base_delay_seconds * (2 ** (attempt - 1)), 8.0)
            logger.warning(
                "Agent._invoke_llm_with_retry: LLM call attempt %s/%s failed (%s). "
                "Retrying in %.2f seconds.",
                attempt,
                max_attempts,
                type(error).__name__,
                delay,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected retry state: no response and no error captured")


def _handle_llm_failure(question: str, contexts: list[str], error: Exception) -> str:
    """
    Handle LLM API failures with comprehensive fallback response and error details.
    
    Args:
        question: The original question
        contexts: Selected context chunks
        error: The exception that occurred
        
    Returns:
        Fallback response string with detailed error information
    """
    # Classify error type for better user guidance
    error_type = type(error).__name__
    error_message = str(error)
    
    if "timeout" in error_message.lower() or "timed out" in error_message.lower():
        error_classification = "timeout"
        user_guidance = "The request timed out. Try again or use a shorter question."
    elif "connection" in error_message.lower() or "network" in error_message.lower():
        error_classification = "network"
        user_guidance = "Network connection issue. Check internet connectivity and try again."
    elif "authentication" in error_message.lower() or "unauthorized" in error_message.lower():
        error_classification = "authentication"
        user_guidance = "Authentication failed. Check API credentials and permissions."
    elif "rate limit" in error_message.lower() or "quota" in error_message.lower():
        error_classification = "rate_limit"
        user_guidance = "API rate limit exceeded. Wait a moment and try again."
    else:
        error_classification = "unknown"
        user_guidance = "Unexpected API error occurred. Contact system administrator if problem persists."
    
    # Log detailed error information for debugging
    logger.error(
        f"Agent._handle_llm_failure: LLM API failure during question processing. "
        f"Question: '{question}' (length: {len(question)}), "
        f"Contexts available: {len(contexts)}, "
        f"Context total length: {sum(len(ctx) for ctx in contexts)} chars, "
        f"Error type: {error_type}, "
        f"Error classification: {error_classification}, "
        f"Error message: '{error_message}', "
        f"User guidance: {user_guidance}"
    )
    
    # Generate a fallback response based on available context
    if contexts:
        # Use first context chunk as fallback with error explanation
        context_preview = contexts[0][:200] + "..." if len(contexts[0]) > 200 else contexts[0]
        return (
            f"Unable to generate answer due to LLM API error ({error_classification}). "
            f"However, here is relevant context that may help: {context_preview} "
            f"[Error details: {error_type}. {user_guidance}]"
        )
    else:
        return (
            f"Unable to generate answer due to LLM API error ({error_classification}) "
            f"and no context is available. "
            f"[Error details: {error_type}. {user_guidance}]"
        )


def _generate_fallback_response(question: str, contexts: list[str]) -> str:
    """
    Generate comprehensive fallback response when model output is empty or invalid.
    
    Args:
        question: The original question
        contexts: Selected context chunks
        
    Returns:
        Fallback response string with helpful information
    """
    logger.warning(
        f"Agent._generate_fallback_response: Generating fallback due to empty/invalid model output. "
        f"Question: '{question}' (length: {len(question)}), "
        f"Contexts available: {len(contexts)}, "
        f"Context lengths: {[len(ctx) for ctx in contexts] if contexts else 'none'}"
    )
    
    if contexts:
        # Analyze context to provide better fallback
        total_context_length = sum(len(ctx) for ctx in contexts)
        context_preview = contexts[0][:200] + "..." if len(contexts[0]) > 200 else contexts[0]
        
        return (
            f"Based on available context ({len(contexts)} documents, {total_context_length} characters): "
            f"{context_preview} "
            f"[Note: LLM generated empty response, showing relevant context instead]"
        )
    else:
        return (
            f"Unable to generate answer for question: '{question[:100]}{'...' if len(question) > 100 else ''}'. "
            f"No context information is available and LLM response was empty. "
            f"[Suggestion: Try rephrasing the question or provide relevant context documents]"
        )


def _handle_general_failure(question: str, error: Exception) -> str:
    """
    Handle general unexpected failures with comprehensive error reporting.
    
    Args:
        question: The original question
        error: The exception that occurred
        
    Returns:
        Error response string with debugging information
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # Log comprehensive error details for debugging
    logger.error(
        f"Agent._handle_general_failure: Unexpected system error during agent execution. "
        f"Question: '{question}' (length: {len(question)}), "
        f"Error type: {error_type}, "
        f"Error message: '{error_message}', "
        f"This indicates a system-level issue that needs investigation. "
        f"User will receive generic error message to avoid exposing internal details."
    )
    
    # Provide user-friendly error message without exposing internal details
    if "memory" in error_message.lower() or "out of memory" in error_message.lower():
        user_message = "System is experiencing memory constraints. Try with a shorter question or fewer context documents."
    elif "permission" in error_message.lower() or "access" in error_message.lower():
        user_message = "System permission error. Contact administrator for assistance."
    elif "timeout" in error_message.lower():
        user_message = "Operation timed out. Try again with a simpler question."
    else:
        user_message = "System error occurred. Please try again or contact support if the problem persists."
    
    return (
        f"Unable to process question due to system error. {user_message} "
        f"[Error ID: {error_type} - Contact support with this ID if needed]"
    )
