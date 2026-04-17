"""Tests for agent module functionality."""

import pytest
from src.agent import parse_model_output, build_prompt, AgentConfig, run_agent


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, response="Mock response", should_fail=False, has_invoke=True):
        self.response = response
        self.should_fail = should_fail
        self.has_invoke = has_invoke
        self.temperature = None
    
    def invoke(self, prompt, temperature=None):
        if self.should_fail:
            raise Exception("Mock API failure")
        self.temperature = temperature
        return self.response
    
    def __call__(self, prompt):
        if self.should_fail:
            raise Exception("Mock API failure")
        return self.response


class TestParseModelOutput:
    """Test cases for parse_model_output function."""
    
    def test_parse_normal_text(self):
        """Test parsing normal text input."""
        result = parse_model_output("This is a normal response.")
        assert result == "This is a normal response."
    
    def test_parse_text_with_extra_whitespace(self):
        """Test parsing text with extra whitespace."""
        result = parse_model_output("  This   has    extra   spaces  ")
        assert result == "This has extra spaces"
    
    def test_parse_text_with_newlines(self):
        """Test parsing text with newlines and tabs."""
        result = parse_model_output("Line 1\n\nLine 2\t\tLine 3")
        assert result == "Line 1 Line 2 Line 3"
    
    def test_parse_none_input(self):
        """Test parsing None input."""
        result = parse_model_output(None)
        assert result == ""
    
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_model_output("")
        assert result == ""
    
    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string."""
        result = parse_model_output("   \n\t   ")
        assert result == ""
    
    def test_parse_non_string_input(self):
        """Test parsing non-string input (should convert to string)."""
        result = parse_model_output(123)
        assert result == "123"
    
    def test_parse_complex_formatting(self):
        """Test parsing complex text with mixed formatting issues."""
        input_text = "\n\n  The answer is:   \n\n  Legal documents must comply\t\twith regulations.  \n  "
        expected = "The answer is: Legal documents must comply with regulations."
        result = parse_model_output(input_text)
        assert result == expected


class TestBuildPrompt:
    """Test cases for build_prompt function."""
    
    def test_build_prompt_basic(self):
        """Test basic prompt construction."""
        question = "What is the law?"
        contexts = ["Context 1", "Context 2"]
        result = build_prompt(question, contexts)
        
        assert "What is the law?" in result
        assert "Context 1" in result
        assert "Context 2" in result
        assert "legal reasoning agent" in result.lower()
    
    def test_build_prompt_empty_contexts(self):
        """Test prompt construction with empty contexts."""
        question = "What is the law?"
        contexts = []
        result = build_prompt(question, contexts)
        
        assert "What is the law?" in result
        assert "Context:" in result


class TestAgentConfig:
    """Test cases for AgentConfig validation."""
    
    def test_valid_config(self):
        """Test creating valid configuration."""
        config = AgentConfig(temperature=0.5, max_context_chunks=3)
        assert config.temperature == 0.5
        assert config.max_context_chunks == 3
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.temperature == 0.0
        assert config.max_context_chunks == 5
    
    def test_invalid_temperature_high(self):
        """Test invalid high temperature."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            AgentConfig(temperature=3.0)
    
    def test_invalid_temperature_low(self):
        """Test invalid low temperature."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            AgentConfig(temperature=-1.0)
    
    def test_invalid_max_context_chunks(self):
        """Test invalid max_context_chunks."""
        with pytest.raises(ValueError, match="max_context_chunks must be positive"):
            AgentConfig(max_context_chunks=0)


class TestRunAgent:
    """Test cases for run_agent function."""
    
    def test_run_agent_basic(self):
        """Test basic agent execution."""
        llm = MockLLM("Legal answer")
        question = "What is the law?"
        contexts = ["Context 1", "Context 2"]
        
        result = run_agent(question, contexts, llm)
        assert result == "Legal answer"
    
    def test_run_agent_with_config(self):
        """Test agent execution with custom config."""
        llm = MockLLM("Legal answer")
        config = AgentConfig(temperature=0.7, max_context_chunks=2)
        question = "What is the law?"
        contexts = ["Context 1", "Context 2", "Context 3"]
        
        result = run_agent(question, contexts, llm, config)
        assert result == "Legal answer"
        assert llm.temperature == 0.7
    
    def test_run_agent_context_limiting(self):
        """Test context window management."""
        llm = MockLLM("Legal answer")
        config = AgentConfig(max_context_chunks=2)
        question = "What is the law?"
        contexts = ["Context 1", "Context 2", "Context 3", "Context 4"]
        
        result = run_agent(question, contexts, llm, config)
        assert result == "Legal answer"
    
    def test_run_agent_empty_contexts(self):
        """Test agent with empty contexts."""
        llm = MockLLM("Legal answer")
        question = "What is the law?"
        contexts = []
        
        result = run_agent(question, contexts, llm)
        assert result == "Legal answer"
    
    def test_run_agent_llm_failure(self):
        """Test agent handling LLM API failure."""
        llm = MockLLM(should_fail=True)
        question = "What is the law?"
        contexts = ["Context 1", "Context 2"]
        
        result = run_agent(question, contexts, llm)
        assert "Unable to generate answer due to API error" in result
        assert "Context 1" in result
    
    def test_run_agent_llm_failure_no_context(self):
        """Test agent handling LLM API failure with no context."""
        llm = MockLLM(should_fail=True)
        question = "What is the law?"
        contexts = []
        
        result = run_agent(question, contexts, llm)
        assert "Unable to generate answer due to API error and no context available" in result
    
    def test_run_agent_empty_response(self):
        """Test agent handling empty model response."""
        llm = MockLLM("")  # Empty response
        question = "What is the law?"
        contexts = ["Context 1"]
        
        result = run_agent(question, contexts, llm)
        assert "Based on available context" in result
        assert "Context 1" in result
    
    def test_run_agent_empty_response_no_context(self):
        """Test agent handling empty model response with no context."""
        llm = MockLLM("")  # Empty response
        question = "What is the law?"
        contexts = []
        
        result = run_agent(question, contexts, llm)
        assert "Unable to generate answer with the provided information" in result
    
    def test_run_agent_callable_llm(self):
        """Test agent with callable LLM (no invoke method)."""
        llm = MockLLM("Legal answer", has_invoke=False)
        # Remove invoke method to test __call__ fallback
        delattr(llm, 'invoke')
        
        question = "What is the law?"
        contexts = ["Context 1"]
        
        result = run_agent(question, contexts, llm)
        assert result == "Legal answer"
    
    def test_run_agent_invalid_llm(self):
        """Test agent with invalid LLM object."""
        llm = "not_callable"  # Invalid LLM
        question = "What is the law?"
        contexts = ["Context 1"]
        
        result = run_agent(question, contexts, llm)
        assert "Unable to process question due to system error" in result