"""
Tests for LLM Service Module

Tests the LLMService, providers, and LLMResponse dataclass.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm_service import (
    LLMService,
    LLMResponse,
    BaseLLMProvider,
    OllamaProvider,
    OpenAIProvider,
    GeminiProvider,
    MistralProvider,
)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""
    
    def test_response_creation(self):
        """Test basic response creation."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
        )
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.usage is None
        assert response.finish_reason is None
    
    def test_response_with_usage(self):
        """Test response with usage statistics."""
        response = LLMResponse(
            content="Test",
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            finish_reason="stop",
        )
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.finish_reason == "stop"
    
    def test_response_str(self):
        """Test string representation."""
        response = LLMResponse(content="Hello world", model="test")
        assert str(response) == "Hello world"


class TestOllamaProvider:
    """Tests for Ollama provider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = OllamaProvider(model="mistral:7b-instruct-q4_K_M")
        assert provider.model_name == "mistral:7b-instruct-q4_K_M"
    
    def test_initialization_custom_url(self):
        """Test with custom base URL."""
        provider = OllamaProvider(
            model="mistral",
            base_url="http://custom:11434",
        )
        assert provider._base_url == "http://custom:11434"
    
    def test_rag_system_prompt(self):
        """Test RAG system prompt generation."""
        provider = OllamaProvider()
        prompt = provider._get_rag_system_prompt()
        assert "context" in prompt.lower()
        assert "answer" in prompt.lower()
    
    def test_rag_prompt_building(self):
        """Test RAG prompt construction."""
        provider = OllamaProvider()
        prompt = provider._build_rag_prompt(
            query="What is AI?",
            context="AI is artificial intelligence..."
        )
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence" in prompt
        assert "context" in prompt.lower()
    
    @patch('src.llm_service.OllamaProvider._get_client')
    def test_generate_calls_client(self, mock_get_client):
        """Test that generate calls the Ollama client correctly."""
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "Response text"},
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        mock_get_client.return_value = mock_client
        
        provider = OllamaProvider()
        response = provider.generate("Test prompt")
        
        assert response.content == "Response text"
        mock_client.chat.assert_called_once()


class TestOpenAIProvider:
    """Tests for OpenAI provider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider(model="gpt-4")
        assert provider.model_name == "gpt-4"
    
    def test_rag_system_prompt(self):
        """Test RAG system prompt."""
        provider = OpenAIProvider()
        prompt = provider._get_rag_system_prompt()
        assert "context" in prompt.lower()
    
    def test_rag_prompt_building(self):
        """Test RAG prompt building."""
        provider = OpenAIProvider()
        prompt = provider._build_rag_prompt(
            query="What is ML?",
            context="ML is machine learning..."
        )
        assert "What is ML?" in prompt
        assert "ML is machine learning" in prompt


class TestGeminiProvider:
    """Tests for Gemini provider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = GeminiProvider(model="gemini-pro")
        assert provider.model_name == "gemini-pro"
    
    def test_rag_system_prompt(self):
        """Test RAG system prompt."""
        provider = GeminiProvider()
        prompt = provider._get_rag_system_prompt()
        assert "context" in prompt.lower()


class TestMistralProvider:
    """Tests for MistralProvider."""
    
    def test_initialization(self):
        """Test basic initialization."""
        provider = MistralProvider(model="mistral-small-latest")
        assert provider.model_name == "mistral-small-latest"
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        provider = MistralProvider(model="mistral-large-latest")
        assert provider.model_name == "mistral-large-latest"
    
    def test_rag_system_prompt(self):
        """Test RAG system prompt content."""
        provider = MistralProvider()
        prompt = provider._get_rag_system_prompt()
        
        assert "AI Bootcamp" in prompt
        assert "context" in prompt.lower()
    
    def test_rag_prompt_building(self):
        """Test RAG prompt construction."""
        provider = MistralProvider()
        context = "The bootcamp is 11 weeks."
        query = "How long is the bootcamp?"
        
        prompt = provider._build_rag_prompt(query, context)
        
        assert context in prompt
        assert query in prompt
        assert "context" in prompt.lower()


class TestLLMService:
    """Tests for the main LLM service."""
    
    @patch('src.llm_service.OllamaProvider')
    def test_ollama_provider_selection(self, mock_ollama):
        """Test that Ollama provider is selected correctly."""
        mock_ollama.return_value = Mock(spec=OllamaProvider)
        
        service = LLMService(provider="ollama")
        
        assert service.provider_name == "ollama"
        mock_ollama.assert_called_once()
    
    @patch('src.llm_service.OpenAIProvider')
    def test_openai_provider_selection(self, mock_openai):
        """Test that OpenAI provider is selected correctly."""
        mock_openai.return_value = Mock(spec=OpenAIProvider)
        
        service = LLMService(provider="openai")
        
        assert service.provider_name == "openai"
        mock_openai.assert_called_once()
    
    @patch('src.llm_service.GeminiProvider')
    def test_gemini_provider_selection(self, mock_gemini):
        """Test that Gemini provider is selected correctly."""
        mock_gemini.return_value = Mock(spec=GeminiProvider)
        
        service = LLMService(provider="gemini")
        
        assert service.provider_name == "gemini"
        mock_gemini.assert_called_once()
    
    @patch('src.llm_service.MistralProvider')
    def test_mistral_provider_selection(self, mock_mistral):
        """Test that Mistral provider is selected correctly."""
        mock_mistral.return_value = Mock(spec=MistralProvider)
        
        service = LLMService(provider="mistral")
        
        assert service.provider_name == "mistral"
        mock_mistral.assert_called_once()
    
    def test_invalid_provider_raises(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMService(provider="invalid_provider")
    
    @patch('src.llm_service.OllamaProvider')
    def test_generate_delegates_to_provider(self, mock_ollama_cls):
        """Test that generate calls the provider."""
        mock_provider = Mock()
        mock_provider.generate.return_value = LLMResponse(
            content="Test",
            model="mistral:7b-instruct-q4_K_M"
        )
        mock_ollama_cls.return_value = mock_provider
        
        service = LLMService(provider="ollama")
        response = service.generate("Test prompt")
        
        mock_provider.generate.assert_called_once()
        assert response.content == "Test"
    
    @patch('src.llm_service.OllamaProvider')
    def test_generate_with_context_delegates(self, mock_ollama_cls):
        """Test that generate_with_context calls the provider."""
        mock_provider = Mock()
        mock_provider.generate_with_context.return_value = LLMResponse(
            content="Context-based response",
            model="mistral:7b-instruct-q4_K_M"
        )
        mock_ollama_cls.return_value = mock_provider
        
        service = LLMService(provider="ollama")
        response = service.generate_with_context(
            query="What is AI?",
            context="AI is artificial intelligence.",
        )
        
        mock_provider.generate_with_context.assert_called_once()
        assert "Context-based response" in response.content
    
    @patch('src.llm_service.OllamaProvider')
    def test_model_name_property(self, mock_ollama_cls):
        """Test model_name property."""
        mock_provider = Mock()
        mock_provider.model_name = "mistral:7b-instruct-q4_K_M"
        mock_ollama_cls.return_value = mock_provider
        
        service = LLMService(provider="ollama")
        
        assert service.model_name == "mistral:7b-instruct-q4_K_M"
