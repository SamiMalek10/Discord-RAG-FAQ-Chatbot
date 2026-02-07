"""
LLM Service Module

Provides an abstraction layer for Large Language Model providers:
- Local: Ollama (Llama2, Mistral, etc.) - Free, runs locally
- Cloud: OpenAI (GPT-3.5/4) - Requires API key
- Cloud: Google Gemini - Requires API key

Design Rationale:
- Abstract interface allows easy switching between providers
- Local option (Ollama) for development without API costs
- Cloud options for production quality
- Configurable via environment variables

Usage:
    llm = LLMService(provider="ollama")
    response = llm.generate("What is machine learning?")
    
    # With system prompt
    response = llm.generate(
        prompt="Explain RAG",
        system_prompt="You are an AI expert."
    )
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass
import os

# Import the new google-genai package (not the deprecated google.generativeai)
from google import genai
from google.genai import types

from config.settings import get_settings, LLMConfig

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers.
    
    Attributes:
        content: The generated text response
        model: Model name used for generation
        usage: Token usage statistics (if available)
        finish_reason: Why generation stopped
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    
    def __str__(self) -> str:
        return self.content


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement:
    - generate: Generate text from a prompt
    - generate_with_context: Generate with retrieved context (for RAG)
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system instructions
            temperature: Creativity (0-1, lower = more deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Generate a response using retrieved context (RAG).
        
        Args:
            query: User's question
            context: Retrieved context from vector store
            system_prompt: Optional system instructions
            temperature: Lower for factual responses
            
        Returns:
            LLMResponse object
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local LLM inference.
    
    Benefits:
    - Free to use (runs locally)
    - No API key required
    - Privacy (data stays local)
    - Works offline
    
    Requirements:
    - Ollama installed: https://ollama.ai
    - Model pulled: ollama pull llama2
    
    Supported models:
    - llama2, llama2:13b, llama2:70b
    - mistral, mistral:7b
    - codellama
    - And many more...
    """
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = None
        
        logger.info(f"Initializing OllamaProvider: model={model}, url={base_url}")
    
    def _get_client(self):
        """Get or create Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self._base_url)
                logger.info("Ollama client initialized")
            except ImportError:
                raise ImportError(
                    "ollama package required. Install with: pip install ollama"
                )
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate response using Ollama."""
        client = self._get_client()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Generate
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        
        try:
            response = client.chat(
                model=self._model,
                messages=messages,
                options=options,
            )
            
            return LLMResponse(
                content=response["message"]["content"],
                model=self._model,
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                },
                finish_reason="stop",
            )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate RAG response with context."""
        # Default RAG system prompt
        if not system_prompt:
            system_prompt = self._get_rag_system_prompt()
        
        # Build the RAG prompt
        rag_prompt = self._build_rag_prompt(query, context)
        
        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    
    def _get_rag_system_prompt(self) -> str:
        """Get the default RAG system prompt."""
        return """You are a knowledgeable AI assistant for the PM Accelerator AI Bootcamp program.
Your job is to provide accurate, detailed answers based on the provided context.

IMPORTANT GUIDELINES:
1. EXTRACT specific information from the context - look for names, dates, weeks, percentages, tier names
2. ALWAYS answer if the context contains ANY relevant information
3. Be DETAILED - include all relevant facts, not just summaries
4. FORMAT responses clearly with bullet points or numbered lists when listing items
5. CITE the source document when providing information
6. For duration questions: Look for week numbers, timeframes, phases
7. For award/tier questions: List ALL tiers with full criteria
8. For process questions: Provide step-by-step details"""

    def _build_rag_prompt(self, query: str, context: str) -> str:
        """Build the RAG prompt with context."""
        return f"""CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Read the context THOROUGHLY before answering
- Extract and include SPECIFIC details (dates, names, criteria, steps)
- If multiple relevant pieces exist, combine them into a comprehensive answer
- Only say "I don't have information" if the context is COMPLETELY irrelevant to the question

ANSWER:"""

    @property
    def model_name(self) -> str:
        return self._model


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider for GPT models.
    
    Benefits:
    - High quality responses
    - Well-documented API
    - Reliable and fast
    
    Models:
    - gpt-3.5-turbo: Fast, cost-effective
    - gpt-4: Most capable
    - gpt-4-turbo: Balance of speed and capability
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name
            api_key: API key (or from environment)
        """
        self._model = model
        self._api_key = api_key
        self._client = None
        
        logger.info(f"Initializing OpenAIProvider: model={model}")
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                import os
                
                api_key = self._api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    settings = get_settings()
                    api_key = settings.llm.openai_api_key
                
                if not api_key:
                    raise ValueError(
                        "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                    )
                
                self._client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
                
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate response using OpenAI."""
        client = self._get_client()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Generate
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        try:
            response = client.chat.completions.create(**kwargs)
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=choice.finish_reason,
            )
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate RAG response with context."""
        if not system_prompt:
            system_prompt = self._get_rag_system_prompt()
        
        rag_prompt = self._build_rag_prompt(query, context)
        
        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    
    def _get_rag_system_prompt(self) -> str:
        return """You are a helpful AI assistant for the AI Bootcamp program. 
Your role is to answer questions based ONLY on the provided context.

Rules:
1. Answer based on the context provided - do not make up information
2. If the context doesn't contain enough information, say so
3. Be concise and helpful
4. Cite which document/source your answer comes from when possible
5. If asked about something not in the context, politely say you don't have that information"""
    
    def _build_rag_prompt(self, query: str, context: str) -> str:
        return f"""Context information from the knowledge base:
---
{context}
---

Based on the context above, please answer the following question:
{query}

If the context doesn't contain relevant information, say "I don't have enough information to answer that question based on the available documents."
"""
    
    @property
    def model_name(self) -> str:
        return self._model




class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider using the NEW google-genai package.
    
    Benefits:
    - Large context window
    - Competitive pricing
    - Good for long documents
    - Supports latest models (gemini-1.5-flash, gemini-2.0-flash, etc.)
    
    Models:
    - gemini-2.0-flash: Latest, fastest, recommended
    - gemini-1.5-flash: Fast and efficient
    - gemini-1.5-pro: More capable, longer context
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini provider.
        
        Args:
            model: Gemini model name (e.g., gemini-2.0-flash, gemini-1.5-flash)
            api_key: API key (or from environment)
        """
        self._model = model
        self._api_key = api_key
        self._client = None
        
        logger.info(f"Initializing GeminiProvider: model={model}")
    
    def _get_client(self):
        """Get or create Gemini client using the new google-genai package."""
        if self._client is None:
            try:
                api_key = self._api_key or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    settings = get_settings()
                    api_key = settings.llm.gemini_api_key
                
                if not api_key:
                    raise ValueError(
                        "Gemini API key not found. Set GEMINI_API_KEY environment variable."
                    )
                
                # Use the new google-genai Client
                self._client = genai.Client(api_key=api_key)
                logger.info(f"Gemini client initialized with model: {self._model}")
                
            except ImportError:
                raise ImportError(
                    "google-genai package required. "
                    "Install with: pip install google-genai"
                )
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate response using Gemini with the new API."""
        client = self._get_client()
        
        # Build contents with system instruction
        contents = prompt
        
        # Configure generation
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )
        if max_tokens:
            config.max_output_tokens = max_tokens
        
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            
            return LLMResponse(
                content=response.text,
                model=self._model,
                finish_reason="stop",
            )
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate RAG response with context."""
        if not system_prompt:
            system_prompt = self._get_rag_system_prompt()
        
        rag_prompt = self._build_rag_prompt(query, context)
        
        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    
    def _get_rag_system_prompt(self) -> str:
        return """You are a knowledgeable AI assistant for the PM Accelerator AI Bootcamp program.
Your job is to provide accurate, detailed answers based on the provided context.

IMPORTANT GUIDELINES:
1. EXTRACT specific information from the context - look for names, dates, weeks, percentages, tier names
2. ALWAYS answer if the context contains ANY relevant information
3. Be DETAILED - include all relevant facts, not just summaries
4. FORMAT responses clearly with bullet points or numbered lists when listing items
5. CITE the source document when providing information
6. For duration questions: Look for week numbers, timeframes, phases
7. For award/tier questions: List ALL tiers with full criteria
8. For process questions: Provide step-by-step details"""

    def _build_rag_prompt(self, query: str, context: str) -> str:
        return f"""CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Read the context THOROUGHLY before answering
- Extract and include SPECIFIC details (dates, names, criteria, steps)
- If multiple relevant pieces exist, combine them into a comprehensive answer
- Only say "I don't have information" if the context is COMPLETELY irrelevant to the question

ANSWER:"""
    @property
    def model_name(self) -> str:
        return self._model


class MistralProvider(BaseLLMProvider):
    """
    Mistral AI cloud provider.
    
    Benefits:
    - Excellent performance/cost ratio
    - Fast inference
    - Good for RAG applications
    - No rate limit issues on paid tier
    
    Models:
    - mistral-small-latest: Fast, efficient (recommended for FAQ)
    - mistral-medium-latest: Balanced
    - mistral-large-latest: Most capable
    - open-mistral-7b: Open source variant
    - open-mixtral-8x7b: Mixture of experts
    """
    
    def __init__(
        self,
        model: str = "mistral-small-latest",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Mistral provider.
        
        Args:
            model: Mistral model name
            api_key: API key (or from environment)
        """
        self._model = model
        self._api_key = api_key
        self._client = None
        
        logger.info(f"Initializing MistralProvider: model={model}")
    
    def _get_client(self):
        """Get or create Mistral client."""
        if self._client is None:
            try:
                from mistralai import Mistral
                
                api_key = self._api_key or os.getenv("MISTRAL_API_KEY")
                if not api_key:
                    settings = get_settings()
                    api_key = settings.llm.mistral_api_key
                
                if not api_key:
                    raise ValueError(
                        "Mistral API key not found. Set MISTRAL_API_KEY environment variable."
                    )
                
                self._client = Mistral(api_key=api_key)
                logger.info(f"Mistral client initialized with model: {self._model}")
                
            except ImportError:
                raise ImportError(
                    "mistralai package required. "
                    "Install with: pip install mistralai"
                )
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate response using Mistral."""
        client = self._get_client()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = client.chat.complete(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content,
                model=self._model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                finish_reason=choice.finish_reason,
            )
        except Exception as e:
            logger.error(f"Mistral generation error: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate RAG response with context."""
        if not system_prompt:
            system_prompt = self._get_rag_system_prompt()
        
        rag_prompt = self._build_rag_prompt(query, context)
        
        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    
    def _get_rag_system_prompt(self) -> str:
        return """You are a knowledgeable AI assistant for the PM Accelerator AI Bootcamp program.
Your job is to provide accurate, detailed answers based on the provided context.

IMPORTANT GUIDELINES:
1. EXTRACT specific information from the context - look for names, dates, weeks, percentages, tier names
2. ALWAYS answer if the context contains ANY relevant information
3. Be DETAILED - include all relevant facts, not just summaries
4. FORMAT responses clearly with bullet points or numbered lists when listing items
5. CITE the source document when providing information
6. For duration questions: Look for week numbers, timeframes, phases
7. For award/tier questions: List ALL tiers with full criteria
8. For process questions: Provide step-by-step details"""

    def _build_rag_prompt(self, query: str, context: str) -> str:
        return f"""CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Read the context THOROUGHLY before answering
- Extract and include SPECIFIC details (dates, names, criteria, steps)
- If multiple relevant pieces exist, combine them into a comprehensive answer
- Only say "I don't have information" if the context is COMPLETELY irrelevant to the question

ANSWER:"""
    
    @property
    def model_name(self) -> str:
        return self._model


class LLMService:
    """
    Main LLM Service with unified interface.
    
    This is the class that other components should use.
    It handles provider selection based on configuration.
    
    Example:
        # Using default provider from config
        llm = LLMService()
        response = llm.generate("What is AI?")
        
        # Specify provider
        llm = LLMService(provider="openai")
        
        # For RAG
        response = llm.generate_with_context(
            query="What is the bootcamp schedule?",
            context="The bootcamp runs for 11 weeks..."
        )
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        """
        Initialize the LLM service.
        
        Args:
            provider: "ollama", "openai", "gemini", or "mistral" (default from config)
            config: Optional LLMConfig instance
        """
        settings = get_settings()
        self.config = config or settings.llm
        
        # Determine provider
        provider = provider or self.config.provider
        
        # Initialize the appropriate provider
        if provider == "ollama":
            self._provider = OllamaProvider(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
            )
        elif provider == "openai":
            self._provider = OpenAIProvider(
                model=self.config.openai_model,
                api_key=self.config.openai_api_key,
            )
        elif provider == "gemini":
            self._provider = GeminiProvider(
                model=self.config.gemini_model,
                api_key=self.config.gemini_api_key,
            )
        elif provider == "mistral":
            self._provider = MistralProvider(
                model=self.config.mistral_model,
                api_key=self.config.mistral_api_key,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        self._provider_name = provider
        logger.info(f"LLMService initialized with {provider} provider")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system instructions
            temperature: Creativity (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse object
        """
        return self._provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Generate a RAG response with retrieved context.
        
        Args:
            query: User's question
            context: Retrieved context from vector store
            system_prompt: Optional system instructions
            temperature: Lower for factual responses
            
        Returns:
            LLMResponse object
        """
        return self._provider.generate_with_context(
            query=query,
            context=context,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._provider.model_name
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self._provider_name
