"""LLM provider integrations."""

from llm_bench.providers.base import BaseProvider, ProviderResponse
from llm_bench.providers.openai_provider import OpenAIProvider
from llm_bench.providers.anthropic_provider import AnthropicProvider
from llm_bench.providers.gemini_provider import GeminiProvider
from llm_bench.providers.mistral_provider import MistralProvider
from llm_bench.providers.groq_provider import GroqProvider

__all__ = [
    "BaseProvider",
    "ProviderResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MistralProvider",
    "GroqProvider",
]

# Registry of all available providers
PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
    "groq": GroqProvider,
}

# Model alias -> provider mapping
MODEL_TO_PROVIDER: dict[str, str] = {
    # OpenAI
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4-turbo": "openai",
    "gpt-4": "openai",
    "gpt-3.5-turbo": "openai",
    "o1": "openai",
    "o1-mini": "openai",
    "o3-mini": "openai",
    # Anthropic
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-sonnet": "anthropic",
    "claude-3-5-haiku": "anthropic",
    "claude-3-opus": "anthropic",
    "claude-3-haiku": "anthropic",
    "claude-opus-4": "anthropic",
    "claude-sonnet-4": "anthropic",
    # Gemini
    "gemini-2.0-flash": "gemini",
    "gemini-2.0-flash-lite": "gemini",
    "gemini-1.5-pro": "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-2.5-pro": "gemini",
    # Mistral
    "mistral-large": "mistral",
    "mistral-medium": "mistral",
    "mistral-small": "mistral",
    "mistral-7b": "mistral",
    "mixtral-8x7b": "mistral",
    "codestral": "mistral",
    # Groq
    "llama-3.3-70b-versatile": "groq",
    "llama-3.1-70b-versatile": "groq",
    "llama-3.1-8b-instant": "groq",
    "llama3-70b-8192": "groq",
    "mixtral-8x7b-32768": "groq",
    "gemma2-9b-it": "groq",
}


def resolve_provider(model: str) -> BaseProvider:
    """Resolve a model name to its provider instance."""
    # Check exact match first
    if model in MODEL_TO_PROVIDER:
        provider_name = MODEL_TO_PROVIDER[model]
        return PROVIDER_REGISTRY[provider_name]()

    # Try prefix matching
    model_lower = model.lower()
    if model_lower.startswith(("gpt-", "o1", "o3")):
        return OpenAIProvider()
    elif model_lower.startswith("claude-"):
        return AnthropicProvider()
    elif model_lower.startswith("gemini-"):
        return GeminiProvider()
    elif model_lower.startswith(("mistral-", "mixtral-", "codestral")):
        return MistralProvider()

    # Default to trying by provider name prefix
    for provider_name, provider_class in PROVIDER_REGISTRY.items():
        if model_lower.startswith(provider_name):
            return provider_class()

    raise ValueError(
        f"Unknown model: {model!r}. "
        f"Use --provider to specify, or check `llm-bench list-models`."
    )
