"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from performance.metrics import PerformanceMetrics


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    This defines the interface that all LLM providers must implement,
    ensuring consistent behavior across different models and services.
    """
    
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, PerformanceMetrics]:
        """Send a chat message to the LLM and return response with performance metrics.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Optional model name to use (uses default if not specified)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Tuple of (response_text, performance_metrics)
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider.
        
        Returns:
            List of model names available
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model name for this provider.
        
        Returns:
            Default model name
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider.
        
        Returns:
            Provider name (e.g., 'ollama', 'anthropic', 'gemini')
        """
        pass
    
    def is_available(self) -> bool:
        """Check if the provider is available and responsive.
        
        Returns:
            True if provider is available, False otherwise
        """
        try:
            self.list_models()
            return True
        except Exception:
            return False
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with model information, or None if not available
        """
        # Default implementation - providers can override
        if model in self.list_models():
            return {
                "name": model,
                "provider": self.get_provider_name(),
                "available": True
            }
        return None