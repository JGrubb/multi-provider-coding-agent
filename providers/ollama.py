"""Ollama provider implementation with comprehensive performance tracking."""

import json
import time
import psutil
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

from .base import LLMProvider
from performance.metrics import PerformanceMetrics, TaskType


class OllamaProvider(LLMProvider):
    """Ollama provider with detailed performance monitoring.
    
    This provider communicates with a local Ollama instance and captures
    comprehensive performance metrics including timing, token usage, and
    system resource consumption.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        default_model: str = "llama2",
        timeout: int = 120
    ):
        """Initialize Ollama provider.
        
        Args:
            base_url: Base URL for Ollama API
            default_model: Default model to use if none specified
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Performance tracking
        self._last_metrics: Optional[PerformanceMetrics] = None
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, PerformanceMetrics]:
        """Send chat messages to Ollama and capture performance metrics.
        
        Args:
            messages: List of message dictionaries
            model: Model name to use
            task_type: Type of task being performed
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Ollama parameters
            
        Returns:
            Tuple of (response_text, performance_metrics)
        """
        model = model or self.default_model
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        # Prepare request
        prompt = self._messages_to_prompt(messages)
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        # Add optional parameters
        if temperature is not None:
            request_data["temperature"] = temperature
        if max_tokens is not None:
            request_data["num_predict"] = max_tokens
        
        try:
            # Make request to Ollama
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            # Parse response
            response_data = response.json()
            response_text = response_data.get("response", "")
            
            # Extract performance metrics from Ollama response
            metrics = self._extract_performance_metrics(
                response_data=response_data,
                model=model,
                task_type=task_type,
                prompt=prompt,
                response_text=response_text,
                start_time=start_time,
                end_time=end_time,
                start_memory=start_memory,
                end_memory=end_memory,
                start_cpu=start_cpu,
                end_cpu=end_cpu,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            self._last_metrics = metrics
            return response_text, metrics
            
        except requests.exceptions.RequestException as e:
            # Create error metrics
            end_time = time.time()
            error_metrics = PerformanceMetrics(
                inference_time=end_time - start_time,
                input_tokens=len(prompt.split()),  # Rough estimate
                output_tokens=0,
                model_name=model,
                task_type=task_type,
                query_length=len(prompt),
                response_length=0,
                provider=self.get_provider_name(),
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            raise OllamaError(f"Failed to communicate with Ollama: {e}") from e
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Combined prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts)
    
    def _extract_performance_metrics(
        self,
        response_data: Dict[str, Any],
        model: str,
        task_type: TaskType,
        prompt: str,
        response_text: str,
        start_time: float,
        end_time: float,
        start_memory: float,
        end_memory: float,
        start_cpu: float,
        end_cpu: float,
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> PerformanceMetrics:
        """Extract performance metrics from Ollama response and system data.
        
        Args:
            response_data: Raw response from Ollama API
            model: Model name used
            task_type: Type of task performed
            prompt: Input prompt
            response_text: Generated response
            start_time: Request start time
            end_time: Request end time
            start_memory: Memory usage at start
            end_memory: Memory usage at end
            start_cpu: CPU usage at start
            end_cpu: CPU usage at end
            temperature: Temperature parameter used
            max_tokens: Max tokens parameter used
            
        Returns:
            PerformanceMetrics instance
        """
        # Extract timing data from Ollama response (in nanoseconds)
        total_duration = response_data.get("total_duration", 0) / 1e9  # Convert to seconds
        load_duration = response_data.get("load_duration", 0) / 1e9
        eval_duration = response_data.get("eval_duration", 0) / 1e9
        
        # Extract token counts
        eval_count = response_data.get("eval_count", 0)  # Output tokens
        prompt_eval_count = response_data.get("prompt_eval_count", 0)  # Input tokens
        
        # Use Ollama's timing if available, otherwise use our measurement
        inference_time = total_duration if total_duration > 0 else (end_time - start_time)
        
        # Calculate tokens per second
        tokens_per_second = 0.0
        if eval_count > 0 and eval_duration > 0:
            tokens_per_second = eval_count / eval_duration
        elif eval_count > 0 and inference_time > 0:
            tokens_per_second = eval_count / inference_time
        
        # Calculate resource usage
        memory_usage = max(0, end_memory - start_memory) if end_memory and start_memory else None
        cpu_usage = (end_cpu - start_cpu) if end_cpu and start_cpu else None
        
        return PerformanceMetrics(
            # Timing metrics
            inference_time=inference_time,
            load_duration=load_duration if load_duration > 0 else None,
            eval_duration=eval_duration if eval_duration > 0 else None,
            
            # Token metrics
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            tokens_per_second=tokens_per_second,
            
            # Resource metrics
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            
            # Context information
            model_name=model,
            task_type=task_type,
            timestamp=datetime.now(timezone.utc),
            query_length=len(prompt),
            response_length=len(response_text),
            
            # Provider information
            provider=self.get_provider_name(),
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB.
        
        Returns:
            Memory usage in MB, or None if unable to measure
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return None
    
    def _get_cpu_usage(self) -> Optional[float]:
        """Get current CPU usage percentage.
        
        Returns:
            CPU usage percentage, or None if unable to measure
        """
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return None
    
    def list_models(self) -> List[str]:
        """List available models from Ollama.
        
        Returns:
            List of available model names
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_info in data.get("models", []):
                model_name = model_info.get("name", "")
                if model_name:
                    # Remove tag if present (e.g., "llama2:latest" -> "llama2")
                    model_name = model_name.split(":")[0]
                    if model_name not in models:
                        models.append(model_name)
            
            return sorted(models)
            
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Failed to list models: {e}") from e
    
    def get_default_model(self) -> str:
        """Get the default model name.
        
        Returns:
            Default model name
        """
        return self.default_model
    
    def get_provider_name(self) -> str:
        """Get the provider name.
        
        Returns:
            Provider name
        """
        return "ollama"
    
    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with model information
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model}
            )
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "name": model,
                "provider": self.get_provider_name(),
                "available": True,
                "details": data.get("details", {}),
                "modelfile": data.get("modelfile", ""),
                "parameters": data.get("parameters", {}),
                "template": data.get("template", "")
            }
            
        except requests.exceptions.RequestException:
            return None
    
    def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the last captured performance metrics.
        
        Returns:
            Last PerformanceMetrics instance, or None if none available
        """
        return self._last_metrics
    
    def is_available(self) -> bool:
        """Check if Ollama is available and responsive.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry.
        
        Args:
            model: Model name to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=600  # Model pulling can take a long time
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException:
            return False


class OllamaError(Exception):
    """Exception raised for Ollama-specific errors."""
    pass