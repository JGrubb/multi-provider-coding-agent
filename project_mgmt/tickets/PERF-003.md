# PERF-003: Implement enhanced Ollama provider with performance tracking

**Type**: Feature  
**Priority**: High  
**Estimate**: 6 hours  
**Milestone**: 1 - Performance Monitoring Foundation

## Description
Create an Ollama provider that captures comprehensive performance metrics during model inference. This provider will be the primary interface for local model interactions with full performance visibility.

## Acceptance Criteria
- [ ] OllamaProvider implements base LLMProvider interface
- [ ] Captures all available performance metrics from Ollama API
- [ ] Supplements with system-level metrics when possible
- [ ] Returns both response and performance data
- [ ] Handles connection errors and timeouts gracefully
- [ ] Supports multiple Ollama models
- [ ] Integration tests with real Ollama instance

## Technical Requirements
- HTTP client for Ollama API (/api/generate endpoint)
- Performance metric extraction from API response
- System resource monitoring during inference
- Error handling for network/model issues
- Async support for non-blocking operations
- Model availability checking

## Implementation Details
```python
class OllamaProvider(LLMProvider):
    def chat(self, messages: List[dict], model: str = None) -> Tuple[str, PerformanceMetrics]:
        # Make request to Ollama API
        # Extract performance metrics from response
        # Add system-level metrics
        # Return response and metrics
        
    async def chat_async(self, messages: List[dict], model: str = None):
        # Async version for non-blocking calls
        
    def list_models(self) -> List[str]:
        # Query available models from Ollama
        
    def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        # Return last captured metrics
```

## Ollama API Response Analysis
Need to capture from Ollama response:
- `eval_duration` - inference time
- `load_duration` - model load time  
- `total_duration` - total request time
- `eval_count` - output token count
- `prompt_eval_count` - input token count

## Files to Create/Modify
- `providers/ollama.py` - Main provider implementation
- `providers/base.py` - Base provider interface updates
- `tests/test_ollama_provider.py` - Unit and integration tests

## Dependencies
- PERF-001 (PerformanceMetrics)
- Ollama running locally for integration tests

## Definition of Done
- Provider successfully communicates with Ollama
- All performance metrics accurately captured
- Error handling covers common failure modes
- Integration tests pass with real Ollama instance
- Performance metrics match expected values
- Code follows project conventions and is well documented