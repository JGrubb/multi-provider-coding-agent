# PERF-001: Design and implement performance metrics data structure

**Type**: Technical Task  
**Priority**: High  
**Estimate**: 4 hours  
**Milestone**: 1 - Performance Monitoring Foundation

## Description
Create comprehensive data structures to capture and store performance metrics from Ollama model inference runs. This will be the foundation for all performance tracking and optimization features.

## Acceptance Criteria
- [ ] `PerformanceMetrics` dataclass includes all relevant metrics
- [ ] Support for different task types (code_generation, code_review, etc.)
- [ ] Timestamps and model identification included
- [ ] Serializable to/from JSON for database storage
- [ ] Type hints and documentation complete
- [ ] Unit tests cover all functionality

## Technical Requirements
- Use Python dataclasses for clean structure
- Include inference timing, token counts, throughput metrics
- Support system resource usage (CPU, memory)
- Extensible design for future metrics
- JSON serialization support

## Implementation Details
```python
@dataclass
class PerformanceMetrics:
    # Timing metrics
    inference_time: float  # total seconds
    load_duration: Optional[float]  # model load time
    eval_duration: Optional[float]  # evaluation time
    
    # Token metrics
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    
    # Resource metrics
    memory_usage: Optional[float]  # MB
    cpu_usage: Optional[float]  # percentage
    
    # Context
    model_name: str
    task_type: str
    timestamp: datetime
    query_length: int
    response_length: int
```

## Files to Create/Modify
- `performance/metrics.py` - Main metrics classes
- `tests/test_performance_metrics.py` - Unit tests

## Dependencies
- None (foundational task)

## Definition of Done
- Data structures implemented with full type hints
- JSON serialization working correctly
- Unit tests achieve 100% coverage
- Documentation strings complete
- Code passes linting