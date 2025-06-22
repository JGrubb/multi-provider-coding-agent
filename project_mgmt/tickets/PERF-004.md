# PERF-004: Create basic agent framework with performance collection

**Type**: Feature  
**Priority**: High  
**Estimate**: 5 hours  
**Milestone**: 1 - Performance Monitoring Foundation

## Description
Implement the core CodingAgent class that orchestrates provider interactions and automatically collects performance data. This is the main interface that users will interact with.

## Acceptance Criteria
- [ ] CodingAgent class manages provider registration
- [ ] Automatic performance data collection on every inference
- [ ] Performance data stored to database after each call
- [ ] Support for model switching and selection
- [ ] Basic conversation history management
- [ ] Error handling and graceful degradation
- [ ] Unit tests cover core functionality

## Technical Requirements
- Provider registration and management
- Automatic performance metric collection
- Database integration for storing metrics
- Model switching API
- Conversation context management
- Logging and error handling

## Implementation Details
```python
class CodingAgent:
    def __init__(self, db_path: str = "performance.db"):
        self.providers: Dict[str, LLMProvider] = {}
        self.db = PerformanceDatabase(db_path)
        self.current_model: Optional[str] = None
        self.conversation_history: List[dict] = []
    
    def add_provider(self, name: str, provider: LLMProvider):
        # Register provider
        
    def use_model(self, model_spec: str):
        # Switch active model (format: "ollama:codellama")
        
    def chat(self, message: str) -> str:
        # Send message to current model
        # Collect performance metrics
        # Store to database
        # Return response
        
    def get_performance_history(self, model: str = None) -> List[PerformanceMetrics]:
        # Query performance data from database
```

## Core Features
1. **Provider Management**: Register and switch between providers
2. **Performance Collection**: Automatic metric collection and storage
3. **Model Selection**: Easy model switching with validation
4. **History Tracking**: Maintain conversation and performance history
5. **Error Recovery**: Handle provider failures gracefully

## Files to Create/Modify
- `agent.py` - Main CodingAgent class
- `tests/test_agent.py` - Unit tests
- `examples/basic_usage.py` - Usage examples

## Dependencies
- PERF-001 (PerformanceMetrics)
- PERF-002 (Database)
- PERF-003 (OllamaProvider)

## Definition of Done
- Agent successfully orchestrates provider calls
- Performance metrics automatically collected and stored
- Model switching works correctly
- Error handling prevents crashes
- Unit tests achieve good coverage
- Basic usage examples work end-to-end