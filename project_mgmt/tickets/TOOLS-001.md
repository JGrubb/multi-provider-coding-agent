# TOOLS-001: Implement basic coding tools for Ollama provider

**Type**: Feature  
**Priority**: Critical  
**Estimate**: 8 hours  
**Milestone**: 1 - Performance Monitoring Foundation

## Description
Implement essential file system and development tools that can be called by local models (specifically Ollama). This enables the coding agent to actually perform coding tasks like reading files, searching code, and making edits. Without these tools, the agent cannot function as a coding assistant.

**Research Phase**: First, investigate existing Python agent frameworks and tool calling libraries to determine if we can leverage existing solutions rather than building from scratch.

## Acceptance Criteria
### Research Phase
- [ ] Research existing Python agent frameworks with tool calling (LangChain, CrewAI, AutoGen, etc.)
- [ ] Evaluate tool calling libraries and their Ollama compatibility
- [ ] Document findings and recommendations for build vs. buy decision
- [ ] Create comparison matrix of existing solutions

### Implementation Phase
- [ ] Tool interface that local models can call through function calling or structured output
- [ ] File operations: read, write, edit files
- [ ] Code search: grep, glob pattern matching
- [ ] Directory operations: list files, create directories
- [ ] Shell command execution with safety controls
- [ ] Tools are exposed to Ollama through chat interface
- [ ] Performance tracking includes tool usage metrics
- [ ] Error handling and security safeguards

## Technical Requirements
### Research Requirements
- Web search for Python agent frameworks and tool calling libraries
- Evaluate frameworks: LangChain, CrewAI, AutoGen, OpenAI Functions, Pydantic AI
- Assess Ollama compatibility and integration complexity
- Performance implications of using existing frameworks
- Licensing and dependency considerations

### Implementation Requirements
- Function calling interface compatible with Ollama models
- File system operations with path validation
- Search functionality (grep, glob patterns)
- Safe shell command execution
- Tool usage logging and performance tracking
- Security controls (no access outside project directory)
- Structured tool responses

## Core Tools to Implement
```python
class ToolExecutor:
    def read_file(self, file_path: str) -> str
    def write_file(self, file_path: str, content: str) -> str
    def edit_file(self, file_path: str, old_text: str, new_text: str) -> str
    def list_files(self, directory: str, pattern: str = "*") -> List[str]
    def grep_files(self, pattern: str, file_pattern: str = "*") -> List[dict]
    def run_command(self, command: str) -> str  # with safety controls
    def create_directory(self, path: str) -> str
```

## Ollama Integration Approach
Two possible approaches:
1. **Structured Output**: Model returns JSON with tool calls, agent executes
2. **Function Calling**: If Ollama model supports function calling, use native approach

Example structured output:
```json
{
  "response": "I'll read the config file first",
  "tool_calls": [
    {"tool": "read_file", "args": {"file_path": "config.py"}}
  ]
}
```

## Safety and Security
- Restrict file operations to project directory
- Whitelist allowed shell commands
- Validate all file paths
- Log all tool usage
- Size limits on file operations
- Timeout controls for shell commands

## Performance Tracking Integration
- Track tool usage in performance metrics
- Measure tool execution time
- Count tool calls per inference
- Track success/failure rates

## Files to Create/Modify
- `tools/base.py` - Tool interface and executor
- `tools/file_tools.py` - File system operations
- `tools/search_tools.py` - Grep and glob functionality
- `tools/shell_tools.py` - Safe shell command execution
- `providers/ollama.py` - Integration with tool system
- `performance/metrics.py` - Add tool usage metrics
- `tests/test_tools.py` - Comprehensive tool testing

## Dependencies
- PERF-003 (OllamaProvider) - needs to be updated
- PERF-001 (PerformanceMetrics) - needs tool usage fields

## Integration with Agent
```python
class CodingAgent:
    def __init__(self):
        self.tool_executor = ToolExecutor()
        # ...
    
    def chat(self, message: str) -> str:
        response, metrics = self.provider.chat(messages)
        
        # Check if response includes tool calls
        if self.has_tool_calls(response):
            tool_results = self.execute_tools(response)
            # Send tool results back to model
            final_response = self.provider.chat_with_tools(tool_results)
            
        return final_response
```

## Research Areas to Investigate
- **LangChain**: Tool calling capabilities and Ollama integration
- **CrewAI**: Agent framework with tool support
- **AutoGen**: Microsoft's multi-agent framework
- **Pydantic AI**: Type-safe agent framework
- **OpenAI Functions**: Function calling patterns adaptable to Ollama
- **Instructor**: Structured output library for LLMs
- **LiteLLM**: Universal LLM interface with tool support

## Definition of Done
### Research Phase Complete
- Comprehensive research document with findings
- Build vs. buy recommendation with justification
- If leveraging existing framework: integration plan documented

### Implementation Phase Complete
- All core tools implemented and tested (either custom or framework-based)
- Ollama can successfully call tools through chosen interface
- Tool usage tracked in performance metrics
- Security controls prevent dangerous operations
- Integration tests demonstrate file operations working
- Tool calls visible in CLI output for transparency
- Performance impact of tool usage measured and reported