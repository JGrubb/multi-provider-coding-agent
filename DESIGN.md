# Multi-Provider Coding Agent Design

## Overview

A Python-based coding agent that can orchestrate conversations between multiple LLM providers (Anthropic, Gemini, Ollama) with transparent output showing all model interactions, delegations, and reasoning.

One of the main points of this agent is to offload inference compute to local models where performance allows.  Therefore, one of the critical implementation details is the ability to retrieve performance profilling information from the Ollama model after it runs.  The frontier model - Claude et al - should use this information iteratively to better understand how to craft queries that can run in a reasonably performant manner locally.  This is the experiemental side of this agent.

Another marquee feature is capability to have various models communicate with each other to compare and refine approaches before implementing.  These conversations are printed back to the human in the REPL so that we can stay in the loop of the thought process.

## Architecture Goals

- **Simple & Transparent**: Show every model interaction in real-time
- **Multi-Provider**: Unified API across Anthropic, Gemini, and Ollama
- **Multi-Model Conversations**: Models can delegate, consult, and collaborate
- **Easy Model Switching**: Simple API for selecting and listing models
- **Extensible**: Easy to add new providers or conversation patterns

## Core Components

### 1. Provider Layer (`providers/`)

**Base Provider Interface**
```python
class LLMProvider:
    def chat(self, messages: List[dict], model: str = None) -> str
    def list_models(self) -> List[str]
    def get_default_model(self) -> str
    def get_provider_name(self) -> str
```

**Provider Implementations**
- `AnthropicProvider`: Uses `anthropic` SDK
- `GeminiProvider`: Uses `google-generativeai` SDK  
- `OllamaProvider`: Uses HTTP requests to localhost:11434

### 2. Agent Core (`agent.py`)

**CodingAgent Class**
- Manages provider registration
- Handles model switching and selection
- Orchestrates multi-model conversations
- Maintains conversation history
- Provides tools integration

**Key Methods**
```python
def use_model(self, model_spec: str)  # "anthropic:claude-3-5-sonnet"
def list_all_models(self) -> Dict[str, List[str]]
def chat(self, message: str) -> str
def delegate(self, target_model: str, task: str) -> str
def consult(self, models: List[str], question: str) -> Dict[str, str]
def add_provider(self, name: str, provider: LLMProvider)
```

### 3. Display Layer (`display.py`)

**ChatDisplay Class**
- Shows model switches and delegations
- Displays all model responses with clear attribution
- Provides conversation flow visualization
- Supports different output formats (console, structured)

**Display Methods**
```python
def show_model_switch(self, from_model: str, to_model: str)
def show_delegation(self, delegator: str, delegatee: str, task: str)
def show_consultation(self, question: str, responses: Dict[str, str])
def show_response(self, model: str, response: str, response_type: str = "chat")
def show_conversation_summary(self, history: List[dict])
```

### 4. Tools Layer (`tools/`)

**Coding Tools**
- File operations (read, write, edit, list)
- Shell command execution
- Git operations
- Code analysis and formatting
- Test execution

**Tool Interface**
```python
class Tool:
    def name(self) -> str
    def description(self) -> str
    def execute(self, **kwargs) -> str
```

### 5. Conversation Patterns (`patterns/`)

**Multi-Model Conversation Types**

1. **Sequential Handoff**: Model A → Task → Model B → Review → Model A
2. **Parallel Consultation**: Question → Multiple Models → Synthesize Responses
3. **Specialist Routing**: Task Analysis → Route to Best Model → Execute
4. **Collaborative Refinement**: Draft → Review → Iterate across models

## Multi-Model Conversation Flow

```
User Input
    ↓
Agent (Model Selection/Routing)
    ↓
Primary Model (Planning/Analysis)
    ↓
[Delegation Decision]
    ↓
Secondary Model(s) (Execution/Consultation)
    ↓
Primary Model (Review/Synthesis)
    ↓
Display Results (All Interactions Visible)
```

## File Structure

```
coding-agent/
├── agent.py                 # Main CodingAgent class
├── display.py              # ChatDisplay and output formatting
├── config.py               # Configuration management
├── main.py                 # CLI entry point
├── providers/
│   ├── __init__.py
│   ├── base.py             # LLMProvider base class
│   ├── anthropic.py        # Anthropic Claude integration
│   ├── gemini.py           # Google Gemini integration
│   └── ollama.py           # Ollama local models integration
├── tools/
│   ├── __init__.py
│   ├── base.py             # Tool base class
│   ├── file_tools.py       # File operations
│   ├── shell_tools.py      # Shell command execution
│   └── git_tools.py        # Git operations
├── patterns/
│   ├── __init__.py
│   ├── delegation.py       # Model delegation patterns
│   ├── consultation.py     # Multi-model consultation
│   └── collaboration.py    # Collaborative workflows
├── examples/
│   ├── basic_usage.py
│   ├── multi_model_refactor.py
│   └── consultation_example.py
├── tests/
├── requirements.txt
└── README.md
```

## Implementation Phases

### Phase 1: Core Foundation
1. **Provider Base Classes** (`providers/base.py`)
   - Define LLMProvider interface
   - Create message format standardization
   - Error handling patterns

2. **Basic Provider Implementations**
   - `AnthropicProvider` with Claude models
   - `OllamaProvider` with local model support
   - `GeminiProvider` with Gemini models

3. **Agent Core** (`agent.py`)
   - CodingAgent class with provider management
   - Basic chat functionality
   - Model switching API

4. **Display System** (`display.py`)
   - ChatDisplay with clear model attribution
   - Conversation flow visualization
   - Real-time interaction logging

### Phase 2: Multi-Model Conversations
1. **Delegation System**
   - Task handoff between models
   - Context preservation across models
   - Transparent delegation display

2. **Consultation System**
   - Parallel model querying
   - Response comparison and synthesis
   - Voting/consensus mechanisms

3. **Conversation Patterns**
   - Sequential workflows
   - Collaborative refinement
   - Specialist routing logic

### Phase 3: Tools Integration
1. **File Operations**
   - Read, write, edit, list files
   - Safe file modification with backups
   - Directory traversal and search

2. **Shell Integration**
   - Command execution with safety checks
   - Output capture and formatting
   - Interactive command confirmation

3. **Code Tools**
   - Syntax checking and linting
   - Code formatting and refactoring
   - Test execution and reporting

### Phase 4: Advanced Features
1. **Configuration System**
   - Provider API key management
   - Model preferences and defaults
   - Conversation pattern customization

2. **History and Context Management**
   - Conversation persistence
   - Context window optimization
   - Smart history summarization

3. **CLI and User Experience**
   - Interactive command-line interface
   - Batch processing capabilities
   - Output formatting options

## API Design Examples

### Basic Usage
```python
from coding_agent import CodingAgent

# Initialize agent
agent = CodingAgent()
agent.load_config()  # Loads API keys, default models

# Simple chat
response = agent.chat("Help me refactor this Python function")

# Model switching
agent.use_model("ollama:codellama")
local_response = agent.chat("Generate docstrings for this code")

# Multi-model consultation
responses = agent.consult([
    "anthropic:claude-3-5-sonnet",
    "gemini:gemini-pro",
    "ollama:codellama"
], "What's the best way to optimize this algorithm?")
```

### Advanced Multi-Model Workflows
```python
# Delegation workflow
agent.use_model("anthropic:claude-3-5-sonnet")
plan = agent.chat("Create a refactoring plan for this codebase")

# Delegate implementation to local model
implementation = agent.delegate("ollama:codellama", 
    f"Implement this refactoring plan: {plan}")

# Review with original model
review = agent.chat(f"Review this implementation: {implementation}")

# Show full conversation trace
agent.display.show_conversation_summary()
```

## Dependencies

```
anthropic>=0.18.0
google-generativeai>=0.3.0
requests>=2.31.0
click>=8.1.0
rich>=13.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

## Configuration

**.env file**
```
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=anthropic:claude-3-5-sonnet
```

**config.yaml**
```yaml
providers:
  anthropic:
    default_model: "claude-3-5-sonnet-20241022"
    max_tokens: 4000
  gemini:
    default_model: "gemini-pro"
  ollama:
    base_url: "http://localhost:11434"
    default_model: "codellama"

display:
  show_delegations: true
  show_model_switches: true
  color_code_models: true
```

## Success Metrics

- **Transparency**: Every model interaction is clearly visible
- **Simplicity**: Easy to add new providers and models
- **Flexibility**: Support for different conversation patterns
- **Extensibility**: Simple to add new tools and capabilities
- **Usability**: Clear API for model selection and management

This design provides a solid foundation for building a multi-provider, multi-model coding agent with full transparency and easy extensibility.