"""Performance metrics data structures and utilities."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any


class TaskType(Enum):
    """Types of tasks that can be performed by the coding agent."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTORING = "code_refactoring"
    EXPLANATION = "explanation"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    FILE_OPERATION = "file_operation"
    SEARCH = "search"
    GENERAL = "general"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for LLM inference runs.
    
    This class captures timing, token usage, resource consumption, and context
    information for each model inference to enable performance analysis and
    optimization.
    """
    
    # Timing metrics (in seconds)
    inference_time: float  # Total inference time
    load_duration: Optional[float] = None  # Model load time (Ollama specific)
    eval_duration: Optional[float] = None  # Evaluation time (Ollama specific)
    
    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    
    # Resource metrics
    memory_usage: Optional[float] = None  # Memory usage in MB
    cpu_usage: Optional[float] = None  # CPU usage percentage
    
    # Context information
    model_name: str = ""
    task_type: TaskType = TaskType.GENERAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_length: int = 0  # Character length of input
    response_length: int = 0  # Character length of output
    
    # Tool usage metrics
    tool_calls_count: int = 0
    tool_execution_time: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    
    # Additional metadata
    provider: str = ""  # e.g., "ollama", "anthropic", "gemini"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        # Only calculate tokens_per_second if it wasn't already set
        if self.tokens_per_second == 0.0 and self.output_tokens > 0 and self.inference_time > 0:
            self.tokens_per_second = self.output_tokens / self.inference_time
    
    @property
    def total_tokens(self) -> int:
        """Total tokens processed (input + output)."""
        return self.input_tokens + self.output_tokens
    
    @property
    def efficiency_score(self) -> float:
        """Simple efficiency score based on tokens per second."""
        return self.tokens_per_second
    
    @property
    def cost_estimate(self) -> float:
        """Rough cost estimate based on token usage (placeholder for future pricing)."""
        # This is a placeholder - would be implemented with actual pricing data
        return (self.input_tokens * 0.001 + self.output_tokens * 0.002) / 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, TaskType):
                result[key] = value.value
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create PerformanceMetrics from dictionary."""
        # Handle datetime conversion
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # Handle TaskType conversion
        if "task_type" in data and isinstance(data["task_type"], str):
            data["task_type"] = TaskType(data["task_type"])
        
        # Handle lists that might be None
        if "tools_used" in data and data["tools_used"] is None:
            data["tools_used"] = []
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PerformanceMetrics":
        """Create PerformanceMetrics from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"PerformanceMetrics(model={self.model_name}, "
            f"inference_time={self.inference_time:.2f}s, "
            f"tokens/sec={self.tokens_per_second:.1f}, "
            f"tokens={self.total_tokens})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"PerformanceMetrics("
            f"model_name='{self.model_name}', "
            f"task_type={self.task_type}, "
            f"inference_time={self.inference_time}, "
            f"input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"tokens_per_second={self.tokens_per_second:.2f}, "
            f"timestamp='{self.timestamp.isoformat()}'"
            f")"
        )