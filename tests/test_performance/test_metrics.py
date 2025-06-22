"""Tests for performance metrics data structures."""

import json
import pytest
from datetime import datetime
from performance.metrics import PerformanceMetrics, TaskType


class TestTaskType:
    """Test TaskType enum."""
    
    def test_task_type_values(self):
        """Test that all task types have correct string values."""
        assert TaskType.CODE_GENERATION.value == "code_generation"
        assert TaskType.CODE_REVIEW.value == "code_review"
        assert TaskType.EXPLANATION.value == "explanation"
        assert TaskType.DEBUGGING.value == "debugging"
    
    def test_task_type_from_string(self):
        """Test creating TaskType from string value."""
        assert TaskType("code_generation") == TaskType.CODE_GENERATION
        assert TaskType("explanation") == TaskType.EXPLANATION


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_basic_creation(self):
        """Test basic PerformanceMetrics creation with required fields."""
        metrics = PerformanceMetrics(
            inference_time=2.5,
            input_tokens=100,
            output_tokens=50,
            model_name="test-model"
        )
        
        assert metrics.inference_time == 2.5
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.model_name == "test-model"
        assert metrics.task_type == TaskType.GENERAL  # default value
        assert isinstance(metrics.timestamp, datetime)
    
    def test_tokens_per_second_calculation(self):
        """Test automatic tokens per second calculation."""
        metrics = PerformanceMetrics(
            inference_time=2.0,
            input_tokens=100,
            output_tokens=40,  # 40 tokens in 2 seconds = 20 tokens/sec
            model_name="test-model"
        )
        
        assert metrics.tokens_per_second == 20.0
    
    def test_total_tokens_property(self):
        """Test total_tokens property calculation."""
        metrics = PerformanceMetrics(
            inference_time=1.0,
            input_tokens=100,
            output_tokens=50,
            model_name="test-model"
        )
        
        assert metrics.total_tokens == 150
    
    def test_efficiency_score_property(self):
        """Test efficiency_score property."""
        metrics = PerformanceMetrics(
            inference_time=2.0,
            input_tokens=100,
            output_tokens=40,
            model_name="test-model"
        )
        
        assert metrics.efficiency_score == 20.0  # Same as tokens_per_second
    
    def test_cost_estimate_property(self):
        """Test cost_estimate property calculation."""
        metrics = PerformanceMetrics(
            inference_time=1.0,
            input_tokens=1000,
            output_tokens=500,
            model_name="test-model"
        )
        
        # (1000 * 0.001 + 500 * 0.002) / 1000 = 0.002
        assert metrics.cost_estimate == 0.002
    
    def test_with_tool_usage(self):
        """Test metrics with tool usage information."""
        metrics = PerformanceMetrics(
            inference_time=3.0,
            input_tokens=100,
            output_tokens=50,
            model_name="test-model",
            tool_calls_count=2,
            tool_execution_time=0.5,
            tools_used=["read_file", "grep_files"]
        )
        
        assert metrics.tool_calls_count == 2
        assert metrics.tool_execution_time == 0.5
        assert metrics.tools_used == ["read_file", "grep_files"]
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metrics = PerformanceMetrics(
            inference_time=2.5,
            input_tokens=100,
            output_tokens=50,
            model_name="test-model",
            task_type=TaskType.CODE_GENERATION,
            timestamp=timestamp,
            tools_used=["read_file"]
        )
        
        result = metrics.to_dict()
        
        assert result["inference_time"] == 2.5
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["model_name"] == "test-model"
        assert result["task_type"] == "code_generation"
        assert result["timestamp"] == "2024-01-01T12:00:00"
        assert result["tools_used"] == ["read_file"]
    
    def test_to_json(self):
        """Test JSON serialization."""
        metrics = PerformanceMetrics(
            inference_time=2.5,
            input_tokens=100,
            output_tokens=50,
            model_name="test-model"
        )
        
        json_str = metrics.to_json()
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["inference_time"] == 2.5
        assert data["model_name"] == "test-model"
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "inference_time": 2.5,
            "input_tokens": 100,
            "output_tokens": 50,
            "model_name": "test-model",
            "task_type": "code_generation",
            "timestamp": "2024-01-01T12:00:00",
            "tools_used": ["read_file"]
        }
        
        metrics = PerformanceMetrics.from_dict(data)
        
        assert metrics.inference_time == 2.5
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.model_name == "test-model"
        assert metrics.task_type == TaskType.CODE_GENERATION
        assert metrics.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert metrics.tools_used == ["read_file"]
    
    def test_from_json(self):
        """Test creation from JSON string."""
        json_data = {
            "inference_time": 2.5,
            "input_tokens": 100,
            "output_tokens": 50,
            "model_name": "test-model",
            "task_type": "code_generation",
            "timestamp": "2024-01-01T12:00:00"
        }
        json_str = json.dumps(json_data)
        
        metrics = PerformanceMetrics.from_json(json_str)
        
        assert metrics.inference_time == 2.5
        assert metrics.model_name == "test-model"
        assert metrics.task_type == TaskType.CODE_GENERATION
    
    def test_round_trip_serialization(self):
        """Test that serialization and deserialization preserve data."""
        original = PerformanceMetrics(
            inference_time=2.5,
            load_duration=0.3,
            eval_duration=2.2,
            input_tokens=100,
            output_tokens=50,
            model_name="test-model",
            task_type=TaskType.CODE_GENERATION,
            memory_usage=512.0,
            cpu_usage=75.5,
            tool_calls_count=1,
            tools_used=["read_file"],
            provider="ollama"
        )
        
        # Convert to JSON and back
        json_str = original.to_json()
        restored = PerformanceMetrics.from_json(json_str)
        
        # Compare all important fields
        assert restored.inference_time == original.inference_time
        assert restored.load_duration == original.load_duration
        assert restored.eval_duration == original.eval_duration
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.model_name == original.model_name
        assert restored.task_type == original.task_type
        assert restored.memory_usage == original.memory_usage
        assert restored.cpu_usage == original.cpu_usage
        assert restored.tool_calls_count == original.tool_calls_count
        assert restored.tools_used == original.tools_used
        assert restored.provider == original.provider
    
    def test_string_representations(self):
        """Test string and repr methods."""
        metrics = PerformanceMetrics(
            inference_time=2.5,
            input_tokens=100,
            output_tokens=50,
            model_name="test-model"
        )
        
        str_repr = str(metrics)
        assert "test-model" in str_repr
        assert "2.50s" in str_repr
        assert "20.0" in str_repr  # tokens per second
        
        repr_str = repr(metrics)
        assert "PerformanceMetrics" in repr_str
        assert "test-model" in repr_str
        assert "2.5" in repr_str
    
    def test_zero_inference_time_handling(self):
        """Test handling of zero inference time."""
        metrics = PerformanceMetrics(
            inference_time=0.0,
            input_tokens=100,
            output_tokens=50,
            model_name="test-model"
        )
        
        # Should not cause division by zero
        assert metrics.tokens_per_second == 0.0
    
    def test_none_tools_used_handling(self):
        """Test handling of None tools_used in from_dict."""
        data = {
            "inference_time": 2.5,
            "input_tokens": 100,
            "output_tokens": 50,
            "model_name": "test-model",
            "tools_used": None
        }
        
        metrics = PerformanceMetrics.from_dict(data)
        assert metrics.tools_used == []