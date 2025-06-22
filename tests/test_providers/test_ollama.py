"""Tests for Ollama provider with performance tracking."""

import json
import pytest
import requests
import requests_mock
from unittest.mock import patch, MagicMock

from providers.ollama import OllamaProvider, OllamaError
from performance.metrics import TaskType


@pytest.fixture
def ollama_provider():
    """Create OllamaProvider instance for testing."""
    return OllamaProvider(
        base_url="http://localhost:11434",
        default_model="test-model",
        timeout=30
    )


@pytest.fixture
def sample_ollama_response():
    """Sample response from Ollama API."""
    return {
        "model": "test-model",
        "created_at": "2024-01-01T12:00:00Z",
        "response": "Hello! How can I help you today?",
        "done": True,
        "context": [1, 2, 3, 4, 5],
        "total_duration": 2500000000,  # 2.5 seconds in nanoseconds
        "load_duration": 300000000,   # 0.3 seconds
        "eval_duration": 2200000000,  # 2.2 seconds
        "eval_count": 50,             # Output tokens
        "prompt_eval_count": 100      # Input tokens
    }


class TestOllamaProvider:
    """Test OllamaProvider class."""
    
    def test_initialization(self, ollama_provider):
        """Test provider initialization."""
        assert ollama_provider.base_url == "http://localhost:11434"
        assert ollama_provider.default_model == "test-model"
        assert ollama_provider.timeout == 30
        assert ollama_provider.get_provider_name() == "ollama"
        assert ollama_provider.get_default_model() == "test-model"
    
    def test_initialization_with_trailing_slash(self):
        """Test URL normalization removes trailing slash."""
        provider = OllamaProvider(base_url="http://localhost:11434/")
        assert provider.base_url == "http://localhost:11434"
    
    def test_messages_to_prompt(self, ollama_provider):
        """Test conversion of messages to prompt string."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        prompt = ollama_provider._messages_to_prompt(messages)
        
        expected = (
            "System: You are a helpful assistant.\n\n"
            "User: Hello!\n\n"
            "Assistant: Hi there!\n\n"
            "User: How are you?"
        )
        
        assert prompt == expected
    
    def test_messages_to_prompt_empty(self, ollama_provider):
        """Test empty messages list."""
        prompt = ollama_provider._messages_to_prompt([])
        assert prompt == ""
    
    def test_messages_to_prompt_unknown_role(self, ollama_provider):
        """Test handling of unknown message roles."""
        messages = [
            {"role": "unknown", "content": "Some content"},
            {"content": "Content without role"}
        ]
        
        prompt = ollama_provider._messages_to_prompt(messages)
        assert "Some content" in prompt
        assert "Content without role" in prompt
    
    @patch('psutil.Process')
    def test_get_memory_usage(self, mock_process, ollama_provider):
        """Test memory usage measurement."""
        # Mock memory info (RSS in bytes)
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        memory_mb = ollama_provider._get_memory_usage()
        assert memory_mb == 100.0
    
    @patch('psutil.Process')
    def test_get_memory_usage_error(self, mock_process, ollama_provider):
        """Test memory usage measurement error handling."""
        mock_process.side_effect = Exception("Memory error")
        
        memory_mb = ollama_provider._get_memory_usage()
        assert memory_mb is None
    
    @patch('psutil.cpu_percent')
    def test_get_cpu_usage(self, mock_cpu_percent, ollama_provider):
        """Test CPU usage measurement."""
        mock_cpu_percent.return_value = 75.5
        
        cpu_usage = ollama_provider._get_cpu_usage()
        assert cpu_usage == 75.5
        mock_cpu_percent.assert_called_once_with(interval=None)
    
    @patch('psutil.cpu_percent')
    def test_get_cpu_usage_error(self, mock_cpu_percent, ollama_provider):
        """Test CPU usage measurement error handling."""
        mock_cpu_percent.side_effect = Exception("CPU error")
        
        cpu_usage = ollama_provider._get_cpu_usage()
        assert cpu_usage is None
    
    def test_extract_performance_metrics(self, ollama_provider, sample_ollama_response):
        """Test performance metrics extraction from Ollama response."""
        start_time = 1000.0
        end_time = 1002.5
        
        metrics = ollama_provider._extract_performance_metrics(
            response_data=sample_ollama_response,
            model="test-model",
            task_type=TaskType.CODE_GENERATION,
            prompt="Test prompt",
            response_text="Test response",
            start_time=start_time,
            end_time=end_time,
            start_memory=100.0,
            end_memory=120.0,
            start_cpu=10.0,
            end_cpu=25.0,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Check timing metrics
        assert metrics.inference_time == 2.5  # total_duration converted from nanoseconds
        assert metrics.load_duration == 0.3
        assert metrics.eval_duration == 2.2
        
        # Check token metrics
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert abs(metrics.tokens_per_second - (50 / 2.2)) < 0.01  # eval_count / eval_duration
        
        # Check resource metrics
        assert metrics.memory_usage == 20.0  # end - start
        assert metrics.cpu_usage == 15.0     # end - start
        
        # Check context information
        assert metrics.model_name == "test-model"
        assert metrics.task_type == TaskType.CODE_GENERATION
        assert metrics.query_length == len("Test prompt")
        assert metrics.response_length == len("Test response")
        assert metrics.provider == "ollama"
        assert metrics.temperature == 0.7
        assert metrics.max_tokens == 1000
    
    def test_extract_performance_metrics_minimal_data(self, ollama_provider):
        """Test performance metrics with minimal Ollama response data."""
        minimal_response = {
            "response": "Hello",
            "done": True
        }
        
        metrics = ollama_provider._extract_performance_metrics(
            response_data=minimal_response,
            model="test-model",
            task_type=TaskType.GENERAL,
            prompt="Hi",
            response_text="Hello",
            start_time=1000.0,
            end_time=1001.0,
            start_memory=None,
            end_memory=None,
            start_cpu=None,
            end_cpu=None,
            temperature=None,
            max_tokens=None
        )
        
        # Should use our timing measurement when Ollama data is missing
        assert metrics.inference_time == 1.0
        assert metrics.load_duration is None
        assert metrics.eval_duration is None
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.tokens_per_second == 0.0
        assert metrics.memory_usage is None
        assert metrics.cpu_usage is None
    
    def test_chat_success(self, ollama_provider, sample_ollama_response):
        """Test successful chat interaction."""
        messages = [{"role": "user", "content": "Hello!"}]
        
        with requests_mock.Mocker() as m:
            m.post(
                "http://localhost:11434/api/generate",
                json=sample_ollama_response
            )
            
            with patch.object(ollama_provider, '_get_memory_usage', return_value=100.0), \
                 patch.object(ollama_provider, '_get_cpu_usage', return_value=10.0):
                
                response_text, metrics = ollama_provider.chat(
                    messages=messages,
                    model="test-model",
                    task_type=TaskType.CODE_GENERATION,
                    temperature=0.7,
                    max_tokens=1000
                )
        
        # Check response
        assert response_text == "Hello! How can I help you today?"
        
        # Check metrics
        assert metrics.model_name == "test-model"
        assert metrics.task_type == TaskType.CODE_GENERATION
        assert metrics.provider == "ollama"
        assert metrics.temperature == 0.7
        assert metrics.max_tokens == 1000
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        
        # Check that metrics were stored
        assert ollama_provider.get_performance_metrics() == metrics
    
    def test_chat_with_default_model(self, ollama_provider, sample_ollama_response):
        """Test chat using default model."""
        messages = [{"role": "user", "content": "Hello!"}]
        
        with requests_mock.Mocker() as m:
            m.post(
                "http://localhost:11434/api/generate",
                json=sample_ollama_response
            )
            
            with patch.object(ollama_provider, '_get_memory_usage', return_value=100.0), \
                 patch.object(ollama_provider, '_get_cpu_usage', return_value=10.0):
                
                response_text, metrics = ollama_provider.chat(messages=messages)
        
        # Should use default model
        assert metrics.model_name == "test-model"
    
    def test_chat_request_error(self, ollama_provider):
        """Test chat with request error."""
        messages = [{"role": "user", "content": "Hello!"}]
        
        with requests_mock.Mocker() as m:
            m.post(
                "http://localhost:11434/api/generate",
                exc=requests.exceptions.ConnectionError("Connection failed")
            )
            
            with pytest.raises(OllamaError, match="Failed to communicate with Ollama"):
                ollama_provider.chat(messages=messages)
    
    def test_chat_http_error(self, ollama_provider):
        """Test chat with HTTP error response."""
        messages = [{"role": "user", "content": "Hello!"}]
        
        with requests_mock.Mocker() as m:
            m.post(
                "http://localhost:11434/api/generate",
                status_code=500,
                text="Internal Server Error"
            )
            
            with pytest.raises(OllamaError, match="Failed to communicate with Ollama"):
                ollama_provider.chat(messages=messages)
    
    def test_list_models_success(self, ollama_provider):
        """Test successful model listing."""
        mock_response = {
            "models": [
                {"name": "llama2:latest", "size": 123456},
                {"name": "codellama:7b", "size": 789012},
                {"name": "llama2:13b", "size": 345678},
            ]
        }
        
        with requests_mock.Mocker() as m:
            m.get("http://localhost:11434/api/tags", json=mock_response)
            
            models = ollama_provider.list_models()
        
        # Should remove tags and deduplicate
        expected_models = ["codellama", "llama2"]
        assert sorted(models) == sorted(expected_models)
    
    def test_list_models_empty(self, ollama_provider):
        """Test listing models when none available."""
        mock_response = {"models": []}
        
        with requests_mock.Mocker() as m:
            m.get("http://localhost:11434/api/tags", json=mock_response)
            
            models = ollama_provider.list_models()
        
        assert models == []
    
    def test_list_models_error(self, ollama_provider):
        """Test model listing with request error."""
        with requests_mock.Mocker() as m:
            m.get(
                "http://localhost:11434/api/tags",
                exc=requests.exceptions.ConnectionError("Connection failed")
            )
            
            with pytest.raises(OllamaError, match="Failed to list models"):
                ollama_provider.list_models()
    
    def test_get_model_info_success(self, ollama_provider):
        """Test successful model info retrieval."""
        mock_response = {
            "details": {
                "family": "llama",
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            },
            "modelfile": "FROM llama2:latest",
            "parameters": {"temperature": 0.8},
            "template": "{{ .Prompt }}"
        }
        
        with requests_mock.Mocker() as m:
            m.post("http://localhost:11434/api/show", json=mock_response)
            
            info = ollama_provider.get_model_info("llama2")
        
        assert info is not None
        assert info["name"] == "llama2"
        assert info["provider"] == "ollama"
        assert info["available"] is True
        assert info["details"]["family"] == "llama"
    
    def test_get_model_info_not_found(self, ollama_provider):
        """Test model info for non-existent model."""
        with requests_mock.Mocker() as m:
            m.post(
                "http://localhost:11434/api/show",
                status_code=404,
                text="Model not found"
            )
            
            info = ollama_provider.get_model_info("non-existent")
        
        assert info is None
    
    def test_is_available_success(self, ollama_provider):
        """Test availability check when Ollama is available."""
        with requests_mock.Mocker() as m:
            m.get("http://localhost:11434/api/tags", json={"models": []})
            
            assert ollama_provider.is_available() is True
    
    def test_is_available_failure(self, ollama_provider):
        """Test availability check when Ollama is not available."""
        with requests_mock.Mocker() as m:
            m.get(
                "http://localhost:11434/api/tags",
                exc=requests.exceptions.ConnectionError("Connection failed")
            )
            
            assert ollama_provider.is_available() is False
    
    def test_pull_model_success(self, ollama_provider):
        """Test successful model pulling."""
        with requests_mock.Mocker() as m:
            m.post("http://localhost:11434/api/pull", json={"status": "success"})
            
            result = ollama_provider.pull_model("llama2")
        
        assert result is True
    
    def test_pull_model_failure(self, ollama_provider):
        """Test model pulling failure."""
        with requests_mock.Mocker() as m:
            m.post(
                "http://localhost:11434/api/pull",
                status_code=500,
                text="Pull failed"
            )
            
            result = ollama_provider.pull_model("non-existent")
        
        assert result is False
    
    def test_get_performance_metrics_none(self, ollama_provider):
        """Test getting performance metrics when none available."""
        assert ollama_provider.get_performance_metrics() is None
    
    def test_chat_request_data_formatting(self, ollama_provider, sample_ollama_response):
        """Test that chat request data is formatted correctly."""
        messages = [{"role": "user", "content": "Hello!"}]
        
        with requests_mock.Mocker() as m:
            m.post("http://localhost:11434/api/generate", json=sample_ollama_response)
            
            with patch.object(ollama_provider, '_get_memory_usage', return_value=100.0), \
                 patch.object(ollama_provider, '_get_cpu_usage', return_value=10.0):
                
                ollama_provider.chat(
                    messages=messages,
                    model="custom-model",
                    temperature=0.5,
                    max_tokens=500,
                    custom_param="test"
                )
            
            # Check the request that was made
            assert len(m.request_history) == 1
            request = m.request_history[0]
            request_data = json.loads(request.text)
            
            assert request_data["model"] == "custom-model"
            assert request_data["prompt"] == "User: Hello!"
            assert request_data["stream"] is False
            assert request_data["temperature"] == 0.5
            assert request_data["num_predict"] == 500  # max_tokens -> num_predict
            assert request_data["custom_param"] == "test"