"""Tests for performance database operations."""

import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from performance.database import PerformanceDatabase
from performance.metrics import PerformanceMetrics, TaskType


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = PerformanceDatabase(db_path)
    yield db
    
    # Cleanup
    db.close()
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_metrics():
    """Create sample performance metrics for testing."""
    return PerformanceMetrics(
        inference_time=2.5,
        load_duration=0.3,
        eval_duration=2.2,
        input_tokens=100,
        output_tokens=50,
        model_name="test-model",
        task_type=TaskType.CODE_GENERATION,
        query_length=500,
        response_length=250,
        tool_calls_count=2,
        tool_execution_time=0.5,
        tools_used=["read_file", "grep_files"],
        provider="ollama",
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def sample_metrics_list():
    """Create a list of sample metrics for bulk testing."""
    base_time = datetime.now(timezone.utc)
    
    metrics_list = []
    for i in range(5):
        metrics = PerformanceMetrics(
            inference_time=1.0 + i * 0.5,
            input_tokens=100 + i * 10,
            output_tokens=50 + i * 5,
            model_name=f"model-{i % 2}",  # Alternate between model-0 and model-1
            task_type=TaskType.CODE_GENERATION if i % 2 == 0 else TaskType.CODE_REVIEW,
            timestamp=base_time + timedelta(minutes=i),
            provider="ollama"
        )
        metrics_list.append(metrics)
    
    return metrics_list


class TestPerformanceDatabase:
    """Test PerformanceDatabase class."""
    
    def test_database_initialization(self, temp_db):
        """Test that database initializes correctly."""
        # Should not raise any exceptions
        assert temp_db.db_path.exists()
        
        # Test that schema was created by trying to insert data
        sample = PerformanceMetrics(
            inference_time=1.0,
            input_tokens=10,
            output_tokens=5,
            model_name="test"
        )
        record_id = temp_db.insert_metrics(sample)
        assert record_id > 0
    
    def test_insert_metrics(self, temp_db, sample_metrics):
        """Test inserting a single metrics record."""
        record_id = temp_db.insert_metrics(sample_metrics)
        
        assert record_id > 0
        assert isinstance(record_id, int)
    
    def test_insert_metrics_bulk(self, temp_db, sample_metrics_list):
        """Test bulk insertion of metrics."""
        record_ids = temp_db.insert_metrics_bulk(sample_metrics_list)
        
        assert len(record_ids) == len(sample_metrics_list)
        assert all(isinstance(id, int) for id in record_ids)
        assert all(id > 0 for id in record_ids)
    
    def test_insert_metrics_bulk_empty_list(self, temp_db):
        """Test bulk insertion with empty list."""
        record_ids = temp_db.insert_metrics_bulk([])
        assert record_ids == []
    
    def test_get_metrics_by_id(self, temp_db, sample_metrics):
        """Test retrieving metrics by ID."""
        # Insert metrics
        record_id = temp_db.insert_metrics(sample_metrics)
        
        # Retrieve metrics
        retrieved = temp_db.get_metrics_by_id(record_id)
        
        assert retrieved is not None
        assert retrieved.model_name == sample_metrics.model_name
        assert retrieved.inference_time == sample_metrics.inference_time
        assert retrieved.task_type == sample_metrics.task_type
        assert retrieved.tools_used == sample_metrics.tools_used
        assert retrieved.provider == sample_metrics.provider
    
    def test_get_metrics_by_id_not_found(self, temp_db):
        """Test retrieving non-existent metrics."""
        retrieved = temp_db.get_metrics_by_id(999999)
        assert retrieved is None
    
    def test_get_metrics_by_model(self, temp_db, sample_metrics_list):
        """Test retrieving metrics by model name."""
        # Insert test data
        temp_db.insert_metrics_bulk(sample_metrics_list)
        
        # Get metrics for model-0
        model_0_metrics = temp_db.get_metrics_by_model("model-0")
        
        # Should have 3 records for model-0 (indexes 0, 2, 4)
        assert len(model_0_metrics) == 3
        assert all(m.model_name == "model-0" for m in model_0_metrics)
        
        # Test with limit
        limited_metrics = temp_db.get_metrics_by_model("model-0", limit=2)
        assert len(limited_metrics) == 2
    
    def test_get_metrics_by_model_with_task_filter(self, temp_db, sample_metrics_list):
        """Test retrieving metrics by model and task type."""
        # Insert test data
        temp_db.insert_metrics_bulk(sample_metrics_list)
        
        # Get CODE_GENERATION metrics for model-0
        metrics = temp_db.get_metrics_by_model(
            "model-0", 
            task_type=TaskType.CODE_GENERATION
        )
        
        # Should have 3 records (indexes 0, 2, 4 are CODE_GENERATION for model-0)
        assert len(metrics) == 3
        assert all(m.task_type == TaskType.CODE_GENERATION for m in metrics)
        assert all(m.model_name == "model-0" for m in metrics)
    
    def test_get_recent_metrics(self, temp_db):
        """Test retrieving recent metrics."""
        # Create metrics with different timestamps
        now = datetime.now(timezone.utc)
        
        # Recent metric (within last hour)
        recent_metric = PerformanceMetrics(
            inference_time=1.0,
            input_tokens=10,
            output_tokens=5,
            model_name="recent",
            timestamp=now - timedelta(minutes=30)
        )
        
        # Old metric (25 hours ago)
        old_metric = PerformanceMetrics(
            inference_time=2.0,
            input_tokens=20,
            output_tokens=10,
            model_name="old",
            timestamp=now - timedelta(hours=25)
        )
        
        temp_db.insert_metrics(recent_metric)
        temp_db.insert_metrics(old_metric)
        
        # Get metrics from last 24 hours
        recent_metrics = temp_db.get_recent_metrics(hours=24)
        
        # Should only get the recent metric
        assert len(recent_metrics) == 1
        assert recent_metrics[0].model_name == "recent"
    
    def test_get_performance_summary(self, temp_db, sample_metrics_list):
        """Test getting performance summary statistics."""
        # Insert test data
        temp_db.insert_metrics_bulk(sample_metrics_list)
        
        summary = temp_db.get_performance_summary()
        
        assert len(summary) > 0
        
        # Check that summary contains expected fields
        for entry in summary:
            assert 'model_name' in entry
            assert 'task_type' in entry
            assert 'total_runs' in entry
            assert 'avg_inference_time' in entry
            assert 'avg_tokens_per_second' in entry
    
    def test_get_average_performance(self, temp_db, sample_metrics_list):
        """Test getting average performance for a model."""
        # Insert test data
        temp_db.insert_metrics_bulk(sample_metrics_list)
        
        # Get average performance for model-0
        avg_perf = temp_db.get_average_performance("model-0")
        
        assert avg_perf is not None
        assert 'avg_inference_time' in avg_perf
        assert 'avg_tokens_per_second' in avg_perf
        assert 'total_runs' in avg_perf
        assert avg_perf['total_runs'] == 3  # model-0 appears 3 times
    
    def test_get_average_performance_with_task_filter(self, temp_db, sample_metrics_list):
        """Test getting average performance with task type filter."""
        # Insert test data
        temp_db.insert_metrics_bulk(sample_metrics_list)
        
        # Get average performance for model-0, CODE_GENERATION tasks only
        avg_perf = temp_db.get_average_performance(
            "model-0", 
            task_type=TaskType.CODE_GENERATION
        )
        
        assert avg_perf is not None
        assert avg_perf['total_runs'] == 3  # model-0 CODE_GENERATION tasks
    
    def test_get_average_performance_no_data(self, temp_db):
        """Test getting average performance for non-existent model."""
        avg_perf = temp_db.get_average_performance("non-existent-model")
        assert avg_perf is None
    
    def test_delete_old_metrics(self, temp_db):
        """Test deleting old metrics."""
        now = datetime.now(timezone.utc)
        
        # Create metrics with different ages
        recent_metric = PerformanceMetrics(
            inference_time=1.0,
            input_tokens=10,
            output_tokens=5,
            model_name="recent",
            timestamp=now - timedelta(days=5)
        )
        
        old_metric = PerformanceMetrics(
            inference_time=2.0,
            input_tokens=20,
            output_tokens=10,
            model_name="old",
            timestamp=now - timedelta(days=35)
        )
        
        temp_db.insert_metrics(recent_metric)
        temp_db.insert_metrics(old_metric)
        
        # Delete metrics older than 30 days
        deleted_count = temp_db.delete_old_metrics(days=30)
        
        assert deleted_count == 1  # Should delete the old metric
        
        # Verify the recent metric is still there
        recent_metrics = temp_db.get_recent_metrics(hours=24*7)  # Last week
        assert len(recent_metrics) == 1
        assert recent_metrics[0].model_name == "recent"
    
    def test_context_manager(self):
        """Test using database as context manager."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            with PerformanceDatabase(db_path) as db:
                # Should work normally within context
                sample = PerformanceMetrics(
                    inference_time=1.0,
                    input_tokens=10,
                    output_tokens=5,
                    model_name="test"
                )
                record_id = db.insert_metrics(sample)
                assert record_id > 0
            
            # Database should be closed after context
            # (We can't easily test this without accessing private attributes)
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_tools_used_serialization(self, temp_db):
        """Test that tools_used list is properly serialized/deserialized."""
        # Test with tools
        metrics_with_tools = PerformanceMetrics(
            inference_time=1.0,
            input_tokens=10,
            output_tokens=5,
            model_name="test",
            tools_used=["read_file", "grep_files", "run_command"]
        )
        
        record_id = temp_db.insert_metrics(metrics_with_tools)
        retrieved = temp_db.get_metrics_by_id(record_id)
        
        assert retrieved.tools_used == ["read_file", "grep_files", "run_command"]
        
        # Test with empty tools
        metrics_no_tools = PerformanceMetrics(
            inference_time=1.0,
            input_tokens=10,
            output_tokens=5,
            model_name="test",
            tools_used=[]
        )
        
        record_id = temp_db.insert_metrics(metrics_no_tools)
        retrieved = temp_db.get_metrics_by_id(record_id)
        
        assert retrieved.tools_used == []
    
    def test_timestamp_handling(self, temp_db):
        """Test proper timestamp handling."""
        # Create metrics with specific timestamp
        specific_time = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        metrics = PerformanceMetrics(
            inference_time=1.0,
            input_tokens=10,
            output_tokens=5,
            model_name="test",
            timestamp=specific_time
        )
        
        record_id = temp_db.insert_metrics(metrics)
        retrieved = temp_db.get_metrics_by_id(record_id)
        
        # Timestamps should match (may lose microseconds in SQLite)
        assert retrieved.timestamp.replace(microsecond=0) == specific_time.replace(microsecond=0)