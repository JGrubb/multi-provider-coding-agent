"""SQLite database operations for performance metrics storage."""

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator

from .metrics import PerformanceMetrics, TaskType


class PerformanceDatabase:
    """SQLite database interface for storing and querying performance metrics."""
    
    def __init__(self, db_path: str = "performance.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._schema_path = Path(__file__).parent / "schema.sql"
        
        # Initialize database schema
        self._initialize_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            # Enable foreign keys and WAL mode for better performance
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            # Return rows as dictionaries
            self._local.connection.row_factory = sqlite3.Row
        
        return self._local.connection
    
    @contextmanager
    def _get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database cursor with automatic cleanup."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _initialize_database(self) -> None:
        """Initialize database schema from SQL file."""
        if not self._schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self._schema_path}")
        
        with open(self._schema_path, 'r') as f:
            schema_sql = f.read()
        
        with self._get_cursor() as cursor:
            cursor.executescript(schema_sql)
    
    def insert_metrics(self, metrics: PerformanceMetrics) -> int:
        """Insert performance metrics into database.
        
        Args:
            metrics: PerformanceMetrics instance to store
            
        Returns:
            ID of inserted record
        """
        # Convert tools_used list to JSON string
        tools_used_json = json.dumps(metrics.tools_used) if metrics.tools_used else None
        
        sql = """
        INSERT INTO performance_metrics (
            inference_time, load_duration, eval_duration,
            input_tokens, output_tokens, tokens_per_second,
            memory_usage, cpu_usage,
            model_name, task_type, timestamp, query_length, response_length,
            tool_calls_count, tool_execution_time, tools_used,
            provider, temperature, max_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            metrics.inference_time,
            metrics.load_duration,
            metrics.eval_duration,
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.tokens_per_second,
            metrics.memory_usage,
            metrics.cpu_usage,
            metrics.model_name,
            metrics.task_type.value,
            metrics.timestamp.isoformat(),
            metrics.query_length,
            metrics.response_length,
            metrics.tool_calls_count,
            metrics.tool_execution_time,
            tools_used_json,
            metrics.provider,
            metrics.temperature,
            metrics.max_tokens
        )
        
        with self._get_cursor() as cursor:
            cursor.execute(sql, values)
            return cursor.lastrowid
    
    def insert_metrics_bulk(self, metrics_list: List[PerformanceMetrics]) -> List[int]:
        """Insert multiple performance metrics in a single transaction.
        
        Args:
            metrics_list: List of PerformanceMetrics to store
            
        Returns:
            List of IDs of inserted records
        """
        if not metrics_list:
            return []
        
        sql = """
        INSERT INTO performance_metrics (
            inference_time, load_duration, eval_duration,
            input_tokens, output_tokens, tokens_per_second,
            memory_usage, cpu_usage,
            model_name, task_type, timestamp, query_length, response_length,
            tool_calls_count, tool_execution_time, tools_used,
            provider, temperature, max_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values_list = []
        for metrics in metrics_list:
            tools_used_json = json.dumps(metrics.tools_used) if metrics.tools_used else None
            values_list.append((
                metrics.inference_time,
                metrics.load_duration,
                metrics.eval_duration,
                metrics.input_tokens,
                metrics.output_tokens,
                metrics.tokens_per_second,
                metrics.memory_usage,
                metrics.cpu_usage,
                metrics.model_name,
                metrics.task_type.value,
                metrics.timestamp.isoformat(),
                metrics.query_length,
                metrics.response_length,
                metrics.tool_calls_count,
                metrics.tool_execution_time,
                tools_used_json,
                metrics.provider,
                metrics.temperature,
                metrics.max_tokens
            ))
        
        with self._get_cursor() as cursor:
            cursor.executemany(sql, values_list)
            # SQLite's lastrowid after executemany only returns the last inserted ID
            # We need to get the actual IDs differently
            if cursor.lastrowid is not None:
                # Approximate range based on last ID
                first_id = cursor.lastrowid - len(metrics_list) + 1
                return list(range(first_id, cursor.lastrowid + 1))
            else:
                # Fallback: get the most recent IDs
                cursor.execute(
                    "SELECT id FROM performance_metrics ORDER BY id DESC LIMIT ?",
                    (len(metrics_list),)
                )
                rows = cursor.fetchall()
                return [row['id'] for row in reversed(rows)]
    
    def get_metrics_by_id(self, metrics_id: int) -> Optional[PerformanceMetrics]:
        """Retrieve metrics by ID.
        
        Args:
            metrics_id: Database ID of metrics record
            
        Returns:
            PerformanceMetrics instance or None if not found
        """
        sql = "SELECT * FROM performance_metrics WHERE id = ?"
        
        with self._get_cursor() as cursor:
            cursor.execute(sql, (metrics_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_metrics(row)
            return None
    
    def get_metrics_by_model(
        self, 
        model_name: str, 
        limit: Optional[int] = None,
        task_type: Optional[TaskType] = None
    ) -> List[PerformanceMetrics]:
        """Retrieve metrics for a specific model.
        
        Args:
            model_name: Name of the model
            limit: Maximum number of records to return
            task_type: Optional task type filter
            
        Returns:
            List of PerformanceMetrics instances
        """
        sql = "SELECT * FROM performance_metrics WHERE model_name = ?"
        params = [model_name]
        
        if task_type:
            sql += " AND task_type = ?"
            params.append(task_type.value)
        
        sql += " ORDER BY timestamp DESC"
        
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        
        with self._get_cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            return [self._row_to_metrics(row) for row in rows]
    
    def get_recent_metrics(self, hours: int = 24) -> List[PerformanceMetrics]:
        """Get metrics from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent PerformanceMetrics instances
        """
        from datetime import datetime, timezone, timedelta
        
        # Calculate cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff_time.isoformat()
        
        sql = """
        SELECT * FROM performance_metrics 
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        """
        
        with self._get_cursor() as cursor:
            cursor.execute(sql, (cutoff_iso,))
            rows = cursor.fetchall()
            
            return [self._row_to_metrics(row) for row in rows]
    
    def get_performance_summary(self) -> List[Dict[str, Any]]:
        """Get performance summary statistics.
        
        Returns:
            List of summary statistics dictionaries
        """
        sql = "SELECT * FROM performance_summary ORDER BY total_runs DESC"
        
        with self._get_cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_average_performance(
        self,
        model_name: str,
        task_type: Optional[TaskType] = None,
        days: int = 7
    ) -> Optional[Dict[str, float]]:
        """Get average performance metrics for a model.
        
        Args:
            model_name: Name of the model
            task_type: Optional task type filter
            days: Number of days to analyze
            
        Returns:
            Dictionary with average performance metrics
        """
        sql = """
        SELECT 
            AVG(inference_time) as avg_inference_time,
            AVG(tokens_per_second) as avg_tokens_per_second,
            AVG(input_tokens) as avg_input_tokens,
            AVG(output_tokens) as avg_output_tokens,
            AVG(tool_calls_count) as avg_tool_calls,
            COUNT(*) as total_runs
        FROM performance_metrics 
        WHERE model_name = ? 
        AND timestamp >= datetime('now', '-{} days')
        """.format(days)
        
        params = [model_name]
        
        if task_type:
            sql += " AND task_type = ?"
            params.append(task_type.value)
        
        with self._get_cursor() as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone()
            
            if row and row['total_runs'] > 0:
                return dict(row)
            return None
    
    def delete_old_metrics(self, days: int = 30) -> int:
        """Delete metrics older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted records
        """
        sql = """
        DELETE FROM performance_metrics 
        WHERE timestamp < datetime('now', '-{} days')
        """.format(days)
        
        with self._get_cursor() as cursor:
            cursor.execute(sql)
            return cursor.rowcount
    
    def _row_to_metrics(self, row: sqlite3.Row) -> PerformanceMetrics:
        """Convert database row to PerformanceMetrics instance.
        
        Args:
            row: SQLite row object
            
        Returns:
            PerformanceMetrics instance
        """
        # Parse tools_used JSON
        tools_used = []
        if row['tools_used']:
            try:
                tools_used = json.loads(row['tools_used'])
            except json.JSONDecodeError:
                tools_used = []
        
        # Convert task_type string back to enum
        task_type = TaskType(row['task_type'])
        
        # Parse timestamp
        from datetime import datetime
        timestamp = datetime.fromisoformat(row['timestamp'])
        
        return PerformanceMetrics(
            inference_time=row['inference_time'],
            load_duration=row['load_duration'],
            eval_duration=row['eval_duration'],
            input_tokens=row['input_tokens'],
            output_tokens=row['output_tokens'],
            tokens_per_second=row['tokens_per_second'],
            memory_usage=row['memory_usage'],
            cpu_usage=row['cpu_usage'],
            model_name=row['model_name'],
            task_type=task_type,
            timestamp=timestamp,
            query_length=row['query_length'],
            response_length=row['response_length'],
            tool_calls_count=row['tool_calls_count'],
            tool_execution_time=row['tool_execution_time'],
            tools_used=tools_used,
            provider=row['provider'],
            temperature=row['temperature'],
            max_tokens=row['max_tokens']
        )
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()