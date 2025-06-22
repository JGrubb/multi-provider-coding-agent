# PERF-002: Create SQLite database schema for performance data

**Type**: Technical Task  
**Priority**: High  
**Estimate**: 3 hours  
**Milestone**: 1 - Performance Monitoring Foundation

## Description
Design and implement SQLite database schema to store performance metrics data. Include database operations for inserting, querying, and analyzing performance data over time.

## Acceptance Criteria
- [ ] SQLite database schema supports all PerformanceMetrics fields
- [ ] Database operations class with CRUD methods
- [ ] Efficient indexing for common queries
- [ ] Database migration support
- [ ] Connection pooling and error handling
- [ ] Unit tests for all database operations

## Technical Requirements
- SQLite database with proper normalization
- Indexes on frequently queried columns (model_name, timestamp, task_type)
- Support for bulk inserts for performance
- Connection management with proper cleanup
- SQL injection protection

## Schema Design
```sql
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    inference_time REAL NOT NULL,
    load_duration REAL,
    eval_duration REAL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    tokens_per_second REAL NOT NULL,
    memory_usage REAL,
    cpu_usage REAL,
    query_length INTEGER NOT NULL,
    response_length INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_timestamp ON performance_metrics(model_name, timestamp);
CREATE INDEX idx_task_type ON performance_metrics(task_type);
```

## Files to Create/Modify
- `performance/database.py` - Database operations class
- `performance/schema.sql` - Database schema
- `tests/test_performance_database.py` - Unit tests

## Dependencies
- PERF-001 (PerformanceMetrics data structure)

## Definition of Done
- Database schema created and tested
- All CRUD operations implemented
- Proper indexing for performance
- Error handling and connection management
- Unit tests cover all database operations
- Database can be created/migrated successfully