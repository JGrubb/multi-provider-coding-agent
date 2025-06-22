-- SQLite database schema for performance metrics storage
-- This schema supports all fields from the PerformanceMetrics dataclass

CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Timing metrics (in seconds)
    inference_time REAL NOT NULL,
    load_duration REAL,
    eval_duration REAL,
    
    -- Token metrics
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    tokens_per_second REAL NOT NULL DEFAULT 0.0,
    
    -- Resource metrics
    memory_usage REAL,  -- Memory usage in MB
    cpu_usage REAL,     -- CPU usage percentage
    
    -- Context information
    model_name TEXT NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'general',
    timestamp DATETIME NOT NULL,
    query_length INTEGER NOT NULL DEFAULT 0,
    response_length INTEGER NOT NULL DEFAULT 0,
    
    -- Tool usage metrics
    tool_calls_count INTEGER NOT NULL DEFAULT 0,
    tool_execution_time REAL NOT NULL DEFAULT 0.0,
    tools_used TEXT,  -- JSON array of tool names
    
    -- Provider information
    provider TEXT NOT NULL DEFAULT '',
    temperature REAL,
    max_tokens INTEGER,
    
    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_model_timestamp ON performance_metrics(model_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_task_type ON performance_metrics(task_type);
CREATE INDEX IF NOT EXISTS idx_provider ON performance_metrics(provider);
CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_task ON performance_metrics(model_name, task_type);

-- Index for performance analysis queries
CREATE INDEX IF NOT EXISTS idx_performance_analysis ON performance_metrics(
    model_name, task_type, timestamp, tokens_per_second
);

-- View for summary statistics
CREATE VIEW IF NOT EXISTS performance_summary AS
SELECT 
    model_name,
    task_type,
    provider,
    COUNT(*) as total_runs,
    AVG(inference_time) as avg_inference_time,
    AVG(tokens_per_second) as avg_tokens_per_second,
    AVG(input_tokens) as avg_input_tokens,
    AVG(output_tokens) as avg_output_tokens,
    MIN(timestamp) as first_run,
    MAX(timestamp) as last_run
FROM performance_metrics 
GROUP BY model_name, task_type, provider;

-- View for recent performance data
CREATE VIEW IF NOT EXISTS recent_performance AS
SELECT 
    id,
    model_name,
    task_type,
    inference_time,
    tokens_per_second,
    input_tokens + output_tokens as total_tokens,
    tool_calls_count,
    timestamp
FROM performance_metrics 
WHERE timestamp >= datetime('now', '-24 hours')
ORDER BY timestamp DESC;