# PERF-005: Build simple CLI for testing performance features

**Type**: Feature  
**Priority**: Medium  
**Estimate**: 3 hours  
**Milestone**: 1 - Performance Monitoring Foundation

## Description
Create a command-line interface for manual testing and validation of performance monitoring features. This CLI will allow developers and users to interact with the agent, test performance tracking, and validate functionality.

## Acceptance Criteria
- [ ] Interactive CLI with clear commands
- [ ] Performance metrics displayed after each inference
- [ ] Model switching commands
- [ ] Performance history queries
- [ ] Help system and command documentation
- [ ] Graceful error handling and user feedback
- [ ] Manual testing validates all Milestone 1 features

## Technical Requirements
- Command-line argument parsing
- Interactive command loop
- Real-time performance display
- Model status and availability checking
- History querying and formatting
- Clear user feedback and error messages

## CLI Commands
```bash
# Basic usage
python cli.py chat "Write a hello world function"
python cli.py use-model ollama:codellama
python cli.py list-models

# Performance features
python cli.py show-performance
python cli.py performance-history --model ollama:codellama
python cli.py performance-stats

# Interactive mode
python cli.py interactive
> chat: Write a Python function
> use-model: ollama:codellama
> performance: show
> history: performance
> help
> exit
```

## Display Features
- Real-time performance metrics after each response
- Performance comparison tables
- Model availability status
- Response time and throughput indicators
- Visual indicators for performance thresholds

## Files to Create/Modify
- `cli.py` - Main CLI implementation
- `display/cli_display.py` - CLI-specific display formatting
- `tests/test_cli.py` - CLI testing
- `README.md` - Usage documentation

## Dependencies
- PERF-004 (Agent framework)
- All previous performance components

## User Experience Goals
- Easy to use for testing and validation
- Clear feedback on performance metrics
- Helpful error messages and guidance
- Discoverable commands and features

## Definition of Done
- CLI successfully demonstrates all Milestone 1 features
- Performance metrics clearly displayed and accurate
- Model switching works through CLI
- Error handling provides helpful feedback
- Documentation covers all CLI features
- Manual testing confirms end-to-end functionality