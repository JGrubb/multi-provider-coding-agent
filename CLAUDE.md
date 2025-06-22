# Project Context for Claude

## Development Guidelines

### Git Commit Messages
- NEVER mention "Claude Code" in commit messages or PR descriptions
- Use standard conventional commit format
- Focus on the actual changes and their purpose
- Keep commits focused and logical

### Code Style
- Follow Python PEP 8 conventions
- Use type hints throughout
- Document all public APIs
- Prefer composition over inheritance

### Project Structure
- Keep modules small and focused
- Use clear, descriptive naming
- Separate concerns (providers, tools, performance, etc.)
- Test everything thoroughly

### Performance Focus
- Always measure before optimizing
- Track performance metrics for all Ollama interactions
- Prioritize local model efficiency
- Make performance data visible to users