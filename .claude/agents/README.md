# Custom Agents for Trading Bot

This directory contains specialized Claude Code agents tailored for the trading bot project. These agents provide domain-specific expertise for various aspects of the system.

## Available Agents

### Core Development
- **api-designer** - Expert in designing robust, scalable APIs for data feeds and execution
- **backend-developer** - Specialized in building event-driven backend systems
- **python-pro** - Expert Python developer with focus on type safety, async programming, and data science

### Quality & Testing
- **code-reviewer** - Ensures code quality, identifies issues, and enforces best practices
- **debugger** - Specialized in diagnosing and fixing complex bugs in event-driven systems
- **performance-engineer** - Optimizes backtesting speed, memory usage, and execution latency
- **qa-expert** - Designs comprehensive testing strategies for trading systems
- **test-automator** - Builds robust test automation frameworks with high coverage

### Data & ML
- **data-engineer** - Expert in data pipelines, historical data storage, and feature engineering
- **data-scientist** - Specializes in quantitative analysis, signal discovery, and strategy validation
- **machine-learning-engineer** - Builds ML models for feature generation and signal processing
- **database-optimizer** - Optimizes database queries and storage for market data

### Infrastructure & Deployment
- **deployment-engineer** - Manages deployment from backtest to paper to live trading
- **devops-engineer** - Sets up CI/CD pipelines, monitoring, and infrastructure

### Developer Experience
- **dependency-manager** - Manages Python dependencies, virtual environments, and package versions
- **documentation-engineer** - Creates comprehensive documentation for strategies and APIs
- **refactoring-specialist** - Improves code structure while maintaining functionality

### Research & Analysis
- **data-researcher** - Explores market data, identifies patterns, and validates hypotheses
- **research-analyst** - Analyzes trading strategies, backtests, and performance metrics

## Usage

Agents are automatically available in Claude Code. Simply reference the agent's name when you need specialized expertise:

```
/agent python-pro
Can you review the strategy SDK implementation?
```

## Agent Selection Guide

### For Strategy Development
- Use **python-pro** for strategy implementation
- Use **data-scientist** for signal discovery and validation
- Use **research-analyst** for backtest analysis

### For Data Pipeline Work
- Use **data-engineer** for data ingestion and storage
- Use **data-researcher** for data exploration
- Use **database-optimizer** for performance tuning

### For Testing & Quality
- Use **test-automator** for building test frameworks
- Use **qa-expert** for test strategy
- Use **code-reviewer** before merging code
- Use **performance-engineer** for optimization

### For Infrastructure
- Use **devops-engineer** for CI/CD setup
- Use **deployment-engineer** for production deployment
- Use **dependency-manager** for package management

### For Debugging
- Use **debugger** for complex issues
- Use **performance-engineer** for performance problems

## Project Context

These agents understand the trading bot architecture:
- Event-driven design with single code path
- Strategy SDK isolation
- Simulation engine for backtest/paper/live modes
- Execution engine with realistic slippage
- Data platform with feature store
- Control plane for deployment and monitoring

## Adding New Agents

To add new agents, copy them from the claude-code-subagents repository:

```bash
cp /path/to/agent.md .claude/agents/
```

Update this README with the new agent's purpose and use cases.
