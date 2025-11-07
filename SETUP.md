# Quick Start Setup Guide

This guide will help you set up the Agentic AI Development project for local development.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/michalvalco/agentic_ai_development.git
cd agentic_ai_development
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (for testing)
pip install -r requirements-dev.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

**Required:**
- `ANTHROPIC_API_KEY` - Get from https://console.anthropic.com/settings/keys

**Optional:**
- `OPENAI_API_KEY` - Get from https://platform.openai.com/api-keys
- `LANGSMITH_API_KEY` - Get from https://smith.langchain.com/settings
- `PHOENIX_API_KEY` - Get from https://phoenix.arize.com

### 5. Verify Setup

```bash
# Test that imports work
python -c "from src.common import models, config, exceptions; print('✅ Setup successful!')"

# Run unit tests
pytest tests/unit -v

# Run a specific test
pytest tests/unit/test_prompt_routing.py -v
```

## Running Examples

```bash
# Set your API key if not in .env
export ANTHROPIC_API_KEY="your-key-here"

# Run the prompt routing example
python examples/01_prompt_routing.py
```

## Project Structure

```
agentic_ai_development/
├── src/
│   ├── common/              # Shared utilities, models, config
│   ├── prompt_routing/      # Intent classification & routing
│   ├── query_writing/       # SQL/API query generation (coming)
│   ├── data_processing/     # Data transformation (coming)
│   ├── tool_orchestration/  # Tool execution (coming)
│   └── decision_support/    # Decision analysis (coming)
├── tests/
│   ├── unit/               # Fast unit tests with mocks
│   └── integration/        # Slow integration tests with real APIs
├── examples/               # Example scripts
├── docs/                   # Documentation
├── pricing_config.yaml     # LLM pricing configuration
├── .env.example           # Example environment variables
└── README.md              # Project overview
```

## Running Tests

### Unit Tests (Fast, Mocked)

```bash
# Run all unit tests
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_prompt_routing.py -v

# Run tests matching pattern
pytest -k "test_classify" -v
```

### Integration Tests (Slow, Real API Calls)

```bash
# Run integration tests (requires valid API keys)
pytest tests/integration -v --no-cov

# Run only LLM integration tests
pytest -m llm_integration -v
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests with mocks
- `@pytest.mark.integration` - Integration tests with real services
- `@pytest.mark.llm_integration` - Tests that make real LLM API calls (cost money!)

## Configuration

The system loads configuration from multiple sources in order of precedence:

1. **Environment variables** (highest priority)
2. **`.env` file** (local development)
3. **Default values** (in `src/common/config.py`)

### Common Configuration Options

```bash
# In .env file:

# LLM Settings
DEFAULT_MODEL=claude-sonnet-4-20250514
FALLBACK_MODEL=gpt-4o-mini
MAX_TOKENS_PER_CALL=4096
TEMPERATURE=0.0

# Application Settings
ENVIRONMENT=development  # or staging, production
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
MAX_RETRIES=3
TIMEOUT_SECONDS=60.0

# Cost Tracking
ENABLE_COST_TRACKING=true
```

## Troubleshooting

### Import Errors

```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=/path/to/agentic_ai_development:$PYTHONPATH

# Or use pytest.ini configuration (already set up)
pytest tests/unit
```

### API Key Errors

```bash
# Verify .env file exists and has correct format
cat .env

# Check if API key is loaded
python -c "from src.common.config import settings; print(f'API Key: {settings.anthropic_api_key[:10]}...')"
```

### Test Failures

```bash
# Run tests with verbose output
pytest tests/unit -vv

# Run specific failing test
pytest tests/unit/test_prompt_routing.py::test_specific_test -vv

# Show local variables on failure
pytest tests/unit -vv --showlocals
```

## Cost Management

This project tracks LLM API costs automatically. Pricing is configured in `pricing_config.yaml`.

```bash
# View current pricing
cat pricing_config.yaml

# Update pricing when providers change rates
# Edit pricing_config.yaml and update last_updated date
```

## Development Workflow

1. **Create a branch** for your changes
2. **Write tests** for new features
3. **Run tests** before committing
4. **Update documentation** if needed
5. **Commit with clear messages**

```bash
# Create branch
git checkout -b feature/my-feature

# Make changes...
# Write tests...

# Run tests
pytest tests/unit -v

# Commit
git add .
git commit -m "Add my feature"

# Push
git push origin feature/my-feature
```

## Getting Help

- **Documentation:** See `docs/` directory
- **Architecture:** Read `docs/ARCHITECTURE.md`
- **Critical Review:** See `CRITICAL_REVIEW.md` for detailed analysis
- **Issues:** Check GitHub issues or create a new one

## Next Steps

1. ✅ Complete setup (you're here!)
2. Read `README.md` for project overview
3. Review `docs/ARCHITECTURE.md` for system design
4. Read `CRITICAL_REVIEW.md` for detailed code review
5. Try `examples/01_prompt_routing.py`
6. Explore the codebase in `src/`

---

**Note:** This project is under active development. See `ROADMAP.md` for planned features and implementation timeline.

**Last Updated:** 2025-11-07
