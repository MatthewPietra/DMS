# DMS - Detection Model Suite

[![Tests](https://github.com/your-org/dms-detection-suite/workflows/Tests/badge.svg)](https://github.com/your-org/dms-detection-suite/actions)
[![Coverage](https://codecov.io/gh/your-org/dms-detection-suite/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/dms-detection-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive object detection pipeline with integrated authentication, annotation, training, and deployment capabilities.

## 🎯 Coverage Strategy

This project uses a **realistic coverage approach**:

- **Target**: 70% coverage for core business logic
- **Excluded**: GUI components, platform-specific code, external APIs
- **Focus**: Integration tests and critical user workflows

### Coverage Breakdown:
- ✅ **Core Business Logic**: 70%+ (project management, basic operations)
- 🟡 **Configuration**: 60%+ (settings, validation)
- 🟡 **Hardware Detection**: 60%+ (device detection, optimization)
- 🟡 **Metrics & Export**: 70%+ (calculation, validation)
- 🔴 **GUI Components**: 30% (manual testing preferred)
- 🔴 **Platform-Specific**: 40% (error handling, edge cases)

## 📁 Project Structure

```
DMS/
├── src/                # Main source code (modules, packages)
├── scripts/            # Runners, launchers, and utility scripts
├── tests/              # All test code
├── data/               # All datasets and generated data (including COCO/)
├── config/             # Configuration files
├── requirements/       # All requirements files
├── docs/               # Documentation
├── .github/            # GitHub workflows and templates
├── .venv/              # (gitignored) Python virtual environment
├── venv/               # (gitignored) Python virtual environment
├── dms_venv310/        # (gitignored) Python virtual environment
├── README.md           # Project overview
├── pyproject.toml      # Build system and dependencies
├── setup.py            # Legacy build script
├── .gitignore          # Git ignore rules
├── ...                # Other project-level files
```

- All scripts and runners are in `scripts/`
- All test files are in `tests/`
- All requirements are in `requirements/`
- All config files are in `config/`
- All documentation is in `docs/`
- All data (including COCO) is in `data/`
- All virtual environments are ignored by git

## Quick Start

```bash
# Install
pip install dms-detection-suite[all]

# Run GUI
dms-studio

# Command line
dms project create my-project
dms capture --duration 60
dms train --data ./dataset
```

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [API Reference](docs/API.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure coverage stays above 70%
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details. 
