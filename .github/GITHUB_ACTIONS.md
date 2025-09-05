# GitHub Actions Workflows for Matrix0

This directory contains GitHub Actions workflows that automatically test and validate Matrix0 on every code change.

> **Note**: This is specific to GitHub Actions CI/CD. For the main project README, see the root directory.

## Workflows

### 1. Code Quality (`code-quality.yml`)
**Triggers**: Every push and pull request
**Purpose**: Ensure code quality and prevent basic issues
**Tests**:
- Linting with flake8
- Code formatting with black
- Type checking with mypy
- **Security scanning with bandit**
- Configuration validation
- Package import testing

### 2. Model Validation (`model-validation.yml`)
**Triggers**: Changes to model code or checkpoints
**Purpose**: Validate model architecture, encoding, and advanced features
**Tests**:
- Model creation and loading
- Forward pass validation
- Encoding function testing
- Configuration compatibility
- **SSL algorithm testing** (threat detection, square control, piece recognition)
- **Tournament system validation** (Glicko-2 ratings, tournament config)
- **Enhanced security scanning** (hardcoded secrets, unsafe patterns)

### 3. Training Pipeline Test (`training-pipeline-test.yml`)
**Triggers**: Changes to training code
**Purpose**: Catch training issues before they waste time
**Tests**:
- Data loading and management
- Training script functionality
- Configuration validation
- Minimal training step validation

## Benefits

- **Immediate Feedback**: Catch issues in minutes, not hours
- **Quality Assurance**: Ensure code meets standards
- **Training Protection**: Prevent failed training runs
- **Professional Standards**: Industry-standard CI/CD practices

## Usage

### Automatic Testing
- Workflows run automatically on every push
- No manual intervention required
- Results visible in GitHub Actions tab

### Manual Testing
- Use "workflow_dispatch" to manually trigger tests
- Useful for testing specific scenarios
- Available in GitHub Actions tab

### Viewing Results
1. Go to your repository on GitHub
2. Click "Actions" tab
3. View workflow runs and results
4. Check logs for detailed information

## Local Testing

You can test these workflows locally before pushing:

```bash
# Install testing tools
pip install flake8 black mypy

# Run code quality checks
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
black --check --diff .
mypy azchess/ --ignore-missing-imports

# Test configuration
python -c "from azchess.config import Config; Config.load('config.yaml')"

# Test imports
python -c "import azchess; print('Package imports successful')"
```

## Customization

### Adding New Tests
- Create new workflow files in `.github/workflows/`
- Follow the existing pattern
- Use appropriate triggers and conditions

### Modifying Existing Workflows
- Edit the YAML files directly
- Test locally before pushing
- Use GitHub's workflow editor for validation

### Performance Optimization
- Use path filters to trigger only relevant workflows
- Cache dependencies when possible
- Use matrix builds for multiple Python versions

## Troubleshooting

### Common Issues
1. **Dependency failures**: Check requirements.txt
2. **Import errors**: Verify package structure
3. **Configuration issues**: Validate config.yaml
4. **Timeout errors**: Optimize test execution time

### Getting Help
- Check workflow logs for detailed error messages
- Use GitHub's workflow editor for syntax validation
- Test workflows locally before pushing
- Review GitHub Actions documentation
