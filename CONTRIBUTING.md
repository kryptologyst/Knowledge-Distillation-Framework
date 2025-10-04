# Contributing to Knowledge Distillation Framework

Thank you for your interest in contributing to the Knowledge Distillation Framework! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of PyTorch and machine learning

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/knowledge-distillation-framework.git
   cd knowledge-distillation-framework
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install development dependencies
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 🛠️ Development Workflow

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep line length under 127 characters

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>90%)

```bash
# Run tests
pytest test_knowledge_distillation.py -v

# Run with coverage
pytest test_knowledge_distillation.py --cov=knowledge_distillation --cov-report=html
```

### Code Formatting
```bash
# Format code with black
black .

# Check formatting
black --check .

# Lint with flake8
flake8 .
```

## 📝 Types of Contributions

### 🐛 Bug Reports
- Use the GitHub issue template
- Provide detailed reproduction steps
- Include system information and error messages

### ✨ Feature Requests
- Describe the feature clearly
- Explain the use case and benefits
- Consider implementation complexity

### 🔧 Code Contributions
- Fork the repository
- Create a feature branch: `git checkout -b feature/amazing-feature`
- Make your changes
- Add tests for new functionality
- Ensure all tests pass
- Submit a pull request

## 📋 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, well-documented code
   - Add appropriate tests
   - Update documentation if needed

3. **Test Your Changes**
   ```bash
   pytest test_knowledge_distillation.py -v
   black --check .
   flake8 .
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Submit Pull Request**
   - Use the PR template
   - Provide clear description
   - Link related issues

## 🧪 Testing Guidelines

### Unit Tests
- Test individual functions and methods
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies

### Integration Tests
- Test complete workflows
- Verify end-to-end functionality
- Test with different configurations

### Example Test Structure
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def tearDown(self):
        """Clean up test fixtures"""
        pass
    
    def test_feature_works_correctly(self):
        """Test that feature works as expected"""
        # Arrange
        # Act
        # Assert
        pass
```

## 📚 Documentation

### Code Documentation
- Use Google-style docstrings
- Include parameter descriptions
- Provide usage examples
- Document return values

### API Documentation
- Keep README.md updated
- Document new features
- Provide usage examples
- Update configuration documentation

## 🎯 Areas for Contribution

### High Priority
- Additional model architectures
- Support for more datasets
- Advanced distillation techniques
- Performance optimizations

### Medium Priority
- Additional visualizations
- Export/import functionality
- Model comparison tools
- Hyperparameter optimization

### Low Priority
- Documentation improvements
- Code refactoring
- Test coverage improvements
- Performance benchmarks

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - PyTorch version
   - Operating system
   - Hardware (CPU/GPU)

2. **Reproduction Steps**
   - Clear, step-by-step instructions
   - Minimal code example
   - Expected vs actual behavior

3. **Error Information**
   - Full error traceback
   - Log files (if applicable)
   - Screenshots (if UI related)

### Feature Requests
When requesting features, please include:

1. **Use Case**
   - Why is this feature needed?
   - How would it be used?
   - What problem does it solve?

2. **Proposed Solution**
   - How should it work?
   - Any design considerations?
   - Alternative approaches considered?

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check README.md and code comments

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors list

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Thank You

Thank you for contributing to the Knowledge Distillation Framework! Your contributions help make this project better for everyone.
