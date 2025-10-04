"""
Setup script for Knowledge Distillation Framework
===============================================

This setup script allows easy installation and distribution of the
knowledge distillation framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="knowledge-distillation-framework",
    version="1.0.0",
    author="AI Projects",
    author_email="ai@example.com",
    description="A comprehensive framework for knowledge distillation with modern PyTorch practices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/knowledge-distillation-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "knowledge-distillation=knowledge_distillation:main",
            "kd-train=knowledge_distillation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    keywords=[
        "knowledge distillation",
        "machine learning",
        "deep learning",
        "model compression",
        "neural networks",
        "pytorch",
        "teacher-student",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/knowledge-distillation-framework/issues",
        "Source": "https://github.com/yourusername/knowledge-distillation-framework",
        "Documentation": "https://knowledge-distillation-framework.readthedocs.io/",
    },
)
