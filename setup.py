#!/usr/bin/env python3
"""
Setup script for DMS (Detection Model Suite)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements" / "requirements_base.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="dms-detection-suite",
    version="1.0.0",
    description="A comprehensive object detection pipeline with integrated authentication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DMS Team",
    author_email="team@dms-detection.com",
    url="https://github.com/your-org/dms-detection-suite",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
        ],
        "gpu": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "ultralytics>=8.0.0",
        ],
        "gui": [
            "PyQt5>=5.15.0",
            "PySide6>=6.0.0",
        ],
        "auth": [
            "cryptography>=41.0.0",
            "requests>=2.31.0",
            "psutil>=5.9.0",
            "defusedxml>=0.7.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "dms=dms.cli:main",
            "dms-studio=dms.studio:main",
            "dms-capture=dms.capture:main",
            "dms-train=dms.training:main",
            "dms-annotate=dms.annotation:main",
        ],
    },
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
    keywords="object-detection, computer-vision, machine-learning, annotation, training",
    project_urls={
        "Bug Reports": "https://github.com/your-org/dms-detection-suite/issues",
        "Source": "https://github.com/your-org/dms-detection-suite",
        "Documentation": "https://dms-detection-suite.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
) 