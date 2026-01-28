"""
Setup script for veFIL Tokenomics Workbench.

This allows the package to be installed via pip, which is required for
Streamlit Cloud deployment and clean imports.

Usage:
    pip install -e .  # Development install (editable)
    pip install .     # Production install
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="vefil-workbench",
    version="2.1.0",
    description="Rigorous cryptoeconomic modeling tool for veFIL token locking mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="veFIL Workbench Contributors",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "vefil": ["config/defaults.yaml"],
    },
    include_package_data=True,
    install_requires=[
        "streamlit>=1.28.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "plotly>=5.17.0",
        "pyyaml>=6.0",
        "scipy>=1.11.0",
        "openai>=1.40.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vefil-workbench=vefil.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
