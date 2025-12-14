"""
RESU: Resurrection of Sparse Units
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="resu",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Resurrection of Sparse Units: A novel sparse neural network training method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/resu",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
        ],
        "examples": [
            "transformers>=4.30.0",
            "datasets>=2.10.0",
            "wandb>=0.15.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
            "transformers>=4.30.0",
            "datasets>=2.10.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resu-verify=scripts.verify_resu:main",
        ],
    },
)
