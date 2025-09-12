"""Minimal setup.py for ergodic_insurance package."""

import os
from pathlib import Path

from setuptools import find_packages, setup

# Read the version from _version.py
__version__ = ""
exec(open(os.path.join("ergodic_insurance", "_version.py")).read())

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ergodic_insurance",
    version=__version__,
    author="Alex Filiakov",
    author_email="alexfiliakov@gmail.com",
    description="Financial modeling for widget manufacturers with ergodic insurance limits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ergodic-insurance-limits",
    packages=find_packages(where="ergodic_insurance/src"),
    package_dir={"": "ergodic_insurance/src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy>=2.3.2",
        "pandas>=2.3.2",
        "pydantic>=2.11.7",
        "pyyaml>=6.0.2",
        "matplotlib>=3.10.5",
        "seaborn>=0.13.2",
        "scipy>=1.16.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.4.1",
            "pytest-cov>=6.2.1",
            "pytest-xdist>=3.8.0",
            "pylint>=3.3.8",
            "black>=25.1.0",
            "mypy>=1.17.1",
            "isort>=6.0.1",
            "types-PyYAML>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.1.1",
            "notebook>=7.4.5",
            "ipykernel>=6.30.1",
            "nbformat>=5.10.4",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
