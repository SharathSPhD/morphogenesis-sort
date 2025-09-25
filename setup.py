#!/usr/bin/env python3
"""Setup script for enhanced morphogenesis implementation."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open('docs/README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Enhanced Morphogenesis Research Platform"

# Read requirements
def read_requirements(filename):
    with open(os.path.join('requirements', filename), 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-r')]

setup(
    name="enhanced-morphogenesis",
    version="1.0.0",
    description="Enhanced Morphogenesis Research Platform with Async Architecture",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Morphogenesis Research Team",
    author_email="research@morphogenesis.org",
    url="https://github.com/SharathSPhD/morphogenesis-sort",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements('base.txt'),
    extras_require={
        'dev': read_requirements('development.txt'),
        'test': read_requirements('testing.txt'),
    },
    entry_points={
        'console_scripts': [
            'morphogenesis-run=scripts.run_experiments:main',
            'morphogenesis-analyze=scripts.analyze_results:main',
            'morphogenesis-validate=scripts.validate_installation:main',
        ],
    },
    package_data={
        'enhanced_morphogenesis': [
            'config/*.yaml',
            'experiments/configs/templates/*.json',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)