from setuptools import setup, find_packages

setup(
    name="state-space-model-universal",
    version="0.1.0",
    description="Universal State Space Model Implementation (S4, S5, Mamba)",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "einops>=0.6.0",
        "hydra-core>=1.3.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.910",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 