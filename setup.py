from setuptools import setup, find_packages

setup(
    name="state_space_models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "matplotlib>=3.5.0",
        "wandb>=0.15.0",  # for logging
        "hydra-core>=1.3.0",  # for configuration
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Implementation of state-of-the-art state space models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 