from setuptools import setup, find_packages

setup(
    author="Samuel JS Wright",
    description="Functions to do data driven sampling and plotting",
    name="data_driven_sampler",
    version="0.1.0",
    packages=find_packages(include=["data_driven_sampler", "data_driven_sampler.*"]),
    install_requires=[
        "pandas >= 1.5",
        "numpy >= 1.22",
        "scikit-learn >= 1.2",
    ],
    python_requires=">=3.9",
)
