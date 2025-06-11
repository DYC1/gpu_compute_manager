from setuptools import setup, find_packages

setup(
    name="gpu-compute-manager",
    version="0.1.0",
    description="Lightweight GPU device manager with fallback to CPU.",
    author="DYC",
    author_email="dyc1go@outlook.com",
    packages=find_packages(),
    py_modules=["compute_manager"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
