from setuptools import setup, find_packages

setup(
    name="gpu-compute-manager",
    version="0.1.0",
    description="Lightweight GPU device manager with fallback to CPU.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    py_modules=["compute_manager"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
