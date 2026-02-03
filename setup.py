"""Setup script for magisterka-detector package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="magisterka-detector",
    version="0.2.0",
    author="bartpom",
    description="AI/Deepfake Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bartpom/magisterka",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "Pillow>=9.0.0",
        "PyQt5>=5.15.0",  # dla GUI
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "magisterka-detect=magisterka_detector.cli:main",  # TODO: implement cli.py
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
