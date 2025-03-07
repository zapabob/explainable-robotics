import os
from setuptools import setup, find_packages

# READMEファイルを読み込まずに直接説明文を指定
long_description = """# Explainable Robotics

神経科学的に妥当なヒューマノイドロボット制御のための説明可能AIフレームワーク

## 概要

Explainable Roboticsは、ヒューマノイドロボットの制御において、AIの意思決定プロセスを説明可能にするためのフレームワークです。
神経科学の知見を活かし、人間の認知プロセスに近い形でロボットの動作を制御・理解することができます。

## インストール

```bash
pip install explainable-robotics
```
"""

setup(
    name="explainable-robotics",
    version="0.1.0",
    author="Ryo Minegishi",
    author_email="info@explainable-robotics.com", 
    description="数理的に妥当なヒューマノイドロボット制御のための説明可能AIフレームワーク",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zapabob/explainable_robotics",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Robotics",
    ],
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0", 
        "torch>=2.0.0",
        "langchain>=0.0.267",
        "langchain-openai>=0.0.2",
        "langchain-anthropic>=0.0.2",
        "langchain-google-genai>=0.0.3",
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "google-generativeai>=0.2.0",
        "faiss-cpu>=1.7.0",
        "pillow>=10.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "huggingface-hub>=0.16.0",
        "pydantic>=2.0.0",
        "requests>=2.30.0",
        "tqdm>=4.65.0",
        "typing-extensions>=4.5.0",
        "genesis-world>=0.1.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0", 
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "faiss-gpu>=1.7.0",
        ],
        "all": [
            "ctransformers>=0.2.0",
            "llama-cpp-python>=0.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "explainable-robotics=explainable_robotics.main:main",
        ],
    }
)