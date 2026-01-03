# slm-pricer

Experimental Python project for fine-tuning a small language model for pricing tasks, based on Ed Donner's LLM Engineering course.

## Project Structure

```
slm-pricer/
├── src/slm_pricer/              # Core package (importable)
│   ├── models.py                # Neural network architectures
│   ├── training.py              # Training and evaluation utilities
│   ├── data.py                  # Dataset classes and data loading
│   └── utils.py                 # Helper functions (price transforms, etc.)
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_training_llama_3_1.ipynb
│   ├── 02_finetuned_llama_3_1_analysis.ipynb
│   ├── ...
│
├── scripts/                     # Standalone utility scripts
│   ├── visualize_models.py
│   ├── model_data_loader.py
│   └── create_ensemble.py
│
├── results/                    # Output artifacts (gitignored)
│
├── pyproject.toml              # Package configuration
├── CLAUDE.md                   # Instructions for Claude Code
├── LICENSE
└── README.md
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install dependencies
uv sync

# Install package in development mode
uv pip install -e .
```

