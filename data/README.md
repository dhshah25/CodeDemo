# DL CIFAR-10 Tutorial Project

## Overview
This project demonstrates best practices in software design principles for deep learning projects using the CIFAR-10 dataset. It covers data loading, model definition, training, evaluation, testing, containerization, and CI/CD integration.

## Directory Structure

project_root/
├── data/
│   └── README.md             # (Optional: instructions on data management)
├── notebooks/
│   └── tutorial.ipynb        # Jupyter Notebook for the interactive tutorial
├── src/
│   ├── __init__.py           # (Empty: marks src as a package)
│   ├── config.yaml           # Configuration file for parameters
│   ├── data_loader.py        # Module to load and preprocess CIFAR-10 data
│   ├── model.py              # Module defining the CNN architecture
│   ├── train.py              # Training script (includes MLflow auto-logging)
│   ├── evaluate.py           # Script for model evaluation and plotting
│   └── utils.py              # Helper functions (e.g., setting random seeds)
├── tests/
│   ├── __init__.py           # (Empty: marks tests as a package)
│   ├── test_data_loader.py   # Unit tests for the data loader module
│   ├── test_model.py         # Unit tests for the model module
│   └── test_train.py         # Tests for the training loop
├── Dockerfile                # Docker configuration for containerizing the app
├── requirements.txt          # List of Python dependencies
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions workflow file for CI/CD
└── README.md                 # Project overview and instructions



## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

2. 
    python src/train.py

3. 
    python src/evaluate.py

