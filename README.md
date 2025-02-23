\# MNIST Deep Learning Project

This repository demonstrates **best practices** for a deep learning project using the **MNIST** dataset. We showcase a modular structure for loading data, building and training a CNN, evaluating performance, writing tests, and optionally containerizing and integrating CI/CD pipelines.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Directory Structure](#directory-structure)  
3. [Prerequisites](#prerequisites)  
4. [Setting Up a Virtual Environment](#setting-up-a-virtual-environment)  
5. [Installing Dependencies](#installing-dependencies)  
6. [Jupyter Notebook Kernel](#jupyter-notebook-kernel)  
7. [Usage](#usage)  
   - [Training](#training)  
   - [Evaluation](#evaluation)  
   - [Notebook Tutorial](#notebook-tutorial)  
8. [Testing](#testing)  
9. [Pushing to GitHub](#pushing-to-github)  
10. [Docker (Optional)](#docker-optional)  
11. [Continuous Integration (Optional)](#continuous-integration-optional)  
12. [Contributing](#contributing)  
13. [License](#license)  

---

## Project Overview

We will use the **MNIST** dataset (28×28 grayscale images of handwritten digits) to illustrate how to structure a DL project with:

- **Data loading** in a dedicated file (`data_loader.py`).  
- **Model definition** in `model.py`.  
- **Training script** (`train.py`) that logs metrics with MLflow.  
- **Evaluation script** (`evaluate.py`) for testing the final model on unseen data.  
- **Tests** in the `tests/` folder to ensure each component works properly.  
- (Optional) **Dockerfile** for containerization and a **CI/CD** workflow in `.github/workflows/ci.yml`.  

---

## Directory Structure

```
project_root/
├── data/
│   └── README.md  # (Instructions on data management)
├── notebooks/
│   └── tutorial.ipynb  # Jupyter Notebook for an interactive tutorial
├── src/
│   ├── __init__.py  # (Empty: marks src as a package)
│   ├── config.yaml  # Configuration file for parameters
│   ├── data_loader.py  # Module to load and preprocess MNIST data
│   ├── model.py  # Module defining the CNN architecture
│   ├── train.py  # Training script (includes MLflow auto-logging)
│   ├── evaluate.py  # Script for model evaluation and plotting
│   └── utils.py  # Helper functions like setting random seeds
├── tests/
│   ├── __init__.py  # (Empty: marks tests as a package)
│   ├── test_data_loader.py  # Unit tests for the data loader
│   ├── test_model.py  # Unit tests for the model
│   ├── test_train.py  # Tests for the training loop
├── Dockerfile  # Docker configuration for containerizing the app
├── requirements.txt  # List of Python dependencies
├── .github/
│   └── workflows/
│       └── ci.yml  # GitHub Actions workflow file for CI/CD
└── README.md  # Project overview and instructions
```

---

## Prerequisites

- **Python 3.8+**  
- **pip** or another package manager (e.g., conda)  
- **Git** (for version control and pushing to GitHub)  
- (Optional) **Docker** if you want to containerize the project  
- (Optional) **GitHub Actions** or another CI service if you plan on using CI/CD  

---

## Setting Up a Virtual Environment

Using a virtual environment keeps dependencies isolated from your system Python.

```sh
cd /path/to/project_root
python3 -m venv venv
```

### Activate the Virtual Environment:

**On macOS/Linux:**
```sh
source venv/bin/activate
```

**On Windows:**
```sh
venv\Scripts\activate
```

---

## Installing Dependencies
After activating the virtual environment, install the necessary packages:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```
This ensures you have all libraries (like tensorflow, mlflow, pytest) needed to run and develop this project.

---

## Jupyter Notebook Kernel
To run Jupyter notebooks in this virtual environment:
```sh
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name mnist-env --display-name "Python (MNIST Env)"
```
Select “Python (MNIST Env)” kernel in Jupyter Notebook.

---

## Usage

### Training
```sh
python src/train.py
```
### Evaluation
```sh
python src/evaluate.py
```
### Notebook Tutorial
```sh
jupyter notebook notebooks/tutorial.ipynb
```

---

## Testing
```sh
pytest
```

---

## Pushing to GitHub (Create Repo First)
```sh
git init
echo "venv/" >> .gitignore
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/mnist-project.git
git push -u origin main
```

---

## Docker (Optional)
```sh
docker build -t mnist-app .
docker run -it mnist-app
```

---

## Continuous Integration (Optional)
GitHub Actions workflow in `.github/workflows/ci.yml` automates:
- Installing dependencies
- Running tests
- Checking code quality

---

## Contributing
1. Fork this repository.
2. Create a feature branch.
3. Write/update tests.
4. Open a pull request.

---

## License
Open to all UB students taking CSE 676B

