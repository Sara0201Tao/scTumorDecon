# scTumorDecon

![License](https://img.shields.io/badge/License-MIT-black)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)

## Overview

**scTumorDecon** is a biology-informed two-stage deep learning framework for subtype-level deconvolution of bulk RNA-seq data using scRNA-seq references. The framework integrates single-cell reference atlases, pseudo-bulk simulation, and biologically constrained deep neural networks to estimate both major cellular compartments and fine-grained cellular subtypes from bulk transcriptomic samples.

The workflow is specifically designed for complex clinical samples, where low-abundance and highly correlated cell populations are difficult to resolve using conventional deconvolution approaches.

<p align="center">

<img src="images/1.png" width="900">

</p>

Single-cell RNA-seq references are first used to generate pseudo-bulk samples with known cellular compositions. A two-stage hierarchical deep learning framework is then trained to predict major cell classes followed by cellular subtypes. The trained models can subsequently be applied to independent bulk RNA-seq cohorts for subtype-level deconvolution and downstream biological analyses.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sara0201Tao/scTumorDecon.git
cd scTumorDecon
```

Install required packages:

```bash
pip install -r requirements.txt
```

Alternatively, install the package locally:

```bash
pip install -e .
```

---

## Repository Structure

```text
scTumorDecon/
├── images/
│
├── scTumorDecon/
│   ├── __init__.py
│   ├── simulation.py
│   ├── utils.py
│   ├── model.py
│   ├── main.py
│   └── nestedCV_evaluation.py
│
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Core modules

| File                   | Description                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| simulation.py          | Pseudo-bulk sample generation from scRNA-seq references                                            |
| utils.py               | Data processing and utility functions                                                              |
| model.py               | PyTorch neural network architectures and training procedures                                       |
| main.py                | Training and prediction workflow                                                                   |
| nestedCV_evaluation.py | Nested cross-validation, model evaluation, permutation testing, and confusion matrix visualization |

---

## Usage

### Step 1: Generate pseudo-bulk training data

Generate simulated bulk RNA-seq samples from annotated scRNA-seq references:

```bash
python -m scTumorDecon.simulation \
    --input xxx.h5ad \
    --output simulated_data/
```

---

### Step 2: Train scTumorDecon models

Train the hierarchical deep learning models:

```bash
python -m scTumorDecon.main \
    --train simulated_data/ \
    --output trained_models/
```

---

### Step 3: Predict cellular compositions

Apply trained models to bulk RNA-seq datasets:

```bash
python -m scTumorDecon.main \
    --predict bulk_expression.csv \
    --model_dir trained_models/
```

---

### Step 4: Evaluate classification models

Evaluate downstream predictive models using repeated nested cross-validation:

```python
from scTumorDecon.nestedCV_evaluation import run_svm_analysis

run_svm_analysis(...)
```

This module provides:

* Repeated nested cross-validation
* Hyperparameter optimization
* Permutation testing
* Parameter stability analysis
* Confusion matrix visualization


## Citation

If you use scTumorDecon in your research, please cite:

```text
xxxx
```
