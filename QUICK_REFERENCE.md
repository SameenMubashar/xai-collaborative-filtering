# Quick Reference Guide

## Virtual Environment Commands

### Activate Virtual Environment

```bash
source venv/bin/activate
```

### Deactivate Virtual Environment

```bash
deactivate
```

### Install New Package

```bash
# Make sure venv is activated first
pip install package_name

# Add to requirements.txt
pip freeze > requirements.txt
```

## Project Structure

```
xai-collaborative-filtering/
├── venv/                   # Virtual environment (not in git)
├── requirements.txt        # Python dependencies
├── test_imports.py        # Test script for verifying installation
├── README.md              # Project documentation
└── .gitignore             # Git ignore rules
```

## Key Libraries Usage

### Surprise (Collaborative Filtering)

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Load data
data = Dataset.load_builtin('ml-100k')

# Train model
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5)
```

### LIME (Explainability)

```python
import lime
import lime.lime_tabular

# Create explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data,
    feature_names=feature_names,
    mode='regression'
)

# Explain prediction
explanation = explainer.explain_instance(instance, model.predict)
```

### SHAP (Explainability)

```python
import shap

# Create explainer
explainer = shap.Explainer(model)

# Calculate SHAP values
shap_values = explainer(X)

# Visualize
shap.plots.waterfall(shap_values[0])
```

### Pandas & NumPy

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Basic operations
array = np.array([1, 2, 3, 4, 5])
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create plots
plt.figure(figsize=(10, 6))
sns.heatmap(data, annot=True)
plt.show()
```

## Common Tasks

### Start Jupyter Notebook

```bash
source venv/bin/activate
jupyter notebook
```

### Start Jupyter Lab

```bash
source venv/bin/activate
jupyter lab
```

### Run Python Script

```bash
source venv/bin/activate
python script_name.py
```

### Update Dependencies

```bash
source venv/bin/activate
pip install --upgrade package_name
pip freeze > requirements.txt
```

## Troubleshooting

### Package Not Found

```bash
# Activate virtual environment first
source venv/bin/activate

# Then install the package
pip install package_name
```

### Virtual Environment Issues

```bash
# Remove and recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Jupyter Kernel Issues

```bash
# Install ipykernel in virtual environment
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=xai-cf --display-name "Python (XAI-CF)"
```

## Next Steps

1. Create data preprocessing scripts
2. Implement collaborative filtering models
3. Add explainability layer with LIME/SHAP
4. Build evaluation metrics
5. Create visualization dashboards
6. Document findings and results
