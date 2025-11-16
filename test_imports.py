"""
Test script to verify all required libraries are properly installed
for XAI Collaborative Filtering project
"""

print("Testing library imports...\n")

# Test Surprise (Collaborative Filtering)
try:
    from surprise import Dataset, Reader, SVD, KNNBasic
    from surprise.model_selection import cross_validate
    print("✓ Surprise library imported successfully")
except ImportError as e:
    print(f"✗ Error importing Surprise: {e}")

# Test Pandas
try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Error importing Pandas: {e}")

# Test NumPy
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Error importing NumPy: {e}")

# Test LIME
try:
    import lime
    import lime.lime_tabular
    print(f"✓ LIME imported successfully")
except ImportError as e:
    print(f"✗ Error importing LIME: {e}")

# Test SHAP
try:
    import shap
    print(f"✓ SHAP {shap.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Error importing SHAP: {e}")

# Test Matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Error importing Matplotlib: {e}")

# Test Seaborn
try:
    import seaborn as sns
    print(f"✓ Seaborn {sns.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Error importing Seaborn: {e}")

# Test Jupyter
try:
    import jupyter
    print(f"✓ Jupyter imported successfully")
except ImportError as e:
    print(f"✗ Error importing Jupyter: {e}")

print("\n✓ All libraries are properly installed and ready to use!")
