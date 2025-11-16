# XAI Collaborative Filtering

Explainable AI framework for collaborative filtering recommendations. Implements SVD with LIME/SHAP interpretability to generate human-readable explanations for movie predictions. Quantitative analysis of explanation quality, model performance, and user trust implications.

## Project Structure

```
xai-collaborative-filtering/
├── data/                      # Dataset directory
│   ├── raw/                   # Raw downloaded datasets
│   ├── processed/             # Cleaned and processed data
│   └── README.md              # Data documentation
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_loading_and_exploration.ipynb
│   └── README.md              # Notebook documentation
├── src/                       # Source code
│   ├── __init__.py
│   └── data_loader.py         # Data loading utilities
├── venv/                      # Virtual environment (not in git)
├── requirements.txt           # Python dependencies
├── test_imports.py           # Import verification script
├── QUICK_REFERENCE.md        # Quick command reference
└── README.md                 # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/SameenMubashar/xai-collaborative-filtering.git
cd xai-collaborative-filtering

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python test_imports.py
```

### 2. Load and Explore Data

```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter Notebook
cd notebooks
jupyter notebook

# Open: 01_data_loading_and_exploration.ipynb
# This will automatically download MovieLens 1M dataset
```

### 3. Start Developing

The notebook will guide you through:

- Downloading MovieLens 1M dataset (1M ratings, 6K users, 4K movies)
- Exploratory data analysis with visualizations
- Data preprocessing and cleaning
- Saving processed datasets

## Dataset

**MovieLens 1M Dataset:**

- 1 million ratings from 6,000 users on 4,000 movies
- User demographics (age, gender, occupation)
- Movie metadata (title, genres, year)
- Rating scale: 1-5 stars
- Automatically downloaded via notebook

See `data/README.md` for detailed dataset information.

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone the repository** (if not already done):

   ```bash
   git clone https://github.com/SameenMubashar/xai-collaborative-filtering.git
   cd xai-collaborative-filtering
   ```

2. **Create and activate virtual environment**:

   ```bash
   # Create virtual environment
   python3 -m venv venv

   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install required packages**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python test_imports.py
   ```
   You should see confirmation messages that all libraries are properly installed.

### Installed Libraries

- **scikit-surprise (1.1.4)**: Collaborative filtering algorithms (SVD, KNN, etc.)
- **pandas (2.1.4)**: Data manipulation and analysis
- **numpy (1.26.2)**: Numerical computing
- **lime (0.2.0.1)**: Local Interpretable Model-agnostic Explanations
- **shap (0.44.1)**: SHapley Additive exPlanations
- **matplotlib (3.8.2)**: Data visualization
- **seaborn (0.13.0)**: Statistical data visualization
- **jupyter (1.0.0)**: Interactive notebooks for development and analysis

### Development Environment

To start working with Jupyter notebooks:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start Jupyter Notebook
jupyter notebook

# Or start Jupyter Lab
jupyter lab
```

### Deactivating Virtual Environment

When you're done working on the project:

```bash
deactivate
```

## Project Workflow

### Phase 1: Data Loading ✓

- Download MovieLens 1M dataset
- Exploratory data analysis
- Data cleaning and preprocessing
- **Notebook:** `notebooks/01_data_loading_and_exploration.ipynb`

### Phase 2: Model Development (Coming Soon)

- Implement collaborative filtering algorithms (SVD, KNN)
- Train and validate models
- Evaluate performance metrics

### Phase 3: Explainability (Coming Soon)

- Integrate LIME for local explanations
- Integrate SHAP for feature importance
- Generate human-readable explanations

### Phase 4: Evaluation (Coming Soon)

- Compare explanation methods
- Analyze explanation quality
- User trust and satisfaction metrics

## Key Features

- 🎬 **MovieLens 1M Dataset**: Rich dataset with user demographics
- 📊 **Comprehensive EDA**: Statistical analysis and visualizations
- 🤖 **Collaborative Filtering**: SVD, Matrix Factorization, KNN
- 🔍 **Explainability**: LIME and SHAP integration
- 📓 **Interactive Notebooks**: Step-by-step implementation
- 📈 **Evaluation Metrics**: Model performance and explanation quality

## Technologies

- **Python 3.11+**
- **Surprise**: Collaborative filtering library
- **Pandas & NumPy**: Data manipulation
- **LIME & SHAP**: Explainable AI
- **Matplotlib & Seaborn**: Visualization
- **Jupyter**: Interactive development

## Documentation

- **README.md** (this file): Project overview and setup
- **data/README.md**: Dataset information and format
- **notebooks/README.md**: Notebook descriptions and usage
- **QUICK_REFERENCE.md**: Command reference and tips

## Contributing

This is a research project focused on explainable AI in recommendation systems. Contributions and suggestions are welcome!

## Citation

If you use this code or dataset, please cite:

```
F. Maxwell Harper and Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
https://doi.org/10.1145/2827872
```

## License

This project is for educational and research purposes. The MovieLens dataset is released under CC BY 4.0 license.

---

**Status:** 🚀 Phase 1 Complete - Data Loading and Exploration Ready!

For questions or issues, please open a GitHub issue.
