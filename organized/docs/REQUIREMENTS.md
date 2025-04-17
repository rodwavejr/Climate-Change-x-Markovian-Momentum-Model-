# Climate Project Requirements

This document lists the required packages to run the Climate Project analysis notebooks.

## Core Requirements

These packages are essential for basic data loading and visualization:

```bash
# Basic data processing and visualization
pip install pandas numpy matplotlib seaborn
```

## Machine Learning

These packages are required for machine learning models:

```bash
# Machine learning
pip install scikit-learn
```

## Time Series Analysis

These packages are needed for time series decomposition and ARIMA modeling:

```bash
# Time series analysis
pip install statsmodels pmdarima
```

## Deep Learning

These packages are optional and only needed for LSTM models:

```bash
# Deep learning (optional)
pip install tensorflow
```

## All-in-One Installation

To install all required packages at once:

```bash
# Install all requirements
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima tensorflow
```

## Installation in Virtual Environment

It's recommended to install these packages in a virtual environment:

```bash
# Create a virtual environment
python -m venv climate_env

# Activate the environment (Linux/Mac)
source climate_env/bin/activate

# Activate the environment (Windows)
climate_env\Scripts\activate

# Install packages
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima tensorflow
```

## Troubleshooting

If you encounter issues with package installation:

1. Make sure you have Python 3.7+ installed
2. Update pip: `pip install --upgrade pip`
3. Some packages (like TensorFlow) may have additional system requirements
4. For Apple Silicon Macs, some packages may require special installation steps