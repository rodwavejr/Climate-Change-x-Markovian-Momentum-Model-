#!/bin/bash
# Setup script for Climate Project Analysis Environment

echo "Setting up Climate Project Analysis Environment..."

# Activate the Machine Learning environment if it exists
if [ -d "Machine_Learning" ]; then
    source Machine_Learning/bin/activate
    echo "Activated Machine Learning environment"
else
    echo "Machine_Learning directory not found. Using system Python."
fi

# Install required packages
echo "Installing required packages..."

# Core packages (required)
pip install pandas numpy matplotlib seaborn

# Machine learning packages (recommended)
pip install scikit-learn

# Time series analysis packages (optional)
pip install statsmodels

# Launch Jupyter Notebook if installed
if command -v jupyter &> /dev/null; then
    echo "Starting Jupyter Notebook..."
    jupyter notebook organized/notebooks/Master_Data_Analysis_Template.ipynb
else
    echo "Jupyter Notebook not found. Please install with:"
    echo "pip install jupyter"
fi

echo "Setup complete!"