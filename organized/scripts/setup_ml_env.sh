#!/bin/bash
# Script to set up Machine Learning virtual environment for climate data processing

# Activate the virtual environment
source ~/Documents/Climate_Project/Machine_Learning/bin/activate

echo "Installing packages with compatible versions..."

# First install base packages
pip install --upgrade pip setuptools wheel
pip install ipykernel jupyter

# Install numpy 1.24.3 (compatible with older libraries)
pip install numpy==1.24.3

# Install other required packages
pip install pandas==2.0.3 matplotlib==3.7.2 scipy==1.11.3

# Install xarray and netCDF4 (dependencies for ERA5 data)
pip install xarray==2023.1.0 netCDF4==1.6.4

# Install CDS API client
pip install cdsapi==0.6.1

# Register the kernel with Jupyter
python -m ipykernel install --user --name=Machine_Learning --display-name="Python (Machine Learning)"

echo "Setup complete! Now you can start Jupyter and select the 'Python (Machine Learning)' kernel."
echo "Run: jupyter notebook"