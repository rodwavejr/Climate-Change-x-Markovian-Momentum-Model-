#!/bin/bash
# Run monthly climate data processing in the ML environment

# Navigate to the project directory
cd ~/Documents/Climate_Project

# Activate the virtual environment
source ./Machine_Learning/bin/activate

# Make sure we're using the correct Python path
PYTHON_PATH=$(which python3)
echo "Python: $PYTHON_PATH"
echo "NumPy version: $(python3 -c 'import numpy; print(numpy.__version__)')"

# Run with default settings (no download by default)
python3 coffee_climate_monthly.py

# To download data, comment the above line and uncomment this:
# python3 coffee_climate_monthly.py --download --start-year 1990 --end-year 2024

# For specific regions only, use:
# python3 coffee_climate_monthly.py --download --regions COLOMBIA "MINAS GERAIS (BRA)" VIETNAM ETHIOPIA

echo "Processing complete!"