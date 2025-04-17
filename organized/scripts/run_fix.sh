#!/bin/bash
# Run the fix script with the correct environment

# Navigate to the project directory
cd ~/Documents/Climate_Project

# Activate the virtual environment
source ./Machine_Learning/bin/activate

# Make sure we're using the correct Python path
PYTHON_PATH=$(which python3)
echo "Python: $PYTHON_PATH"
echo "NumPy version: $(python3 -c 'import numpy; print(numpy.__version__)')"

# Run the fix script
python3 fix_climate_data.py