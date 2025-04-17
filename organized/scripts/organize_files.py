#!/usr/bin/env python3
"""
Organize Climate Project Files

This script organizes the files in the Climate Project directory into
a clean, structured format.

Usage:
    python organize_files.py
"""

import os
import shutil
import glob

def main():
    """Organize files into appropriate directories."""
    print("Organizing Climate Project files...")
    
    # Create directory structure if it doesn't exist
    dirs = {
        "data": os.path.join(os.getcwd(), "organized", "data_files"),
        "scripts": os.path.join(os.getcwd(), "organized", "scripts"),
        "notebooks": os.path.join(os.getcwd(), "organized", "notebooks"),
        "raw_data": os.path.join(os.getcwd(), "organized", "raw_data"),
        "docs": os.path.join(os.getcwd(), "organized", "docs"),
        "plots": os.path.join(os.getcwd(), "organized", "plots")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Move files to appropriate directories
    
    # 1. Move CSV data files to data_files directory
    csv_files = glob.glob("*.csv")
    for file in csv_files:
        # Skip temporary files
        if file.startswith(".") or file.startswith("~"):
            continue
        print(f"Moving {file} to data_files directory")
        # Copy instead of move to preserve originals
        shutil.copy2(file, os.path.join(dirs["data"], file))
    
    # 2. Move Python scripts to scripts directory
    py_files = glob.glob("*.py")
    for file in py_files:
        if file.startswith(".") or file.startswith("~"):
            continue
        print(f"Moving {file} to scripts directory")
        shutil.copy2(file, os.path.join(dirs["scripts"], file))
    
    # 3. Move notebooks to notebooks directory
    notebook_files = glob.glob("*.ipynb")
    for file in notebook_files:
        if file.startswith(".") or file.startswith("~"):
            continue
        print(f"Moving {file} to notebooks directory")
        shutil.copy2(file, os.path.join(dirs["notebooks"], file))
    
    # 4. Move shell scripts to scripts directory
    sh_files = glob.glob("*.sh")
    for file in sh_files:
        if file.startswith(".") or file.startswith("~"):
            continue
        print(f"Moving {file} to scripts directory")
        shutil.copy2(file, os.path.join(dirs["scripts"], file))
    
    # 5. Move documentation files to docs directory
    doc_files = glob.glob("*.md") + glob.glob("README*")
    for file in doc_files:
        if file.startswith(".") or file.startswith("~"):
            continue
        print(f"Moving {file} to docs directory")
        shutil.copy2(file, os.path.join(dirs["docs"], file))
    
    # 6. Move raw data files (NetCDF, GRIB, ZIP) to raw_data directory
    raw_files = glob.glob("*.nc") + glob.glob("*.grib") + glob.glob("*.zip")
    for file in raw_files:
        if file.startswith(".") or file.startswith("~"):
            continue
        print(f"Moving {file} to raw_data directory")
        shutil.copy2(file, os.path.join(dirs["raw_data"], file))
    
    # 7. Move plot files to plots directory
    plot_files = glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.pdf")
    for file in plot_files:
        if file.startswith(".") or file.startswith("~"):
            continue
        print(f"Moving {file} to plots directory")
        shutil.copy2(file, os.path.join(dirs["plots"], file))
    
    # 8. Move existing plots directory contents to organized/plots
    if os.path.exists("plots"):
        plot_files = glob.glob("plots/*.*")
        for file in plot_files:
            if os.path.isfile(file):
                dest_file = os.path.join(dirs["plots"], os.path.basename(file))
                print(f"Moving {file} to plots directory")
                shutil.copy2(file, dest_file)
    
    # 9. Move data directory contents to organized/raw_data
    if os.path.exists("data"):
        # Create subdirectories to match the structure
        for subdir in ["", "commodities", "extracted"]:
            if subdir:
                subdir_path = os.path.join("data", subdir)
                if os.path.exists(subdir_path):
                    target_path = os.path.join(dirs["raw_data"], subdir)
                    os.makedirs(target_path, exist_ok=True)
                    
                    # Copy files from this subdirectory
                    subdir_files = glob.glob(os.path.join(subdir_path, "*.*"))
                    for file in subdir_files:
                        if os.path.isfile(file):
                            dest_file = os.path.join(target_path, os.path.basename(file))
                            print(f"Moving {file} to {target_path}")
                            shutil.copy2(file, dest_file)
            else:
                # Handle files in the root of data directory
                data_files = glob.glob(os.path.join("data", "*.*"))
                for file in data_files:
                    if os.path.isfile(file):
                        dest_file = os.path.join(dirs["raw_data"], os.path.basename(file))
                        print(f"Moving {file} to raw_data directory")
                        shutil.copy2(file, dest_file)
    
    # 10. Create a new README in the main directory
    create_main_readme()
    
    print("\nFile organization complete!")
    print("Original files remain in place. Organized copies are in the 'organized' directory.")
    print("\nDirectory structure:")
    print("  organized/")
    print("    data_files/ - Contains all CSV data files")
    print("    scripts/ - Contains Python and shell scripts")
    print("    notebooks/ - Contains Jupyter notebooks")
    print("    raw_data/ - Raw climate and commodity data files")
    print("    docs/ - Documentation files")
    print("    plots/ - Plot images and visualization files")
    
    print("\nKey files:")
    print("  MASTER_climate_commodity_data.csv - Complete dataset with all commodity prices and climate variables")
    print("  price_data.csv - Just the commodity prices")
    print("  climate_data_long.csv - Climate data in long format")
    print("  Combined_Data_Analysis.ipynb - Notebook for analyzing the combined data")

def create_main_readme():
    """Create a new README file with information about the organized directory."""
    readme_content = """# Climate and Commodity Analysis Project

This project analyzes the relationship between climate data and commodity prices for multiple commodities including coffee, cocoa, wheat, maize, rice, soybeans, and cotton.

## Organized Directory Structure

The project files have been organized into the following directories:

- `organized/data_files/` - Contains all CSV data files with commodity prices and climate variables
- `organized/scripts/` - Contains Python and shell scripts for data processing
- `organized/notebooks/` - Contains Jupyter notebooks for analysis and visualization
- `organized/raw_data/` - Raw climate data (NetCDF, GRIB) and commodity price data
- `organized/docs/` - Documentation files and guides
- `organized/plots/` - Generated data visualizations and plots

## Key Files

- `MASTER_climate_commodity_data.csv` - Complete dataset with all commodity prices and climate variables
- `price_data.csv` - Just the commodity prices for all commodities
- `climate_data_long.csv` - Climate data in long format
- `Master_Dataset_Analysis.ipynb` - Comprehensive analysis of all commodities and climate variables
- `Combined_Data_Analysis.ipynb` - Focused analysis of specific relationships

## Available Commodities

The dataset includes the following commodities:
- Coffee (Colombia, Minas Gerais Brazil regions)
- Cocoa (Ghana, Ivory Coast regions)
- Wheat
- Maize
- Rice
- Soybeans
- Cotton

## How to Use

1. Activate the Machine_Learning environment:
   ```
   source ./Machine_Learning/bin/activate
   ```

2. Open the Master Dataset Analysis notebook:
   ```
   jupyter notebook organized/notebooks/Master_Dataset_Analysis.ipynb
   ```

3. For specific commodity analyses, open the corresponding commodity notebook:
   ```
   jupyter notebook organized/notebooks/Multi_Commodity_Analysis.ipynb
   ```

## Climate Variables

The dataset includes the following climate variables:
- Temperature
- Precipitation
- Temperature anomalies
- Drought indices
- Seasonal patterns

For detailed documentation, see the files in the `organized/docs/` directory.
"""
    
    with open("ORGANIZED_README.md", "w") as f:
        f.write(readme_content)
    
    print("Created new README file: ORGANIZED_README.md")

if __name__ == "__main__":
    main()