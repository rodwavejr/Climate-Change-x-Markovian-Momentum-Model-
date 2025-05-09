#!/bin/bash
# WARNING: This script will remove the original files.
# Only run this after verifying that all files have been properly archived.

echo "This will remove the original files from the project directory."
echo "The files will still be available in both the 'organized' directory and the 'archive' directory."
echo "Press Ctrl+C to cancel or Enter to continue..."
read

rm "wheat_climate_joined.csv"
rm "Project Data Loader.ipynb"
rm "all_commodity_prices.csv"
rm "MINAS_GERAIS_(BRA)_era5land_data.nc"
rm "combined_commodity_data.csv"
rm "cocoa_climate_joined.csv"
rm "MINAS GERAIS (BRA)_era5land_data.zip"
rm "combine_all_commodities_improved.py"
rm "cleanup_original_files.py"
rm "rice_climate_joined.csv"
rm "climate_data_long_form.csv"
rm "coffee_climate_extended.csv"
rm "run_extended_data.py"
rm "price_data.csv"
rm "multi_commodity_climate.py"
rm "Multi_Commodity_Analysis.ipynb"
rm "all_commodities_combined.csv"
rm "MASTER_climate_commodity_data.csv"
rm "coffee_climate_monthly.py"
rm "climate_data_long.csv"
rm "commodity_regions.py"
rm "combine_all_commodities.py"
rm "run_monthly_climate.sh"
rm "README.md"
rm "run_fix.sh"
rm "Climate_Commodity_Analysis.ipynb"
rm "climate_commodity_joined.csv"
rm "COLOMBIA_era5land_data.zip"
rm "climate_data_simple.py"
rm "coffee_climate_data_backup.csv"
rm "Combined_Data_Analysis.ipynb"
rm "coffee_climate_extended_backup.csv"
rm "create_master_dataset.py"
rm "all_data_wide.csv"
rm "combine_climate_commodity.py"
rm "soybeans_climate_joined.csv"
rm "fix_climate_data.py"
rm "coffee_climate_joined.csv"
rm -rf ".ipynb_checkpoints"
rm "MULTI_COMMODITY_README.md"
rm "maize_climate_joined.csv"
rm "coffee_climate_data.csv"
rm "complete_commodity_climate.csv"
rm "test_climate_data.csv"
rm "setup_ml_env.sh"
rm -rf "data"
rm "run_extended.sh"
rm "climate_data_loader.py"
rm "cotton_climate_joined.csv"
rm "organize_files.py"
rm "COLOMBIA_era5land_data.nc"
rm -rf "plots"
rm "ORGANIZED_README.md"

echo "Original files have been removed."
echo "Your project is now using the organized directory structure."
