# Climate and Commodity Analysis Project

This project analyzes the relationship between climate data and commodity prices, with a focus on coffee. It processes ERA5-Land climate data for coffee-growing regions globally and combines it with commodity price data for analysis.

## Project Structure

- `coffee_climate_monthly.py`: Main script for downloading and processing monthly climate data for coffee regions
- `run_monthly_climate.sh`: Shell script to run the monthly climate processing in the correct environment
- `Climate_Commodity_Analysis.ipynb`: Jupyter notebook for analyzing the combined climate and commodity data
- `coffee_climate_extended.csv`: Extended climate data for coffee regions (2015-2022, quarterly)
- `combined_commodity_data.csv`: Commodity price data including coffee (monthly)
- `climate_commodity_joined.csv`: (Generated) Combined dataset with climate and price data

## Getting Started

### Prerequisites

This project requires Python 3.9+ with the following packages:
- pandas
- numpy (1.24.3 recommended for compatibility)
- xarray
- matplotlib
- cdsapi (CDS API Client for Copernicus data)
- cfgrib (for GRIB file processing)
- seaborn
- statsmodels

Use the provided Machine_Learning virtual environment for the correct package versions.

### Setup

1. Activate the environment:
   ```
   source ./Machine_Learning/bin/activate
   ```

2. Run the data processing:
   ```
   ./run_monthly_climate.sh
   ```
   
   Or to download new data (requires CDS API credentials):
   ```
   python coffee_climate_monthly.py --download --start-year 1990 --end-year 2024
   ```

3. Open the Jupyter notebook for analysis:
   ```
   jupyter notebook Climate_Commodity_Analysis.ipynb
   ```

## Coffee Growing Regions

The project includes data for multiple coffee-growing regions:
- Colombia
- Brazil (Minas Gerais)
- Vietnam
- Ethiopia
- Indonesia
- Costa Rica
- Guatemala
- Kenya
- India (Karnataka)
- Honduras

## Data Sources

- Climate data: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) - ERA5-Land monthly data
- Commodity price data: Various sources compiled in combined_commodity_data.csv

## Analysis Features

- Monthly climate data processing
- Climate signature calculation (temperature anomalies, drought index, etc.)
- Time series analysis of climate and price relationships
- Regional climate comparison
- Lag effect analysis
- Climate change trend analysis

## Existing Scripts

- `climate_data_loader.py`: Original script for quarterly climate data
- `run_extended_data.py`: Script for extended period (2015-2022) data processing
- `fix_climate_data.py`: Utility to fix incorrect file formats
- `run_fix.sh` and `run_extended.sh`: Shell scripts for running the utilities

## Troubleshooting

- **Package version conflicts**: If you see errors about incompatible packages, make sure you're using the "Python (Machine Learning)" kernel that was set up with compatible versions.
- **NetCDF file format errors**: If you encounter "Unknown file format" errors, the files might actually be ZIP archives. The scripts handle this automatically.
- **API access issues**: Make sure you've accepted the terms for the ERA5-Land dataset on the CDS website.
- **NumPy 2.x compatibility issues**: Use NumPy 1.24.3 for compatibility with NetCDF and SciPy libraries.

## License

This project is for educational/research purposes only.

## Acknowledgements

- ERA5-Land data provided by the Copernicus Climate Change Service
- Coffee region definitions from various agricultural sources