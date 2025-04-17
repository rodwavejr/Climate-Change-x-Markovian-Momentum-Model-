# Multi-Commodity Climate Analysis

This project analyzes the relationship between climate data and prices for various agricultural commodities. It combines climate data from growing regions with global commodity price data to explore correlations and patterns.

## Commodities Covered

- Coffee
- Cocoa
- Maize (Corn)
- Wheat
- Soybeans
- Cotton
- Rice

## Key Components

### Data Files

- `combined_commodity_data.csv`: Global commodity price data (monthly)
- `coffee_climate_extended.csv`: Extended climate data for coffee regions (quarterly, 2015-2022)
- `*_climate_joined.csv`: Combined climate and price data for each commodity (created by scripts)

### Scripts

1. **commodity_regions.py**
   - Defines growing regions for each commodity
   - Maps commodity codes to friendly names
   - Contains geographic bounding boxes for data extraction

2. **combine_climate_commodity.py**
   - Simple script to combine coffee climate data with commodity prices
   - Uses existing quarterly climate data and expands to monthly
   - Adds climate signatures like temperature anomalies and drought index

3. **multi_commodity_climate.py**
   - Advanced script for processing multiple commodities
   - Can download climate data for different growing regions
   - Can simulate climate data when downloads aren't available
   - Combines climate data with commodity prices

### Notebooks

1. **Climate_Commodity_Analysis.ipynb**
   - Analyzes coffee climate and price relationships
   - Visualizes trends, correlations, and patterns
   - Focuses specifically on coffee growing regions

2. **Multi_Commodity_Analysis.ipynb**
   - Compares climate impacts across different commodities
   - Analyzes cross-commodity price correlations
   - Examines how climate in one region affects multiple commodities

## How to Use

### Setting Up the Environment

```bash
# Navigate to project directory
cd /Users/Apexr/Documents/Climate_Project

# Activate the virtual environment (contains compatible package versions)
source ./Machine_Learning/bin/activate
```

### Processing Data

1. **For Coffee Data Only:**
   ```bash
   python combine_climate_commodity.py
   ```
   This creates `climate_commodity_joined.csv` for coffee price analysis.

2. **For All Commodities (with simulation):**
   ```bash
   python multi_commodity_climate.py --simulate
   ```
   This creates separate files for each commodity using simulated climate data.

3. **For All Commodities (with data download - requires API key):**
   ```bash
   python multi_commodity_climate.py --download --start-year 2015 --end-year 2022
   ```
   Downloads climate data for all commodity regions from the Copernicus Climate Data Store.

### Running Analyses

1. **Analyzing Coffee Climate Data:**
   ```bash
   jupyter notebook Climate_Commodity_Analysis.ipynb
   ```

2. **Analyzing Multiple Commodities:**
   ```bash
   jupyter notebook Multi_Commodity_Analysis.ipynb
   ```

## Technical Notes

- The scripts use NumPy 1.24.3 for compatibility with NetCDF4 and GRIB libraries
- Climate data is processed using xarray and pandas
- Statistical analyses include:
  - Correlation analysis
  - Time-lagged correlations
  - Seasonal patterns
  - Climate threshold analysis
  - Drought impact assessment

## Climate Signatures

The following climate signatures are calculated for analysis:

- **Temperature anomalies**: Deviation from monthly average temperatures
- **Precipitation anomalies**: Deviation from monthly average precipitation
- **3-month rolling average temperature**: Smoothed temperature trends
- **3-month cumulative precipitation**: Cumulative water availability
- **Drought index**: Combination of precipitation and temperature anomalies
- **Heat stress index**: Combination of temperature and humidity

## Future Enhancements

1. Add more growing regions for each commodity
2. Incorporate more commodity types (sugar, natural gas, etc.)
3. Extend climate data time range to match commodity data (1990-2024)
4. Add predictive models for price forecasting
5. Include more sophisticated climate indicators (ENSO, NAO, etc.)