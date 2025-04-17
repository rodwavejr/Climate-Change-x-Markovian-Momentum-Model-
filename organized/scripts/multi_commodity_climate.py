#!/usr/bin/env python3
"""
Multi-Commodity Climate Data Processor

This script:
1. Downloads climate data for multiple commodity growing regions
2. Processes the data to match with commodity price data
3. Creates combined datasets for each commodity with its corresponding growing regions

Usage:
    python multi_commodity_climate.py [--download] [--start-year YEAR] [--end-year YEAR]
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import zipfile
import tempfile
import shutil
import cdsapi
import xarray as xr
from commodity_regions import COMMODITY_REGIONS, COMMODITY_NAMES, PRIMARY_REGIONS

def download_era5_monthly_data(region_name, bbox, start_year, end_year, output_dir="data/commodities"):
    """
    Download ERA5-Land monthly mean data for a specific region.
    
    Parameters:
    -----------
    region_name : str
        Name of the region
    bbox : list
        Bounding box [North, West, South, East]
    start_year : int
        Start year for the data download
    end_year : int
        End year for the data download
    output_dir : str
        Directory to save the output files
        
    Returns:
    --------
    str : Path to the downloaded file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file name: Region_StartYear_EndYear.nc
    filename = f"{region_name.replace(' ', '_')}_{start_year}_{end_year}.nc"
    output_file = os.path.join(output_dir, filename)
    
    # Years to download as strings
    years = [str(year) for year in range(start_year, end_year + 1)]
    
    # Define the variables we want
    variables = [
        "2m_temperature", 
        "2m_dewpoint_temperature", 
        "total_precipitation"
    ]
    
    print(f"Downloading monthly data for {region_name} ({start_year}-{end_year})...")
    
    try:
        # Initialize the CDS API client
        client = cdsapi.Client()
        
        # Request the data
        client.retrieve(
            "reanalysis-era5-land-monthly-means",
            {
                'product_type': 'monthly_averaged_reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'year': years,
                'month': [f"{m:02d}" for m in range(1, 13)],
                'time': '00:00',
                'area': bbox
            },
            output_file
        )
        
        print(f"Data saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading data for {region_name}: {e}")
        return None

def process_climate_file(file_path, region, commodity=None):
    """
    Process a NetCDF or GRIB file into a pandas DataFrame with monthly data.
    
    Parameters:
    -----------
    file_path : str
        Path to the climate data file
    region : str
        Name of the region
    commodity : str, optional
        Name of the commodity this region produces
        
    Returns:
    --------
    pd.DataFrame
        Processed monthly climate data
    """
    print(f"Processing {file_path} for {region}...")
    
    try:
        # Determine file type and open appropriately
        if file_path.endswith('.grib'):
            ds = xr.open_dataset(file_path, engine='cfgrib', 
                                backend_kwargs={'filter_by_keys': {'edition': 1}})
            print("Opened as GRIB file")
        else:
            ds = xr.open_dataset(file_path)
            print("Opened as NetCDF file")
        
        # Map expected variables to actual names in the dataset
        var_mapping = {
            "2m_temperature": next((v for v in ds.data_vars if "2m_temperature" in v or "t2m" in v or "temperature" in v.lower()), None),
            "2m_dewpoint_temperature": next((v for v in ds.data_vars if "2m_dewpoint" in v or "d2m" in v or "dewpoint" in v.lower()), None),
            "total_precipitation": next((v for v in ds.data_vars if "precipitation" in v.lower() or "tp" in v), None),
        }
        
        # Check if temperature and precipitation variables are found
        temp_var = var_mapping["2m_temperature"]
        precip_var = var_mapping["total_precipitation"]
        
        if not temp_var or not precip_var:
            missing = []
            if not temp_var: missing.append("temperature")
            if not precip_var: missing.append("precipitation")
            print(f"Missing critical variables: {missing}")
            return None
        
        # Calculate spatial mean across the region
        ds_mean = ds.mean(dim=["latitude", "longitude"], skipna=True)
        df = ds_mean.to_dataframe().reset_index()
        
        # Process temperature (K to °C)
        df["temperature_C"] = df[temp_var] - 273.15
        
        # Process precipitation
        df["precip_m"] = df[precip_var]
        
        # Process dewpoint and calculate humidity if available
        if var_mapping["2m_dewpoint_temperature"]:
            dewp_var = var_mapping["2m_dewpoint_temperature"]
            df["dewpoint_C"] = df[dewp_var] - 273.15
            
            # Calculate relative humidity (%)
            temp_C = df["temperature_C"]
            dew_C = df["dewpoint_C"]
            a, b = 17.27, 237.7
            sat_vp = 6.1094 * np.exp((a * temp_C) / (b + temp_C))
            act_vp = 6.1094 * np.exp((a * dew_C) / (b + dew_C))
            df["relative_humidity"] = 100 * act_vp / sat_vp
        
        # Add date and region
        df["Date"] = pd.to_datetime(df["time"])
        df["Region"] = region
        
        # Add commodity if provided
        if commodity:
            df["Commodity"] = commodity
        
        # Select relevant columns
        columns = ["Date", "Region", "temperature_C", "precip_m"]
        if commodity:
            columns.insert(2, "Commodity")
        if var_mapping["2m_dewpoint_temperature"]:
            columns.extend(["dewpoint_C", "relative_humidity"])
        
        monthly_df = df[columns].copy()
        
        # Add year and month columns
        monthly_df["Year"] = monthly_df["Date"].dt.year
        monthly_df["Month"] = monthly_df["Date"].dt.month
        
        print(f"Successfully processed {region} data with {len(monthly_df)} records")
        return monthly_df
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_commodity_data():
    """
    Load and preprocess the commodity data file.
    
    Returns:
    --------
    pd.DataFrame
        Processed commodity data
    """
    commodity_file = "combined_commodity_data.csv"
    print(f"Loading commodity data from {commodity_file}...")
    
    if not os.path.exists(commodity_file):
        print(f"Error: Commodity data file not found at {commodity_file}")
        return None
    
    try:
        # Load the data
        df = pd.read_csv(commodity_file)
        
        # Process the observation_date column
        df["Date"] = pd.to_datetime(df["observation_date"])
        
        # Extract year and month
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        
        # Create a list of all commodity code columns
        commodity_codes = [col for col in df.columns if col.startswith('P') and col != "PERIOD"]
        
        # Rename columns to friendly names for clarity
        column_renames = {}
        for code in commodity_codes:
            if code in COMMODITY_NAMES:
                column_renames[code] = COMMODITY_NAMES[code] + "_Price"
        
        # Rename columns
        if column_renames:
            df.rename(columns=column_renames, inplace=True)
        
        print(f"Loaded commodity data with {len(df)} records from "
              f"{df['Year'].min()} to {df['Year'].max()}")
        return df
        
    except Exception as e:
        print(f"Error loading commodity data: {e}")
        return None

def calculate_climate_signatures(climate_df):
    """
    Calculate climate signatures for analysis.
    
    Parameters:
    -----------
    climate_df : pd.DataFrame
        Monthly climate data
        
    Returns:
    --------
    pd.DataFrame
        Climate data with signature features added
    """
    print("Calculating climate signatures...")
    
    # Create a copy to avoid modifying the original
    df = climate_df.copy()
    
    # Group by region
    regions = df["Region"].unique()
    signature_dfs = []
    
    for region in regions:
        region_df = df[df["Region"] == region].copy()
        
        # Sort by date
        region_df = region_df.sort_values('Date')
        
        # Calculate temperature anomalies (difference from monthly mean)
        region_df["temp_anomaly"] = region_df.groupby("Month")["temperature_C"].transform(
            lambda x: x - x.mean())
        
        # Calculate precipitation anomalies
        region_df["precip_anomaly"] = region_df.groupby("Month")["precip_m"].transform(
            lambda x: x - x.mean())
        
        # Calculate rolling average temperature (3-month window)
        region_df["temp_3m_avg"] = region_df["temperature_C"].rolling(window=3, min_periods=1).mean()
        
        # Calculate cumulative precipitation (3-month window)
        region_df["precip_3m_sum"] = region_df["precip_m"].rolling(window=3, min_periods=1).sum()
        
        # Add drought index (simplistic)
        # Negative values indicate drought conditions
        region_df["drought_index"] = region_df["precip_anomaly"] - 0.1 * region_df["temp_anomaly"]
        
        # If humidity is available, calculate heat stress index
        if "relative_humidity" in region_df.columns:
            # Heat index calculation (simplified)
            region_df["heat_stress"] = region_df["temperature_C"] + 0.05 * region_df["relative_humidity"]
        
        signature_dfs.append(region_df)
    
    # Combine all regions back
    signatures_df = pd.concat(signature_dfs, ignore_index=True)
    
    print(f"Added climate signatures to data")
    return signatures_df

def join_climate_commodity_data(climate_df, commodity_df, commodity_name):
    """
    Join climate and commodity datasets for a specific commodity.
    
    Parameters:
    -----------
    climate_df : pd.DataFrame
        Climate data with Year and Month columns
    commodity_df : pd.DataFrame
        Commodity data with Year and Month columns
    commodity_name : str
        Name of the commodity to join on
        
    Returns:
    --------
    pd.DataFrame
        Joined dataset with both climate and commodity data
    """
    print(f"Joining climate and {commodity_name} price data...")
    
    # Create price column name
    price_col = f"{commodity_name}_Price"
    
    # Check if the price column exists
    if price_col not in commodity_df.columns:
        print(f"Error: {price_col} not found in commodity data")
        return None
    
    # Create a copy to avoid modifying originals
    climate = climate_df.copy()
    commodity = commodity_df[["Date", "Year", "Month", price_col]].copy()
    
    # Merge on Year and Month
    joined = pd.merge(climate, commodity, on=["Year", "Month"], how="inner",
                     suffixes=("_climate", "_commodity"))
    
    # Use the commodity Date as the primary date
    joined.drop(columns=["Date_climate"], inplace=True)
    joined.rename(columns={"Date_commodity": "Date"}, inplace=True)
    
    print(f"Joined data has {len(joined)} records covering {joined['Year'].min()}-{joined['Year'].max()}")
    
    return joined

def create_simulation_data(region_name, commodity, start_year=2015, end_year=2022):
    """
    Create simulated climate data when real data is not available.
    
    Parameters:
    -----------
    region_name : str
        Name of the region to simulate
    commodity : str
        Name of the commodity for this region
    start_year : int
        Start year for simulation
    end_year : int
        End year for simulation
        
    Returns:
    --------
    pd.DataFrame
        Simulated monthly climate data
    """
    print(f"Generating simulated climate data for {region_name} ({commodity})...")
    
    # Create all months in the date range
    start_date = pd.Timestamp(year=start_year, month=1, day=15)
    end_date = pd.Timestamp(year=end_year, month=12, day=15)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Initialize dataframe
    df = pd.DataFrame({'Date': dates})
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Region'] = region_name
    df['Commodity'] = commodity
    
    # Generate random but realistic temperature patterns based on region
    # Set baseline temperatures and seasonal amplitude based on region characteristics
    baseline_temps = {
        # Coffee regions tend to be mild, around 20°C
        "COLOMBIA": 20.0, 
        "MINAS GERAIS (BRA)": 22.0,
        "VIETNAM": 26.0,
        "ETHIOPIA": 21.0,
        
        # Cocoa regions are typically hot and humid
        "GHANA": 27.0,
        "IVORY COAST": 26.0,
        "INDONESIA": 28.0,
        
        # Wheat regions vary more by season
        "US GREAT PLAINS": 15.0,
        "UKRAINE": 10.0,
        "AUSTRALIA": 18.0,
        
        # Default for other regions
        "DEFAULT": 20.0
    }
    
    # Regional seasonal amplitude (how much temp varies through the year)
    seasonal_amp = {
        # Tropical regions have less seasonal variation
        "COLOMBIA": 3.0,
        "MINAS GERAIS (BRA)": 5.0,
        "VIETNAM": 2.5,
        "ETHIOPIA": 3.0,
        "GHANA": 2.0,
        "IVORY COAST": 2.5,
        "INDONESIA": 1.5,
        
        # Temperate regions have more seasonal variation
        "US GREAT PLAINS": 15.0,
        "UKRAINE": 20.0,
        "AUSTRALIA": 10.0,
        
        # Default
        "DEFAULT": 8.0
    }
    
    # Precipitation patterns (average monthly in meters)
    precip_base = {
        # Tropical regions have higher precipitation
        "COLOMBIA": 0.03,
        "MINAS GERAIS (BRA)": 0.015,
        "VIETNAM": 0.04,
        "ETHIOPIA": 0.02,
        "GHANA": 0.035,
        "IVORY COAST": 0.04,
        "INDONESIA": 0.05,
        
        # Temperate regions vary more
        "US GREAT PLAINS": 0.015,
        "UKRAINE": 0.01,
        "AUSTRALIA": 0.008,
        
        # Default
        "DEFAULT": 0.02
    }
    
    # Get baseline values for this region or use defaults
    base_temp = baseline_temps.get(region_name, baseline_temps["DEFAULT"])
    amp_temp = seasonal_amp.get(region_name, seasonal_amp["DEFAULT"])
    base_precip = precip_base.get(region_name, precip_base["DEFAULT"])
    
    # Generate temperature with seasonal pattern and random variation
    # Northern and Southern hemisphere have opposite seasons
    is_southern = region_name in ["MINAS GERAIS (BRA)", "AUSTRALIA", "BRAZIL SOY BELT", "ARGENTINA", "ARGENTINA SOY"]
    season_offset = 6 if is_southern else 0  # Shift seasons by 6 months for Southern Hemisphere
    
    # Generate temperatures with seasonal patterns
    df['temperature_C'] = df.apply(
        lambda row: base_temp + 
                   amp_temp * np.sin(2 * np.pi * ((row['Month'] + season_offset) % 12) / 12) + 
                   np.random.normal(0, 1),  # Random variation
        axis=1
    )
    
    # Generate precipitation with seasonal patterns
    # For many crops, precipitation is higher in growing season
    df['precip_m'] = df.apply(
        lambda row: max(0, base_precip * 
                       (1 + 0.5 * np.sin(2 * np.pi * ((row['Month'] + season_offset) % 12) / 12)) +
                       np.random.normal(0, base_precip/4)),  # Random variation
        axis=1
    )
    
    # Add dewpoint and relative humidity
    df['dewpoint_C'] = df['temperature_C'] - np.random.uniform(3, 8)  # Dewpoint is typically lower than temp
    
    # Calculate relative humidity
    temp_C = df['temperature_C']
    dew_C = df['dewpoint_C']
    a, b = 17.27, 237.7
    sat_vp = 6.1094 * np.exp((a * temp_C) / (b + temp_C))
    act_vp = 6.1094 * np.exp((a * dew_C) / (b + dew_C))
    df['relative_humidity'] = 100 * act_vp / sat_vp
    
    print(f"Generated {len(df)} months of simulated data for {region_name}")
    return df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download and process climate data for commodity regions")
    parser.add_argument("--download", action="store_true", help="Download new climate data")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for data")
    parser.add_argument("--end-year", type=int, default=2022, help="End year for data")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data instead of downloading")
    parser.add_argument("--commodities", nargs="+", 
                       choices=list(COMMODITY_REGIONS.keys()),
                       default=list(COMMODITY_REGIONS.keys()),
                       help="Specific commodities to process")
    args = parser.parse_args()
    
    # Configure paths
    base_dir = os.path.abspath(os.getcwd())
    data_dir = os.path.join(base_dir, "data")
    commodity_data_dir = os.path.join(data_dir, "commodities")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(commodity_data_dir, exist_ok=True)
    
    # Step 1: Load commodity data
    commodity_data = load_commodity_data()
    if commodity_data is None:
        print("Cannot proceed without commodity data.")
        return
    
    # Process each commodity
    for commodity in args.commodities:
        print(f"\n=== Processing {commodity} ===")
        
        # Get regions for this commodity
        commodity_regions = COMMODITY_REGIONS.get(commodity, {})
        if not commodity_regions:
            print(f"No regions defined for {commodity}. Skipping.")
            continue
        
        # First try to use primary region if specified
        primary_region = PRIMARY_REGIONS.get(commodity)
        if primary_region and primary_region in commodity_regions:
            regions_to_process = [primary_region]
            print(f"Using primary region for {commodity}: {primary_region}")
        else:
            # Otherwise use all regions
            regions_to_process = list(commodity_regions.keys())
            print(f"Using all regions for {commodity}: {regions_to_process}")
        
        # Track processed data for this commodity
        commodity_climate_dfs = []
        
        for region in regions_to_process:
            bbox = commodity_regions[region]
            
            # Option 1: Use simulated data
            if args.simulate:
                climate_df = create_simulation_data(
                    region,
                    commodity, 
                    args.start_year, 
                    args.end_year
                )
                
                if climate_df is not None:
                    # Add signatures
                    climate_df_with_signatures = calculate_climate_signatures(climate_df)
                    commodity_climate_dfs.append(climate_df_with_signatures)
                continue
            
            # Option 2: Download data
            if args.download:
                output_file = os.path.join(
                    commodity_data_dir,
                    f"{region.replace(' ', '_')}_{commodity}_{args.start_year}_{args.end_year}.nc"
                )
                
                downloaded_file = download_era5_monthly_data(
                    region, bbox, args.start_year, args.end_year, commodity_data_dir)
                
                if downloaded_file:
                    # Process the downloaded file
                    climate_df = process_climate_file(downloaded_file, region, commodity)
                    
                    if climate_df is not None:
                        # Add signatures
                        climate_df_with_signatures = calculate_climate_signatures(climate_df)
                        commodity_climate_dfs.append(climate_df_with_signatures)
                
            # Option 3: Check for existing file with commodity name
            else:
                # Look for files matching the pattern
                file_pattern = os.path.join(
                    commodity_data_dir,
                    f"{region.replace(' ', '_')}*{args.start_year}*.nc"
                )
                existing_files = glob.glob(file_pattern)
                
                if existing_files:
                    print(f"Found existing file for {region}: {existing_files[0]}")
                    climate_df = process_climate_file(existing_files[0], region, commodity)
                    
                    if climate_df is not None:
                        # Add signatures
                        climate_df_with_signatures = calculate_climate_signatures(climate_df)
                        commodity_climate_dfs.append(climate_df_with_signatures)
                else:
                    # Use simulated data as fallback
                    print(f"No data found for {region}, using simulated data")
                    climate_df = create_simulation_data(
                        region,
                        commodity, 
                        args.start_year, 
                        args.end_year
                    )
                    
                    if climate_df is not None:
                        # Add signatures
                        climate_df_with_signatures = calculate_climate_signatures(climate_df)
                        commodity_climate_dfs.append(climate_df_with_signatures)
        
        # Continue with the next commodity if no data was processed
        if not commodity_climate_dfs:
            print(f"No climate data was processed for {commodity}")
            continue
        
        # Combine all regions for this commodity
        combined_climate = pd.concat(commodity_climate_dfs, ignore_index=True)
        
        # Join with commodity price data
        joined_data = join_climate_commodity_data(combined_climate, commodity_data, commodity)
        
        if joined_data is not None:
            # Save to file
            output_file = os.path.join(base_dir, f"{commodity.lower()}_climate_joined.csv")
            joined_data.to_csv(output_file, index=False)
            print(f"Saved {commodity} climate and price data to {output_file}")
            
            # Report on regions
            print(f"\nRegions covered for {commodity}:")
            for region, count in joined_data.groupby('Region').size().items():
                print(f"- {region}: {count} records")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()