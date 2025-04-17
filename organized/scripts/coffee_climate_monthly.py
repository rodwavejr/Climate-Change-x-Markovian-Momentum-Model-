#!/usr/bin/env python3
"""
Coffee Climate Monthly Data Processor

This script:
1. Downloads monthly climate data for coffee-growing regions from ERA5-Land
2. Processes the data to match the commodity data timeline (1990-2024)
3. Provides utilities to join climate and commodity datasets
4. Adds additional coffee growing regions beyond Colombia and Brazil

Usage:
    python coffee_climate_monthly.py [--download] [--start-year YEAR] [--end-year YEAR]
"""

import os
import argparse
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import zipfile
import tempfile
import shutil

# Define coffee region bounding boxes [North, West, South, East]
COFFEE_REGION_BOUNDS = {
    "COLOMBIA": [7.0, -77.0, 3.0, -73.0],              # Colombian coffee region
    "MINAS GERAIS (BRA)": [-18.0, -48.0, -22.0, -44.0], # Brazil - Minas Gerais
    "VIETNAM": [16.0, 105.0, 10.0, 108.0],             # Vietnam Central Highlands
    "ETHIOPIA": [9.0, 35.0, 5.0, 39.0],                # Ethiopian coffee regions
    "INDONESIA": [0.0, 116.0, -8.0, 119.0],            # Indonesia (Sulawesi)
    "COSTA RICA": [11.0, -85.5, 8.0, -82.5],           # Costa Rica
    "GUATEMALA": [16.0, -92.0, 13.5, -88.0],           # Guatemala
    "KENYA": [1.0, 36.0, -2.0, 39.0],                  # Kenya
    "INDIA": [13.0, 75.0, 10.0, 78.0],                 # India (Karnataka)
    "HONDURAS": [15.5, -89.0, 13.0, -86.0]             # Honduras
}

def download_era5_monthly_data(region_name, bbox, start_year, end_year, output_dir="data"):
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

def extract_netcdf_from_zip(input_file, output_dir):
    """
    If the downloaded file is actually a ZIP archive, extract its contents.
    
    Parameters:
    -----------
    input_file : str
        Path to the input file (possibly a ZIP)
    output_dir : str
        Directory to extract files to
        
    Returns:
    --------
    str
        Path to the extracted NetCDF or GRIB file, or original file if not a ZIP
    """
    # Check if file is a ZIP
    try:
        with zipfile.ZipFile(input_file, 'r') as zipf:
            # It's a ZIP file, extract it
            print(f"Extracting {input_file} as a ZIP archive...")
            with tempfile.TemporaryDirectory() as temp_dir:
                zipf.extractall(temp_dir)
                
                # Look for NetCDF or GRIB files
                nc_files = glob.glob(os.path.join(temp_dir, "*.nc"))
                grib_files = glob.glob(os.path.join(temp_dir, "*.grib"))
                
                extract_files = nc_files or grib_files
                
                if extract_files:
                    # Get region name from original filename
                    base_name = os.path.basename(input_file)
                    region_name = base_name.split('_')[0]
                    
                    # Get the extension of the first found file
                    file_ext = os.path.splitext(extract_files[0])[1]
                    output_file = os.path.join(output_dir, f"{region_name}_extracted{file_ext}")
                    
                    # Copy the file to the output directory
                    shutil.copy2(extract_files[0], output_file)
                    print(f"Extracted to {output_file}")
                    return output_file
                else:
                    print(f"No NetCDF or GRIB files found in {input_file}")
                    return input_file  # Return original file if no extractable files found
        
    except zipfile.BadZipFile:
        # Not a ZIP file, return the original file
        return input_file

def process_climate_file(file_path, region):
    """
    Process a NetCDF or GRIB file into a pandas DataFrame with monthly data.
    
    Parameters:
    -----------
    file_path : str
        Path to the climate data file
    region : str
        Name of the region
        
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
        
        # Print available variables
        print(f"Available variables: {list(ds.data_vars)}")
        
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
        
        # Select relevant columns
        columns = ["Date", "Region", "temperature_C", "precip_m"]
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

def load_commodity_data(file_path):
    """
    Load and preprocess the commodity data file.
    
    Parameters:
    -----------
    file_path : str
        Path to the commodity data CSV
        
    Returns:
    --------
    pd.DataFrame
        Processed commodity data
    """
    print(f"Loading commodity data from {file_path}...")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Process the observation_date column
        df["Date"] = pd.to_datetime(df["observation_date"])
        
        # Extract year and month
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        
        # Select relevant columns
        # PCOFFOTMUSDM is likely the coffee price
        columns = ["Date", "Year", "Month", "PCOFFOTMUSDM"]
        
        # Add other commodity columns if needed
        other_commodities = ["PMAIZMTUSDM", "PWHEAMTUSDM", "PSOYBUSDM", 
                            "PCOTTINDUSDM", "PRICENPQUSDM", "PCOCOUSDM"]
        
        for col in other_commodities:
            if col in df.columns:
                columns.append(col)
        
        # Select final columns
        commodity_df = df[columns].copy()
        
        # Rename coffee price column for clarity
        commodity_df.rename(columns={"PCOFFOTMUSDM": "Coffee_Price"}, inplace=True)
        
        print(f"Loaded commodity data with {len(commodity_df)} records from {commodity_df['Year'].min()} to {commodity_df['Year'].max()}")
        return commodity_df
        
    except Exception as e:
        print(f"Error loading commodity data: {e}")
        return None

def join_climate_commodity_data(climate_df, commodity_df):
    """
    Join climate and commodity datasets on Year and Month.
    
    Parameters:
    -----------
    climate_df : pd.DataFrame
        Climate data with Year and Month columns
    commodity_df : pd.DataFrame
        Commodity data with Year and Month columns
        
    Returns:
    --------
    pd.DataFrame
        Joined dataset with both climate and commodity data
    """
    print("Joining climate and commodity data...")
    
    # Create a copy to avoid modifying originals
    climate = climate_df.copy()
    commodity = commodity_df.copy()
    
    # Merge on Year and Month
    joined = pd.merge(climate, commodity, on=["Year", "Month"], how="inner",
                     suffixes=("_climate", "_commodity"))
    
    # Use the commodity Date as the primary date
    joined.drop(columns=["Date_climate"], inplace=True)
    joined.rename(columns={"Date_commodity": "Date"}, inplace=True)
    
    print(f"Joined data has {len(joined)} records covering {joined['Year'].min()}-{joined['Year'].max()}")
    
    return joined

def calculate_climate_signatures(climate_df):
    """
    Calculate monthly climate signatures for each region.
    
    Parameters:
    -----------
    climate_df : pd.DataFrame
        Monthly climate data
        
    Returns:
    --------
    pd.DataFrame
        Climate data with additional signature features
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
        region_df.sort_values("Date", inplace=True)
        
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download and process monthly climate data for coffee regions")
    parser.add_argument("--download", action="store_true", help="Download new climate data")
    parser.add_argument("--start-year", type=int, default=1990, help="Start year for data")
    parser.add_argument("--end-year", type=int, default=2024, help="End year for data")
    parser.add_argument("--output", type=str, default="coffee_climate_monthly.csv", 
                       help="Output CSV file name")
    parser.add_argument("--regions", nargs="+", help="Specific regions to process (optional)")
    args = parser.parse_args()
    
    # Configure paths
    project_dir = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_dir, "data")
    extracted_dir = os.path.join(data_dir, "extracted")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(extracted_dir, exist_ok=True)
    
    # Filter regions if specified
    if args.regions:
        selected_regions = {r: COFFEE_REGION_BOUNDS[r] for r in args.regions if r in COFFEE_REGION_BOUNDS}
        if not selected_regions:
            print(f"Error: No valid regions found in {args.regions}")
            print(f"Available regions: {list(COFFEE_REGION_BOUNDS.keys())}")
            return
    else:
        selected_regions = COFFEE_REGION_BOUNDS
    
    # Step 1: Download data for each region if requested
    climate_files = {}
    if args.download:
        for region, bbox in selected_regions.items():
            output_file = download_era5_monthly_data(
                region, bbox, args.start_year, args.end_year, data_dir)
            
            if output_file:
                climate_files[region] = output_file
    else:
        # Use existing files
        for region in selected_regions:
            filename = f"{region.replace(' ', '_')}_{args.start_year}_{args.end_year}.nc"
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                climate_files[region] = filepath
            else:
                print(f"Warning: No data file found for {region}. Use --download to get the data.")
    
    # Step 2: Process climate files
    region_dfs = {}
    for region, file_path in climate_files.items():
        # Check if the file is a ZIP and extract if needed
        actual_file = extract_netcdf_from_zip(file_path, extracted_dir)
        
        # Process the file
        df = process_climate_file(actual_file, region)
        if df is not None:
            region_dfs[region] = df
    
    # Step 3: Combine all regions' climate data
    if region_dfs:
        combined_climate = pd.concat(list(region_dfs.values()), ignore_index=True)
        
        # Calculate climate signatures
        combined_climate_with_signatures = calculate_climate_signatures(combined_climate)
        
        # Save climate data
        climate_output = os.path.join(project_dir, args.output)
        combined_climate_with_signatures.to_csv(climate_output, index=False)
        print(f"Saved monthly climate data to {climate_output}")
        
        # Step 4: Load commodity data
        commodity_file = os.path.join(project_dir, "combined_commodity_data.csv")
        if os.path.exists(commodity_file):
            commodity_df = load_commodity_data(commodity_file)
            
            if commodity_df is not None:
                # Step 5: Join climate and commodity data
                joined_data = join_climate_commodity_data(combined_climate_with_signatures, commodity_df)
                
                # Save joined data
                joined_output = os.path.join(project_dir, "climate_commodity_joined.csv")
                joined_data.to_csv(joined_output, index=False)
                print(f"Saved joined climate and commodity data to {joined_output}")
                
                # Display summary
                print("\nData summary:")
                print(f"Climate data: {len(combined_climate_with_signatures)} records from {combined_climate_with_signatures['Year'].min()} to {combined_climate_with_signatures['Year'].max()}")
                print(f"Commodity data: {len(commodity_df)} records from {commodity_df['Year'].min()} to {commodity_df['Year'].max()}")
                print(f"Joined data: {len(joined_data)} records from {joined_data['Year'].min()} to {joined_data['Year'].max()}")
                
                # Report on regions
                print("\nRegions covered:")
                for region, count in combined_climate_with_signatures.groupby('Region').size().items():
                    print(f"- {region}: {count} records")
        else:
            print(f"Warning: Commodity data file not found at {commodity_file}")
    else:
        print("No climate data was successfully processed")

if __name__ == "__main__":
    main()