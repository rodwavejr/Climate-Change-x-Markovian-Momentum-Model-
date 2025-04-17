#!/usr/bin/env python3
"""
Process Extended Climate Data (2015-2022)

This script:
1. Extracts data from the 2015-2022 NetCDF files (which are actually ZIP archives)
2. Processes the GRIB files inside them
3. Creates an extended CSV file with data from 2015-2022

Usage:
    python3 run_extended_data.py
"""

import os
import zipfile
import tempfile
import glob
import shutil
import pandas as pd
import numpy as np
import xarray as xr
import cfgrib
from datetime import datetime

def process_region_data(zip_file, region_name):
    """
    Process data for a specific region from a ZIP file.
    
    Parameters:
    -----------
    zip_file : str
        Path to the ZIP file with .nc extension
    region_name : str
        Name of the region
        
    Returns:
    --------
    pd.DataFrame
        Processed climate data, or None if processing failed
    """
    print(f"Processing {region_name} data from {os.path.basename(zip_file)}...")
    
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                print(f"  Extracted files: {zip_ref.namelist()}")
            
            # Find GRIB files in the temp directory
            grib_files = glob.glob(os.path.join(temp_dir, "*.grib"))
            
            if grib_files:
                print(f"  Found {len(grib_files)} GRIB files")
                grib_file = grib_files[0]
                
                try:
                    # Process the GRIB file
                    print(f"  Reading GRIB file with cfgrib engine")
                    
                    # Load the dataset with filter_by_keys to avoid index errors
                    ds = xr.open_dataset(grib_file, engine='cfgrib', 
                                       backend_kwargs={'filter_by_keys': {'edition': 1}})
                    
                    # Print available variables
                    variables = list(ds.data_vars)
                    print(f"  Available variables: {variables}")
                    
                    # Calculate spatial mean across the region
                    print(f"  Calculating spatial mean...")
                    ds_mean = ds.mean(dim=["latitude", "longitude"], skipna=True)
                    
                    # Convert to pandas DataFrame
                    df = ds_mean.to_dataframe().reset_index()
                    
                    # Process temperature data (convert K to °C)
                    df["temperature_C"] = df["t2m"] - 273.15
                    
                    # Process dewpoint if available
                    if "d2m" in variables:
                        df["dewpoint_C"] = df["d2m"] - 273.15
                        
                        # Calculate relative humidity
                        temp_C = df["temperature_C"]
                        dew_C = df["dewpoint_C"]
                        a, b = 17.27, 237.7
                        sat_vp = 6.1094 * np.exp((a * temp_C) / (b + temp_C))
                        act_vp = 6.1094 * np.exp((a * dew_C) / (b + dew_C))
                        df["relative_humidity"] = 100 * act_vp / sat_vp
                    
                    # Add precipitation if available
                    if "tp" in variables:
                        df["precip_m"] = df["tp"]
                    
                    # Add date and region
                    df["Date"] = pd.to_datetime(df["time"])
                    df["Region"] = region_name
                    
                    # Select and prepare data for resampling
                    print(f"  Preparing data for resampling...")
                    df_clean = df[["Date", "Region", "temperature_C", "precip_m"]].copy()
                    if "d2m" in variables:
                        df_clean["dewpoint_C"] = df["dewpoint_C"]
                        df_clean["relative_humidity"] = df["relative_humidity"]
                        
                    # Set index for resampling
                    df_clean.set_index("Date", inplace=True)
                    
                    # Resample to quarterly data
                    print(f"  Resampling to quarterly data...")
                    resampled_data = {
                        "avg_temp_C": df_clean["temperature_C"].resample("Q").mean(),
                        "total_precip_m": df_clean["precip_m"].resample("Q").sum(),
                    }
                    
                    if "d2m" in variables:
                        resampled_data["avg_dewpoint_C"] = df_clean["dewpoint_C"].resample("Q").mean()
                        resampled_data["avg_rel_humidity_pct"] = df_clean["relative_humidity"].resample("Q").mean()
                        
                    # Create quarterly DataFrame
                    climate_q = pd.DataFrame(resampled_data)
                    climate_q["Year"] = climate_q.index.year
                    climate_q["Quarter"] = climate_q.index.quarter
                    climate_q["Region"] = region_name
                    climate_q.reset_index(drop=True, inplace=True)
                    
                    print(f"  Successfully processed {region_name} with {len(climate_q)} records")
                    return climate_q
                    
                except Exception as e:
                    print(f"  Error processing GRIB file: {e}")
                    return None
            else:
                print(f"  No GRIB files found in extracted data for {region_name}")
                return None
    except Exception as e:
        print(f"  Error processing {region_name}: {e}")
        return None

def main():
    # Configure paths
    project_dir = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_dir, "data")
    
    # Define region names and files
    region_files = {
        "COLOMBIA": os.path.join(data_dir, "COLOMBIA_2015_2022.nc"),
        "MINAS GERAIS (BRA)": os.path.join(data_dir, "MINAS_GERAIS_(BRA)_2015_2022.nc")
    }
    
    # Process each region
    region_dfs = {}
    for region_name, zip_file in region_files.items():
        if os.path.exists(zip_file):
            region_df = process_region_data(zip_file, region_name)
            if region_df is not None:
                region_dfs[region_name] = region_df
    
    # Combine all regions' data
    if region_dfs:
        combined_data = pd.concat(region_dfs.values(), ignore_index=True)
        combined_data.sort_values(["Region", "Year", "Quarter"], inplace=True)
        combined_data.reset_index(drop=True, inplace=True)
        
        # Save to CSV
        output_file = os.path.join(project_dir, "coffee_climate_extended.csv")
        combined_data.to_csv(output_file, index=False)
        print(f"Saved extended climate data to {output_file}")
        
        # Display summary
        print("\nData points per region:")
        for region, count in combined_data.groupby('Region').size().items():
            print(f"- {region}: {count} data points")
            
        # Show years covered
        years = sorted(combined_data['Year'].unique())
        print(f"\nYears covered: {years[0]}-{years[-1]}")
    else:
        print("No regions were successfully processed")

if __name__ == "__main__":
    main()