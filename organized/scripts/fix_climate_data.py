#!/usr/bin/env python3
"""
Fix Climate Data Files and Generate CSV

This script:
1. Checks existing .nc files that are actually ZIP archives
2. Extracts the NetCDF or GRIB files from the ZIPs
3. Processes the data and saves it to a CSV file

Usage:
    python fix_climate_data.py
"""

import os
import zipfile
import glob
import tempfile
import shutil
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import cfgrib  # For reading GRIB files

def extract_netcdf_from_zip(input_file, output_dir):
    """
    Extract NetCDF files from a ZIP archive.
    
    Parameters:
    -----------
    input_file : str
        Path to the input ZIP file
    output_dir : str
        Directory to extract files to
        
    Returns:
    --------
    str
        Path to the extracted NetCDF file, or None if extraction failed
    """
    print(f"Extracting {input_file}...")
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the ZIP file
            with zipfile.ZipFile(input_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find NetCDF files or GRIB files in the temporary directory
            nc_files = glob.glob(os.path.join(temp_dir, "*.nc"))
            if not nc_files:
                # Try to find GRIB files if no NetCDF files are found
                grib_files = glob.glob(os.path.join(temp_dir, "*.grib"))
                if grib_files:
                    print(f"Found GRIB file instead of NetCDF in {input_file}")
                    # We can process GRIB files too
                    nc_files = grib_files
                else:
                    print(f"No NetCDF or GRIB files found in {input_file}")
                    return None
            
            # Get the first NetCDF file
            first_nc = nc_files[0]
            
            # Generate output filename
            region_name = os.path.basename(input_file).split('_')[0]
            # Keep the original extension for the output file
            file_ext = os.path.splitext(first_nc)[1]
            output_file = os.path.join(output_dir, f"{region_name}_extracted{file_ext}")
            
            # Copy the file to the output directory
            shutil.copy2(first_nc, output_file)
            print(f"Extracted to {output_file}")
            return output_file
            
    except Exception as e:
        print(f"Error extracting {input_file}: {e}")
        return None

def process_netcdf_file(nc_file, region):
    """
    Process a NetCDF file or GRIB file into a pandas DataFrame.
    
    Parameters:
    -----------
    nc_file : str
        Path to the NetCDF or GRIB file
    region : str
        Name of the region
        
    Returns:
    --------
    pd.DataFrame
        Processed climate data, or None if processing failed
    """
    print(f"Processing {nc_file} for {region}...")
    try:
        # Check if it's a GRIB file
        if nc_file.endswith('.grib'):
            # Open with cfgrib engine
            ds = xr.open_dataset(nc_file, engine='cfgrib')
            print("Opened as GRIB file")
        else:
            # Open as NetCDF
            ds = xr.open_dataset(nc_file)
            print("Opened as NetCDF file")
        
        # Check the available variables
        print(f"Available variables: {list(ds.data_vars)}")
        
        # Map expected variables to actual names in the dataset
        var_mapping = {
            "2m_temperature": next((v for v in ds.data_vars if "2m_temperature" in v or "t2m" in v or "temperature" in v.lower()), None),
            "2m_dewpoint_temperature": next((v for v in ds.data_vars if "2m_dewpoint" in v or "d2m" in v or "dewpoint" in v.lower()), None),
            "total_precipitation": next((v for v in ds.data_vars if "precipitation" in v.lower() or "tp" in v), None),
            "volumetric_soil_water_layer_1": next((v for v in ds.data_vars if "soil_water" in v or "swvl1" in v or "soil" in v.lower()), None)
        }
        
        # Check if required temperature and precipitation variables are found
        critical_vars = ["2m_temperature", "total_precipitation"]
        missing_critical = [k for k in critical_vars if var_mapping[k] is None]
        
        if missing_critical:
            print(f"Missing critical variables: {missing_critical}")
            return None
            
        # Report any missing but optional variables
        optional_vars = ["2m_dewpoint_temperature", "volumetric_soil_water_layer_1"]
        missing_optional = [k for k in optional_vars if var_mapping[k] is None]
        
        if missing_optional:
            print(f"Missing optional variables: {missing_optional} - will continue processing")
            
        # If dewpoint is missing, we can't calculate humidity
        if var_mapping["2m_dewpoint_temperature"] is None:
            print("Dewpoint missing - cannot calculate relative humidity")
        
        # Calculate spatial mean across the region
        ds_mean = ds.mean(dim=["latitude", "longitude"], skipna=True)
        df = ds_mean.to_dataframe().reset_index()
        
        # Extract the correct variable columns
        temp_var = var_mapping["2m_temperature"]
        precip_var = var_mapping["total_precipitation"]
        
        # Initialize optional variables
        has_dewpoint = var_mapping["2m_dewpoint_temperature"] is not None
        has_soil = var_mapping["volumetric_soil_water_layer_1"] is not None
        
        # Convert Kelvin to Celsius for temperature
        df["t2m"] = df[temp_var] - 273.15
        
        # Process dewpoint and calculate humidity if available
        has_humidity = False
        if has_dewpoint:
            dewp_var = var_mapping["2m_dewpoint_temperature"]
            df["d2m"] = df[dewp_var] - 273.15
            
            # Calculate relative humidity (%)
            temp_C = df["t2m"]
            dew_C = df["d2m"]
            a, b = 17.27, 237.7
            sat_vp = 6.1094 * np.exp((a * temp_C) / (b + temp_C))
            act_vp = 6.1094 * np.exp((a * dew_C) / (b + dew_C))
            df["relative_humidity"] = 100 * act_vp / sat_vp
            has_humidity = True
        
        # Rename columns
        col_mapping = {
            "time": "Date",
            precip_var: "precip_m",
        }
        
        # Add soil moisture if available
        if has_soil:
            soil_var = var_mapping["volumetric_soil_water_layer_1"]
            col_mapping[soil_var] = "soil_moisture"
            
        df.rename(columns=col_mapping, inplace=True)
        
        # Select columns based on what's available
        columns = ["Date", "t2m"]
        if has_dewpoint:
            columns.append("d2m")
        if has_humidity:
            columns.append("relative_humidity")
        columns.append("precip_m")
        if has_soil:
            columns.append("soil_moisture")
            
        # Select and index by Date
        df = df[columns]
        df["Region"] = region
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        
        # Resample to quarterly data
        # Start with required fields
        resampled_data = {
            "avg_temp_C": df["t2m"].resample("Q").mean(),
            "total_precip_m": df["precip_m"].resample("Q").sum(),
        }
        
        # Add optional fields if available
        if has_dewpoint:
            resampled_data["avg_dewpoint_C"] = df["d2m"].resample("Q").mean()
        if has_humidity:
            resampled_data["avg_rel_humidity_pct"] = df["relative_humidity"].resample("Q").mean()
        if has_soil:
            resampled_data["avg_soil_moisture"] = df["soil_moisture"].resample("Q").mean()
            
        # Create the quarterly DataFrame    
        climate_q = pd.DataFrame(resampled_data)
        climate_q["Year"] = climate_q.index.year
        climate_q["Quarter"] = climate_q.index.quarter
        climate_q.reset_index(drop=True, inplace=True)
        climate_q["Region"] = region
        
        return climate_q
        
    except Exception as e:
        print(f"Error processing {nc_file}: {e}")
        return None

def main():
    # Configure paths
    project_dir = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_dir, "data")
    extracted_dir = os.path.join(data_dir, "extracted")
    
    # Create extracted directory if it doesn't exist
    os.makedirs(extracted_dir, exist_ok=True)
    
    # Define region names
    region_mapping = {
        "COLOMBIA": "COLOMBIA",
        "MINAS_GERAIS_(BRA)": "MINAS GERAIS (BRA)",
        "MINAS": "MINAS GERAIS (BRA)"  # Add this in case the parentheses cause filename issues
    }
    
    # Find all .nc files in the data directory (which are actually ZIP files)
    zip_files = glob.glob(os.path.join(data_dir, "*.nc"))
    
    if not zip_files:
        print("No .nc files found in the data directory")
        return
    
    # Extract NetCDF files from ZIP archives
    extracted_files = {}
    for zip_file in zip_files:
        basename = os.path.basename(zip_file)
        region_key = basename.split('_')[0]
        
        if region_key in region_mapping:
            extracted_file = extract_netcdf_from_zip(zip_file, extracted_dir)
            if extracted_file:
                extracted_files[region_mapping[region_key]] = extracted_file
    
    if not extracted_files:
        print("No valid NetCDF files were extracted")
        return
    
    # Process the extracted NetCDF files
    region_dfs = {}
    for region, nc_file in extracted_files.items():
        df = process_netcdf_file(nc_file, region)
        if df is not None:
            region_dfs[region] = df
    
    if not region_dfs:
        print("No regions were successfully processed")
        return
    
    # Combine all regions' data
    combined_data = pd.concat(region_dfs.values(), ignore_index=True)
    combined_data.sort_values(["Region", "Year", "Quarter"], inplace=True)
    combined_data.reset_index(drop=True, inplace=True)
    print(f"Successfully processed data for {len(region_dfs)} regions")
    
    # Save to CSV
    output_file = os.path.join(project_dir, "coffee_climate_data.csv")
    combined_data.to_csv(output_file, index=False)
    print(f"Saved climate data to {output_file}")
    
    # Display summary
    print("\nData points per region:")
    for region, count in combined_data.groupby('Region').size().items():
        print(f"- {region}: {count} data points")

if __name__ == "__main__":
    main()