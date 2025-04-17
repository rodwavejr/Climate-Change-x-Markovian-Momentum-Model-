#!/usr/bin/env python3
"""
Climate Data Downloader and Processor for Coffee Regions
--------------------------------------------------------
A simplified script that downloads and processes climate data for coffee-growing regions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Check NumPy version to warn about compatibility issues
np_version = np.__version__
if np_version.startswith('2.'):
    print(f"WARNING: Using NumPy {np_version}. This may cause issues with NetCDF4 libraries.")
    print("Consider using the Machine_Learning virtual environment with NumPy 1.24.3.")

def generate_sample_data(start_year=2020, end_year=2022, output_file=None):
    """
    Generate sample coffee climate data when API or NetCDF processing fails.
    
    Parameters:
    -----------
    start_year : int
        First year to include in the data
    end_year : int
        Last year to include in the data
    output_file : str, optional
        Path to save the CSV output
        
    Returns:
    --------
    pd.DataFrame
        Generated climate data
    """
    print(f"Generating sample data for years {start_year}-{end_year}...")
    
    # Define coffee region data 
    coffee_regions = ["MINAS GERAIS (BRA)", "COLOMBIA"]
    
    # Calculate quarters
    quarters = (end_year - start_year + 1) * 4
    
    # Create empty dataframe
    coffee_climate = pd.DataFrame()
    
    for region in coffee_regions:
        # Generate some random but plausible climate data
        np.random.seed(42 if region == "COLOMBIA" else 24)  # Different seed for different regions
        
        # Temperature ranges (Celsius)
        if region == "COLOMBIA":
            temp_mean, temp_std = 22, 2  # Colombian coffee highlands
        else:
            temp_mean, temp_std = 24, 3  # Brazilian coffee regions
        
        # Generate data for each quarter
        years = []
        quarters_list = []
        
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                years.append(year)
                quarters_list.append(quarter)
        
        # Generate climate variables
        avg_temp = np.random.normal(temp_mean, temp_std, quarters)
        avg_dewpoint = avg_temp - np.random.uniform(2, 5, quarters)
        avg_humidity = np.random.uniform(65, 85, quarters)
        total_precip = np.random.uniform(0.1, 0.5, quarters)  # in meters
        soil_moisture = np.random.uniform(0.2, 0.6, quarters)
        
        # Create region dataframe
        df = pd.DataFrame({
            "avg_temp_C": avg_temp,
            "avg_dewpoint_C": avg_dewpoint,
            "avg_rel_humidity_pct": avg_humidity,
            "total_precip_m": total_precip,
            "avg_soil_moisture": soil_moisture,
            "Year": years,
            "Quarter": quarters_list,
            "Region": region
        })
        
        # Append to master dataframe
        coffee_climate = pd.concat([coffee_climate, df])
    
    # Reset index and make sure data is sorted
    coffee_climate.reset_index(drop=True, inplace=True)
    coffee_climate.sort_values(["Region", "Year", "Quarter"], inplace=True)
    
    # Save to CSV if output file specified
    if output_file:
        coffee_climate.to_csv(output_file, index=False)
        print(f"Saved simulated climate data to {output_file}")
    
    return coffee_climate

def download_real_data(start_year=2020, end_year=2022, output_file=None):
    """
    Try to download real data using CDS API.
    This function requires the proper virtual environment with NumPy 1.24.3.
    
    Parameters:
    -----------
    start_year : int
        First year to include in the data
    end_year : int
        Last year to include in the data
    output_file : str, optional
        Path to save the CSV output
        
    Returns:
    --------
    pd.DataFrame or None
        Downloaded and processed climate data, or None if failed
    """
    try:
        # Only try to import CDSAPI if we're in the right environment
        import cdsapi
        import xarray as xr
        
        # Define coffee region bounding boxes [North, West, South, East]
        coffee_region_bounds = {
            "MINAS GERAIS (BRA)": [-18.0, -48.0, -22.0, -44.0],  # Coffee areas in Minas Gerais
            "COLOMBIA": [7.0, -77.0, 3.0, -73.0]                # Colombian coffee region (Andes)
        }
        
        # Define variables and years for retrieval
        variables = ["2m_temperature", "2m_dewpoint_temperature", 
                     "total_precipitation", "volumetric_soil_water_layer_1"]
        years = [str(year) for year in range(start_year, end_year + 1)]
        
        print(f"Attempting to download real climate data for years {start_year}-{end_year}...")
        
        # Create a directory for data files if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        coffee_climate_dfs = {}
        client = cdsapi.Client()  # Initialize the client once
        
        for region, bbox in coffee_region_bounds.items():
            try:
                print(f"Processing {region}...")
                
                # Define a unique filename for this region's NetCDF file
                nc_file = os.path.join(data_dir, f"{region.replace(' ', '_')}_{start_year}_{end_year}.nc")
                
                # Download data if file doesn't exist
                if not os.path.exists(nc_file):
                    print(f"Downloading data for {region}...")
                    
                    # Retrieve ERA5-Land monthly data for this coffee region
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
                        nc_file
                    )
                    print(f"Data saved to {nc_file}")
                else:
                    print(f"Using existing data file: {nc_file}")
                
                # Try to open the dataset - with retry on different engines if needed
                try:
                    print(f"Opening dataset with netcdf4 engine...")
                    ds = xr.open_dataset(nc_file, engine='netcdf4')
                except Exception as e1:
                    print(f"Failed with netcdf4 engine: {e1}")
                    try:
                        print(f"Trying scipy engine...")
                        ds = xr.open_dataset(nc_file, engine='scipy')
                    except Exception as e2:
                        print(f"Failed with scipy engine: {e2}")
                        raise Exception("Could not open NetCDF file with any available engine")
                
                # Check the actual variable names in the dataset
                print(f"Available variables: {list(ds.data_vars)}")
                
                # Map expected variables to actual names in the dataset
                var_mapping = {
                    "2m_temperature": next((v for v in ds.data_vars if "2m_temperature" in v), None),
                    "2m_dewpoint_temperature": next((v for v in ds.data_vars if "2m_dewpoint_temperature" in v), None),
                    "total_precipitation": next((v for v in ds.data_vars if "precipitation" in v), None),
                    "volumetric_soil_water_layer_1": next((v for v in ds.data_vars if "soil_water" in v), None)
                }
                
                # Calculate spatial mean across the region
                ds_mean = ds.mean(dim=["latitude", "longitude"], skipna=True)
                df = ds_mean.to_dataframe().reset_index()
                
                # Extract the correct variable columns based on what's actually in the dataset
                temp_var = var_mapping["2m_temperature"]
                dewp_var = var_mapping["2m_dewpoint_temperature"]
                precip_var = var_mapping["total_precipitation"]
                soil_var = var_mapping["volumetric_soil_water_layer_1"]
                
                # Continue only if we have the necessary variables
                if all([temp_var, dewp_var, precip_var, soil_var]):
                    # Convert Kelvin to Celsius
                    df["t2m"] = df[temp_var] - 273.15
                    df["d2m"] = df[dewp_var] - 273.15
                    
                    # Calculate relative humidity (%)
                    temp_C = df["t2m"]
                    dew_C = df["d2m"]
                    a, b = 17.27, 237.7
                    sat_vp = 6.1094 * np.exp((a * temp_C) / (b + temp_C))
                    act_vp = 6.1094 * np.exp((a * dew_C) / (b + dew_C))
                    df["relative_humidity"] = 100 * act_vp / sat_vp
                    
                    # Rename columns
                    col_mapping = {
                        "time": "Date",
                        precip_var: "precip_m",
                        soil_var: "soil_moisture"
                    }
                    df.rename(columns=col_mapping, inplace=True)
                    
                    # Select and index by Date
                    df = df[["Date", "t2m", "d2m", "relative_humidity", "precip_m", "soil_moisture"]]
                    df["Region"] = region
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    
                    # Resample to quarterly
                    climate_q = pd.DataFrame({
                        "avg_temp_C": df["t2m"].resample("Q").mean(),
                        "avg_dewpoint_C": df["d2m"].resample("Q").mean(),
                        "avg_rel_humidity_pct": df["relative_humidity"].resample("Q").mean(),
                        "total_precip_m": df["precip_m"].resample("Q").sum(),
                        "avg_soil_moisture": df["soil_moisture"].resample("Q").mean()
                    })
                    climate_q["Year"] = climate_q.index.year
                    climate_q["Quarter"] = climate_q.index.quarter
                    climate_q.reset_index(drop=True, inplace=True)
                    climate_q["Region"] = region
                    coffee_climate_dfs[region] = climate_q
                    
                    print(f"Successfully processed {region}")
                else:
                    print(f"Error: Could not find all required variables in the dataset for {region}")
                    print(f"Available variables: {list(ds.data_vars)}")
                    print(f"Variable mapping: {var_mapping}")
            
            except Exception as e:
                print(f"Error processing {region}: {e}")
        
        # Combine climate data for regions that were successfully processed
        if coffee_climate_dfs:
            coffee_climate = pd.concat(coffee_climate_dfs.values(), ignore_index=True)
            print("\nSuccessfully processed the following regions:")
            for region in coffee_climate_dfs.keys():
                print(f"- {region}")
            
            # Save to CSV if specified
            if output_file:
                coffee_climate.to_csv(output_file, index=False)
                print(f"Saved real climate data to {output_file}")
            
            return coffee_climate
        else:
            print("No regions were successfully processed. Falling back to sample data.")
            return None
            
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure you're using the correct virtual environment with compatible packages.")
        return None
    except Exception as e:
        print(f"Error downloading real data: {e}")
        return None

def plot_climate_data(climate_df, output_dir=None, show_plots=False):
    """
    Create visualizations of the climate data.
    
    Parameters:
    -----------
    climate_df : pd.DataFrame
        Processed climate data
    output_dir : str, optional
        Directory to save the plot images
    show_plots : bool, default=False
        Whether to display plots interactively
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Configure matplotlib for non-interactive backend if not showing plots
    if not show_plots:
        plt.switch_backend('Agg')
    
    # Temperature comparison
    plt.figure(figsize=(12, 6))
    regions = climate_df['Region'].unique()
    
    for region in regions:
        region_data = climate_df[climate_df['Region'] == region]
        region_data = region_data.sort_values(['Year', 'Quarter'])
        x = [f"{year}-Q{quarter}" for year, quarter in 
             zip(region_data['Year'], region_data['Quarter'])]
        plt.plot(x, region_data['avg_temp_C'], marker='o', label=region)
    
    plt.title('Average Temperature Comparison Between Coffee Regions')
    plt.xlabel('Time Period')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'temperature_comparison.png'))
    
    # Precipitation comparison
    plt.figure(figsize=(12, 6))
    
    for region in regions:
        region_data = climate_df[climate_df['Region'] == region]
        region_data = region_data.sort_values(['Year', 'Quarter'])
        x = [f"{year}-Q{quarter}" for year, quarter in 
             zip(region_data['Year'], region_data['Quarter'])]
        plt.plot(x, region_data['total_precip_m'], marker='o', label=region)
    
    plt.title('Total Precipitation Comparison Between Coffee Regions')
    plt.xlabel('Time Period')
    plt.ylabel('Precipitation (m)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'precipitation_comparison.png'))
        print(f"Plots saved to {output_dir}")
    
    if show_plots:
        plt.show()

def main():
    """Main function to handle command line arguments and run the script."""
    parser = argparse.ArgumentParser(description='Download and process climate data for coffee regions')
    parser.add_argument('--start', type=int, default=2020, help='Start year for data')
    parser.add_argument('--end', type=int, default=2022, help='End year for data')
    parser.add_argument('--output', type=str, default='coffee_climate_data.csv', help='Output CSV file path')
    parser.add_argument('--sample-only', action='store_true', help='Use sample data only, skip API download')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    parser.add_argument('--plot-dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--show-plots', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Try to download real data unless sample-only is specified
    climate_data = None
    if not args.sample_only:
        climate_data = download_real_data(args.start, args.end, args.output)
    
    # Fall back to sample data if real data download failed
    if climate_data is None:
        climate_data = generate_sample_data(args.start, args.end, args.output)
    
    # Generate plots if requested
    if args.plots:
        plot_climate_data(climate_data, args.plot_dir, show_plots=args.show_plots)
    
    print("Processing complete!")
    return climate_data

if __name__ == "__main__":
    main()