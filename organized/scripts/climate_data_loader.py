# Climate Data Loader for Coffee Regions
import os
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def download_era5_data(region_name, bbox, years, output_file=None):
    """
    Download ERA5-Land monthly mean data for a specific region.
    
    Parameters:
    -----------
    region_name : str
        Name of the region (for labeling only)
    bbox : list
        Bounding box [North, West, South, East]
    years : list
        List of years to download as strings
    output_file : str, optional
        Output file path. If None, a default name will be generated
        
    Returns:
    --------
    str : Path to the downloaded file
    """
    if output_file is None:
        output_file = f"{region_name.replace(' ', '_')}_era5land_data.nc"
    
    # Define the variables we want
    variables = [
        "2m_temperature", 
        "2m_dewpoint_temperature", 
        "total_precipitation", 
        "volumetric_soil_water_layer_1"
    ]
    
    print(f"Downloading data for {region_name}...")
    
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

def process_climate_data(nc_file, region_name):
    """
    Process the NetCDF file into a pandas DataFrame with climate variables.
    
    Parameters:
    -----------
    nc_file : str
        Path to the NetCDF file
    region_name : str
        Name of the region for labeling
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with processed climate data
    """
    print(f"Processing data for {region_name}...")
    
    # Open the dataset
    try:
        ds = xr.open_dataset(nc_file)
        print(f"Dataset opened successfully. Variables: {list(ds.data_vars)}")
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return None
    
    # Calculate mean over the region
    ds_mean = ds.mean(dim=["latitude", "longitude"], skipna=True)
    
    # Convert to DataFrame
    df = ds_mean.to_dataframe().reset_index()
    
    # Identify variable names (may differ between versions)
    var_names = list(ds.data_vars)
    temp_var = next((v for v in var_names if "2m_temperature" in v), None)
    dewp_var = next((v for v in var_names if "2m_dewpoint_temperature" in v), None)
    precip_var = next((v for v in var_names if "precipitation" in v), None)
    soil_var = next((v for v in var_names if "soil_water" in v), None)
    
    if not all([temp_var, dewp_var, precip_var, soil_var]):
        print(f"Warning: Not all required variables found. Available: {var_names}")
        # Fall back to expected names if needed
        temp_var = temp_var or "2m_temperature"
        dewp_var = dewp_var or "2m_dewpoint_temperature" 
        precip_var = precip_var or "total_precipitation"
        soil_var = soil_var or "volumetric_soil_water_layer_1"
    
    # Convert temperature from Kelvin to Celsius
    df["t2m"] = df[temp_var] - 273.15
    df["d2m"] = df[dewp_var] - 273.15
    
    # Calculate relative humidity
    temp_C = df["t2m"]
    dew_C = df["d2m"]
    a, b = 17.27, 237.7
    sat_vp = 6.1094 * np.exp((a * temp_C) / (b + temp_C))
    act_vp = 6.1094 * np.exp((a * dew_C) / (b + dew_C))
    df["relative_humidity"] = 100 * act_vp / sat_vp
    
    # Rename columns
    df.rename(columns={
        "time": "Date",
        precip_var: "precip_m",
        soil_var: "soil_moisture"
    }, inplace=True)
    
    # Select columns and prepare final DataFrame
    df = df[["Date", "t2m", "d2m", "relative_humidity", "precip_m", "soil_moisture"]]
    df["Region"] = region_name
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    
    # Resample to quarterly data
    climate_q = pd.DataFrame({
        "avg_temp_C": df["t2m"].resample("Q").mean(),
        "avg_dewpoint_C": df["d2m"].resample("Q").mean(),
        "avg_rel_humidity_pct": df["relative_humidity"].resample("Q").mean(),
        "total_precip_m": df["precip_m"].resample("Q").sum(),
        "avg_soil_moisture": df["soil_moisture"].resample("Q").mean()
    })
    
    # Add year and quarter columns
    climate_q["Year"] = climate_q.index.year
    climate_q["Quarter"] = climate_q.index.quarter
    climate_q.reset_index(drop=True, inplace=True)
    climate_q["Region"] = region_name
    
    return climate_q

def get_coffee_climate_data(download=True, years=None, output_csv=None):
    """
    Main function to download and process climate data for coffee regions.
    
    Parameters:
    -----------
    download : bool
        Whether to download new data or use existing files
    years : list, optional
        List of years to download as strings. Default is 2020-2022
    output_csv : str, optional
        Path to save the CSV output. Default is "coffee_climate_data.csv"
        
    Returns:
    --------
    pd.DataFrame
        Processed climate data for all regions
    """
    if years is None:
        years = [str(year) for year in range(2020, 2023)]
    
    if output_csv is None:
        output_csv = "coffee_climate_data.csv"
    
    # Define coffee region bounding boxes [North, West, South, East]
    coffee_region_bounds = {
        "MINAS GERAIS (BRA)": [-18.0, -48.0, -22.0, -44.0],  # Coffee areas in Minas Gerais
        "COLOMBIA": [7.0, -77.0, 3.0, -73.0]                # Colombian coffee region (Andes)
    }
    
    coffee_climate_dfs = {}
    
    for region, bbox in coffee_region_bounds.items():
        nc_file = f"{region.replace(' ', '_')}_era5land_data.nc"
        
        # Download data if requested or if file doesn't exist
        if download or not os.path.exists(nc_file):
            nc_file = download_era5_data(region, bbox, years, nc_file)
        
        # Process the data
        climate_q = process_climate_data(nc_file, region)
        
        if climate_q is not None:
            coffee_climate_dfs[region] = climate_q
    
    # Combine all regions' data
    if coffee_climate_dfs:
        coffee_climate = pd.concat(list(coffee_climate_dfs.values()), ignore_index=True)
        
        # Save to CSV
        coffee_climate.to_csv(output_csv, index=False)
        print(f"Combined data saved to {output_csv}")
        
        return coffee_climate
    else:
        print("No data could be processed.")
        return None

def plot_climate_comparison(climate_df, variable, title=None):
    """
    Plot a comparison of climate variables between regions.
    
    Parameters:
    -----------
    climate_df : pd.DataFrame
        Processed climate data
    variable : str
        Variable to plot (column name)
    title : str, optional
        Plot title
    """
    if title is None:
        title = f"{variable} Comparison Between Coffee Regions"
    
    # Prepare plot data
    regions = climate_df['Region'].unique()
    
    plt.figure(figsize=(12, 6))
    
    for region in regions:
        region_data = climate_df[climate_df['Region'] == region]
        # Create a date-like x-axis
        x = [f"{year}-Q{quarter}" for year, quarter in 
             zip(region_data['Year'], region_data['Quarter'])]
        plt.plot(x, region_data[variable], marker='o', label=region)
    
    plt.title(title)
    plt.xlabel("Time Period")
    plt.ylabel(variable)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return plt

# Example usage
if __name__ == "__main__":
    # Get data for 2020-2022
    climate_data = get_coffee_climate_data(
        download=True,
        years=[str(year) for year in range(2020, 2023)],
        output_csv="coffee_climate_data.csv"
    )
    
    if climate_data is not None:
        # Plot temperature comparison
        plt = plot_climate_comparison(climate_data, 'avg_temp_C')
        plt.savefig('temperature_comparison.png')
        
        # Plot precipitation comparison
        plt = plot_climate_comparison(climate_data, 'total_precip_m')
        plt.savefig('precipitation_comparison.png')
        
        print("Plots saved to temperature_comparison.png and precipitation_comparison.png")