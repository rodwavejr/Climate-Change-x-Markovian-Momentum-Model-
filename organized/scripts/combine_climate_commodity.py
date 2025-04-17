#!/usr/bin/env python3
"""
Combine Climate and Commodity Data

This script:
1. Loads the existing extended climate data for 2015-2022
2. Loads the commodity data
3. Resamples and joins the datasets
4. Calculates climate signatures
5. Produces a combined dataset for analysis

Usage:
    python combine_climate_commodity.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_climate_data():
    """
    Load the extended climate data (quarterly, 2015-2022).
    
    Returns:
    --------
    pd.DataFrame or None
        The loaded climate data, or None if loading failed
    """
    climate_file = "coffee_climate_extended.csv"
    print(f"Loading climate data from {climate_file}...")
    
    if not os.path.exists(climate_file):
        print(f"Error: Climate data file not found at {climate_file}")
        return None
    
    try:
        # Load the data
        climate_df = pd.read_csv(climate_file)
        
        # Create date column from year and quarter
        climate_df['Month'] = climate_df['Quarter'] * 3 - 2  # Q1->1, Q2->4, Q3->7, Q4->10
        climate_df['Day'] = 15  # Middle of the month
        climate_df['Date'] = pd.to_datetime(
            climate_df[['Year', 'Month', 'Day']])
        
        # Drop the day column as it's not needed
        climate_df.drop(columns=['Day'], inplace=True)
        
        print(f"Loaded climate data with {len(climate_df)} records from " 
              f"{climate_df['Year'].min()} to {climate_df['Year'].max()}")
        return climate_df
    
    except Exception as e:
        print(f"Error loading climate data: {e}")
        return None


def load_commodity_data():
    """
    Load the commodity price data.
    
    Returns:
    --------
    pd.DataFrame or None
        The loaded commodity data, or None if loading failed
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
        df['Quarter'] = df['Month'].apply(lambda x: (x-1)//3 + 1)
        
        # Select relevant columns
        # PCOFFOTMUSDM is the coffee price
        columns = ["Date", "Year", "Month", "Quarter", "PCOFFOTMUSDM"]
        
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
        
        print(f"Loaded commodity data with {len(commodity_df)} records from "
              f"{commodity_df['Year'].min()} to {commodity_df['Year'].max()}")
        return commodity_df
        
    except Exception as e:
        print(f"Error loading commodity data: {e}")
        return None


def expand_quarterly_to_monthly(climate_df):
    """
    Expand quarterly climate data to monthly data using interpolation.
    
    Parameters:
    -----------
    climate_df : pd.DataFrame
        Quarterly climate data
        
    Returns:
    --------
    pd.DataFrame
        Monthly climate data
    """
    print("Expanding quarterly climate data to monthly...")
    
    # Create a copy to avoid modifying the original
    df = climate_df.copy()
    
    # Group by region
    regions = df["Region"].unique()
    region_dfs = []
    
    for region in regions:
        region_df = df[df["Region"] == region].copy()
        
        # Create a complete date range for all months
        start_date = pd.Timestamp(year=region_df["Year"].min(), month=1, day=15)
        end_date = pd.Timestamp(year=region_df["Year"].max(), month=12, day=15)
        all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Create a template dataframe with all months
        monthly_template = pd.DataFrame({'Date': all_months})
        monthly_template['Year'] = monthly_template['Date'].dt.year
        monthly_template['Month'] = monthly_template['Date'].dt.month
        monthly_template['Quarter'] = monthly_template['Month'].apply(lambda x: (x-1)//3 + 1)
        monthly_template['Region'] = region
        
        # Merge with the quarterly data
        merged = pd.merge(
            monthly_template, 
            region_df, 
            on=['Year', 'Quarter', 'Region'],
            how='left',
            suffixes=('', '_q')
        )
        
        # Use the date from the monthly template
        merged['Date'] = merged['Date']
        merged.drop(columns=['Date_q', 'Month_q'], inplace=True, errors='ignore')
        
        # Sort by date for interpolation
        merged = merged.sort_values('Date')
        
        # Interpolate missing values
        climate_cols = ['avg_temp_C', 'total_precip_m', 'avg_dewpoint_C', 'avg_rel_humidity_pct']
        for col in climate_cols:
            if col in merged.columns:
                # Use linear interpolation for most variables
                if col != 'total_precip_m':
                    merged[col] = merged[col].interpolate(method='linear')
                else:
                    # For precipitation, distribute quarterly values evenly across months
                    quarterly_values = merged.dropna(subset=[col])[['Year', 'Quarter', col]]
                    for _, row in quarterly_values.iterrows():
                        year, quarter, value = row['Year'], row['Quarter'], row[col]
                        # Determine which months are in this quarter
                        quarter_months = merged[
                            (merged['Year'] == year) & 
                            (merged['Quarter'] == quarter)
                        ].index
                        # Distribute precipitation evenly
                        if len(quarter_months) > 0:
                            merged.loc[quarter_months, col] = value / len(quarter_months)
        
        region_dfs.append(merged)
    
    # Combine all regions
    monthly_df = pd.concat(region_dfs, ignore_index=True)
    print(f"Expanded to {len(monthly_df)} monthly records")
    
    return monthly_df


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
        region_df["temp_anomaly"] = region_df.groupby("Month")["avg_temp_C"].transform(
            lambda x: x - x.mean())
        
        # Calculate precipitation anomalies
        region_df["precip_anomaly"] = region_df.groupby("Month")["total_precip_m"].transform(
            lambda x: x - x.mean())
        
        # Calculate rolling average temperature (3-month window)
        region_df["temp_3m_avg"] = region_df["avg_temp_C"].rolling(window=3, min_periods=1).mean()
        
        # Calculate cumulative precipitation (3-month window)
        region_df["precip_3m_sum"] = region_df["total_precip_m"].rolling(window=3, min_periods=1).sum()
        
        # Add drought index (simplistic)
        # Negative values indicate drought conditions
        region_df["drought_index"] = region_df["precip_anomaly"] - 0.1 * region_df["temp_anomaly"]
        
        # If humidity is available, calculate heat stress index
        if "avg_rel_humidity_pct" in region_df.columns:
            # Heat index calculation (simplified)
            region_df["heat_stress"] = region_df["avg_temp_C"] + 0.05 * region_df["avg_rel_humidity_pct"]
        
        signature_dfs.append(region_df)
    
    # Combine all regions back
    signatures_df = pd.concat(signature_dfs, ignore_index=True)
    
    print(f"Added climate signatures to data")
    return signatures_df


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


def main():
    # Step 1: Load climate data (quarterly)
    climate_data = load_climate_data()
    if climate_data is None:
        print("Cannot proceed without climate data.")
        return
    
    # Step 2: Load commodity data (monthly)
    commodity_data = load_commodity_data()
    if commodity_data is None:
        print("Cannot proceed without commodity data.")
        return
    
    # Step 3: Expand climate data from quarterly to monthly
    monthly_climate = expand_quarterly_to_monthly(climate_data)
    
    # Step 4: Calculate climate signatures
    climate_with_signatures = calculate_climate_signatures(monthly_climate)
    
    # Step 5: Join climate and commodity data
    joined_data = join_climate_commodity_data(climate_with_signatures, commodity_data)
    
    # Step 6: Save the combined dataset
    output_file = "climate_commodity_joined.csv"
    joined_data.to_csv(output_file, index=False)
    print(f"Saved joined climate and commodity data to {output_file}")
    
    # Display summary
    print("\nData summary:")
    print(f"Climate data: {len(climate_with_signatures)} records from {climate_with_signatures['Year'].min()} to {climate_with_signatures['Year'].max()}")
    print(f"Commodity data: {len(commodity_data)} records from {commodity_data['Year'].min()} to {commodity_data['Year'].max()}")
    print(f"Joined data: {len(joined_data)} records from {joined_data['Year'].min()} to {joined_data['Year'].max()}")
    
    # Report on regions
    print("\nRegions covered in combined dataset:")
    for region, count in joined_data.groupby('Region').size().items():
        print(f"- {region}: {count} records")


if __name__ == "__main__":
    main()