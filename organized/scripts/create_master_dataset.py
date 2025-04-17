#!/usr/bin/env python3
"""
Create Master Climate-Commodity Dataset

This script creates a single master CSV file containing all commodity prices
and climate variables in a clean, organized format.

Usage:
    python create_master_dataset.py
"""

import pandas as pd
import os
import glob

def main():
    """Create a master dataset with all commodity prices and climate data."""
    print("Creating master climate-commodity dataset...")
    
    # Check if the all_data_wide.csv file exists
    if os.path.exists("all_data_wide.csv"):
        # Load the wide-format data
        wide_df = pd.read_csv("all_data_wide.csv")
        wide_df['Date'] = pd.to_datetime(wide_df['Date'])
        
        print(f"Loaded wide data with {wide_df.shape[0]} rows and {wide_df.shape[1]} columns")
        
        # Create a clean master dataset
        master_df = create_clean_master_dataset(wide_df)
        
        # Save the master dataset
        master_file = "MASTER_climate_commodity_data.csv"
        master_df.to_csv(master_file, index=False)
        print(f"Saved master dataset to {master_file}")
        
        # Display column information
        print(f"\nMaster dataset has {master_df.shape[0]} rows and {master_df.shape[1]} columns")
        
        # Group columns by category
        date_cols = ['Date', 'Year', 'Month']
        price_cols = [col for col in master_df.columns if col.endswith('_Price')]
        region_cols = [col for col in master_df.columns if col.endswith('_Region')]
        temp_cols = [col for col in master_df.columns if 'temperature' in col or 'temp' in col]
        precip_cols = [col for col in master_df.columns if 'precip' in col]
        other_climate_cols = [col for col in master_df.columns 
                           if col not in date_cols + price_cols + region_cols + temp_cols + precip_cols]
        
        print(f"\nColumn categories:")
        print(f"  Date columns: {len(date_cols)}")
        print(f"  Price columns: {len(price_cols)}")
        print(f"  Region columns: {len(region_cols)}")
        print(f"  Temperature columns: {len(temp_cols)}")
        print(f"  Precipitation columns: {len(precip_cols)}")
        print(f"  Other climate columns: {len(other_climate_cols)}")
        
        # Print the commodities included
        commodities = [col.split('_')[0] for col in price_cols]
        print(f"\nCommodities included: {', '.join(commodities)}")
        
        # Print the date range
        date_range = f"{master_df['Date'].min().strftime('%Y-%m-%d')} to {master_df['Date'].max().strftime('%Y-%m-%d')}"
        print(f"\nDate range: {date_range}")
        
    else:
        print("Error: all_data_wide.csv not found. Please run combine_all_commodities_improved.py first.")

def create_clean_master_dataset(wide_df):
    """
    Create a clean master dataset from the wide-format data.
    
    Parameters:
    -----------
    wide_df : pd.DataFrame
        Wide-format data with all commodity prices and climate variables
        
    Returns:
    --------
    pd.DataFrame
        Clean master dataset
    """
    # Start with the basic date columns
    master_df = wide_df[['Date', 'Year', 'Month']].copy()
    
    # Find commodity price columns
    price_cols = [col for col in wide_df.columns if col.endswith('_Price')]
    master_df[price_cols] = wide_df[price_cols]
    
    # Find region columns
    region_cols = [col for col in wide_df.columns if col.endswith('_Region')]
    for col in region_cols:
        master_df[col] = wide_df[col]
    
    # Get list of commodities
    commodities = [col.split('_')[0] for col in price_cols]
    
    # Process climate variables for each commodity
    for commodity in commodities:
        # Find climate columns for this commodity
        commodity_cols = [col for col in wide_df.columns 
                       if col.startswith(f"{commodity}_") and 
                       not col.endswith('_Price') and 
                       not col.endswith('_Region') and
                       not col.endswith('_Commodity')]
        
        # Add each climate variable with a clean name
        for col in commodity_cols:
            # Extract the variable name from the column name
            var_name = col.replace(f"{commodity}_", "")
            # Create a new column with a cleaner name
            new_col = f"{commodity}_{var_name}"
            master_df[new_col] = wide_df[col]
    
    return master_df

if __name__ == "__main__":
    main()