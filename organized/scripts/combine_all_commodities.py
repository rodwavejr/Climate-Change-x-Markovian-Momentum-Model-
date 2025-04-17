#!/usr/bin/env python3
"""
Combine All Commodity Data

This script combines all individual commodity climate data files into a single
comprehensive dataset for unified analysis.

Usage:
    python combine_all_commodities.py
"""

import os
import pandas as pd
import glob
import numpy as np
from commodity_regions import COMMODITY_NAMES

def main():
    """
    Main function to combine all commodity data into one CSV file.
    """
    print("Combining all commodity climate data...")
    
    # Find all commodity climate joined files
    commodity_files = glob.glob("*_climate_joined.csv")
    
    if not commodity_files:
        print("No commodity climate data files found.")
        return
    
    print(f"Found {len(commodity_files)} commodity files:")
    for file in commodity_files:
        print(f"  - {file}")
    
    # Process each file and store in a list
    all_dfs = []
    
    # Track column metadata to help with reorganization
    climate_columns = set()
    price_columns = set()
    common_columns = set()
    
    for file in commodity_files:
        # Extract commodity name from the filename
        commodity = file.split('_')[0].capitalize()
        print(f"Processing {commodity} data...")
        
        # Load the data
        try:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Make sure we have a price column
            price_col = f"{commodity}_Price"
            if price_col not in df.columns:
                print(f"  Warning: No price column '{price_col}' found in {file}")
                continue
                
            # Identify columns that are specific to this commodity
            if len(all_dfs) == 0:
                # First file, establish baseline columns
                climate_columns = {col for col in df.columns 
                                 if col not in ['Date', 'Year', 'Month', price_col]}
                price_columns = {price_col}
                common_columns = {'Date', 'Year', 'Month'}
            else:
                # For subsequent files, update column sets
                climate_cols_in_file = {col for col in df.columns 
                                      if col not in ['Date', 'Year', 'Month'] and
                                      not col.endswith('_Price')}
                climate_columns.update(climate_cols_in_file)
                price_columns.add(price_col)
            
            # Add commodity information to this dataframe
            if 'Commodity' not in df.columns:
                df['Commodity'] = commodity
                
            # Add region information in a standardized way
            if 'Region' in df.columns:
                df.rename(columns={'Region': f'{commodity}_Region'}, inplace=True)
                
            # Store the dataframe
            all_dfs.append(df)
            print(f"  Added {len(df)} records")
            
        except Exception as e:
            print(f"  Error processing {file}: {e}")
    
    if not all_dfs:
        print("No valid commodity data files were processed.")
        return
        
    print("\nMerging all data...")
    
    # Method 1: Merge on Date, Year, Month
    # This will keep climate data from each region separate
    merged_df = all_dfs[0]
    
    for df in all_dfs[1:]:
        # For each dataframe after the first, merge on common columns
        merged_df = pd.merge(
            merged_df, 
            df, 
            on=['Date', 'Year', 'Month'], 
            how='outer',
            suffixes=('', '_duplicate')
        )
        
        # Remove any duplicate columns
        duplicate_cols = [col for col in merged_df.columns if col.endswith('_duplicate')]
        if duplicate_cols:
            merged_df.drop(columns=duplicate_cols, inplace=True)
    
    # Sort by date
    merged_df.sort_values('Date', inplace=True)
    
    # Method 2: Pivot the data for a cleaner structure
    # Extract just the price columns and date information
    price_cols = [col for col in merged_df.columns if col.endswith('_Price')]
    date_cols = ['Date', 'Year', 'Month']
    
    # Create a clean price dataset
    price_df = merged_df[date_cols + price_cols].copy()
    
    # Now create an organized climate dataset
    # Extract region columns
    region_cols = [col for col in merged_df.columns if col.endswith('_Region')]
    region_df = merged_df[date_cols + region_cols].copy()
    
    # Create a long-form climate dataset
    climate_data = []
    commodities = [col.split('_')[0] for col in price_cols]
    
    for commodity in commodities:
        # Find the region column for this commodity
        region_col = f"{commodity}_Region"
        
        # Find climate columns for this region
        climate_cols = [col for col in merged_df.columns 
                      if col not in date_cols + price_cols + region_cols and
                      not col.endswith('_Region') and
                      not col.endswith('_Price') and
                      col != 'Commodity']
                      
        if 'Commodity' in merged_df.columns:
            commodity_data = merged_df[merged_df['Commodity'] == commodity]
        else:
            # If we don't have a commodity column, assume the first region matches
            commodity_data = merged_df
            
        # Extract climate data with commodity and region
        if len(commodity_data) > 0:
            for _, row in commodity_data.iterrows():
                data_row = {
                    'Date': row['Date'],
                    'Year': row['Year'],
                    'Month': row['Month'],
                    'Commodity': commodity
                }
                
                # Add region if available
                if region_col in row and not pd.isna(row[region_col]):
                    data_row['Region'] = row[region_col]
                
                # Add climate variables
                for col in climate_cols:
                    if col in row and not pd.isna(row[col]):
                        data_row[col] = row[col]
                
                climate_data.append(data_row)
    
    # Convert to DataFrame
    if climate_data:
        climate_long_df = pd.DataFrame(climate_data)
        climate_long_df.sort_values(['Date', 'Commodity'], inplace=True)
    else:
        climate_long_df = pd.DataFrame()
        print("Warning: Could not create long-form climate data")
    
    # Save all datasets
    output_file = "all_commodities_combined.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Saved combined commodity data to {output_file}")
    
    # Save just the prices in a clean format
    price_file = "all_commodity_prices.csv"
    price_df.to_csv(price_file, index=False)
    print(f"Saved price-only data to {price_file}")
    
    # Save the long-form climate data
    if not climate_long_df.empty:
        climate_file = "climate_data_long_form.csv"
        climate_long_df.to_csv(climate_file, index=False)
        print(f"Saved long-form climate data to {climate_file}")
    
    # Print column statistics
    print("\nDataset statistics:")
    print(f"  Total records: {len(merged_df)}")
    print(f"  Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"  Commodities: {', '.join(commodities)}")
    print(f"  Total columns: {len(merged_df.columns)}")
    
    # Print column categories
    price_cols = [col for col in merged_df.columns if col.endswith('_Price')]
    region_cols = [col for col in merged_df.columns if col.endswith('_Region')]
    climate_cols = [col for col in merged_df.columns 
                  if col not in date_cols + price_cols + region_cols and 
                  not col.endswith('_Region') and
                  not col.endswith('_Price') and
                  col != 'Commodity']
    
    print(f"  Price columns: {len(price_cols)}")
    print(f"  Region columns: {len(region_cols)}")
    print(f"  Climate variables: {len(climate_cols)}")
    
    print("\nCombining complete!")
    
if __name__ == "__main__":
    main()