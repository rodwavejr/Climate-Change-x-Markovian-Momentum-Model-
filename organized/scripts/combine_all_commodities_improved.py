#!/usr/bin/env python3
"""
Combine All Commodity Data (Improved Version)

This script creates multiple formats of combined commodity-climate data 
for more flexible analysis.

Usage:
    python combine_all_commodities_improved.py
"""

import os
import pandas as pd
import glob
import numpy as np
from commodity_regions import COMMODITY_NAMES

def main():
    # Find all commodity climate joined files
    commodity_files = glob.glob("*_climate_joined.csv")
    
    if not commodity_files:
        print("No commodity climate data files found.")
        return
    
    print(f"Found {len(commodity_files)} commodity files:")
    for file in commodity_files:
        print(f"  - {file}")
    
    # Load each file into a dictionary by commodity
    commodity_dfs = {}
    commodities = []
    
    for file in commodity_files:
        # Extract commodity name from the filename
        commodity = file.split('_')[0].capitalize()
        commodities.append(commodity)
        
        print(f"Processing {commodity} data...")
        
        # Load the data
        try:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Make sure we have a price column
            price_col = f"{commodity}_Price"
            if price_col not in df.columns:
                # Rename any price column to match expected format
                price_cols = [col for col in df.columns if "Price" in col]
                if price_cols:
                    df.rename(columns={price_cols[0]: price_col}, inplace=True)
                else:
                    print(f"  Warning: No price column found in {file}")
                    continue
            
            # Store the DataFrame
            commodity_dfs[commodity] = df
            print(f"  Added {len(df)} records")
            
        except Exception as e:
            print(f"  Error processing {file}: {e}")
    
    if not commodity_dfs:
        print("No valid commodity data files were processed.")
        return
    
    print("\nCreating combined formats...")
    
    # Format 1: All price data for easy comparison
    # This is simple and clean, containing only dates and prices
    price_df = create_price_only_format(commodity_dfs, commodities)
    
    # Format 2: Climate data in long format with commodity tags
    # Each row is a date-region-commodity combination with all climate variables
    climate_long_df = create_climate_long_format(commodity_dfs, commodities)
    
    # Format 3: Ultra-wide format with all data (for wide-format analysis)
    # Each row is a date with all price and climate data
    wide_df = create_wide_format(commodity_dfs, commodities)
    
    # Format 4: Complete merged dataset with consistent organization
    # Each row is a date-commodity combination with multiple regions per month
    merged_df = create_merged_format(commodity_dfs, commodities)
    
    # Save all formats
    price_df.to_csv("price_data.csv", index=False)
    print(f"Saved price-only data to price_data.csv")
    
    climate_long_df.to_csv("climate_data_long.csv", index=False)
    print(f"Saved long-format climate data to climate_data_long.csv")
    
    wide_df.to_csv("all_data_wide.csv", index=False)
    print(f"Saved wide-format data to all_data_wide.csv")
    
    merged_df.to_csv("complete_commodity_climate.csv", index=False)
    print(f"Saved complete merged data to complete_commodity_climate.csv")
    
    print("\nCombining complete!")

def create_price_only_format(commodity_dfs, commodities):
    """Create a simple format with just dates and prices."""
    # Find the date range for all data
    all_dates = set()
    for df in commodity_dfs.values():
        all_dates.update(df['Date'].dt.date)
    
    # Create a template DataFrame with all dates
    date_range = pd.DataFrame({
        'Date': sorted(all_dates)
    })
    date_range['Date'] = pd.to_datetime(date_range['Date'])
    date_range['Year'] = date_range['Date'].dt.year
    date_range['Month'] = date_range['Date'].dt.month
    
    # Add price columns for each commodity
    for commodity in commodities:
        if commodity in commodity_dfs:
            df = commodity_dfs[commodity]
            price_col = f"{commodity}_Price"
            
            # Create a temporary DataFrame with date and price
            temp_df = df[['Date', price_col]].copy()
            
            # Merge with the template
            date_range = pd.merge(
                date_range, temp_df, 
                on='Date', how='left'
            )
    
    return date_range

def create_climate_long_format(commodity_dfs, commodities):
    """Create a long-format dataset with climate data."""
    all_climate_data = []
    
    for commodity in commodities:
        if commodity in commodity_dfs:
            df = commodity_dfs[commodity]
            
            # Identify climate variables (excluding date and price columns)
            date_cols = ['Date', 'Year', 'Month']
            price_col = f"{commodity}_Price"
            
            climate_cols = [col for col in df.columns 
                          if col not in date_cols and 
                          col != price_col and
                          col != 'Quarter' and
                          col != 'Quarter_climate' and
                          col != 'Quarter_commodity']
            
            # Create a subset with climate data
            climate_df = df[date_cols + climate_cols].copy()
            
            # Add commodity column if not present
            if 'Commodity' not in climate_df.columns:
                climate_df['Commodity'] = commodity
            
            # Standardize region column
            if 'Region' in climate_df.columns:
                # Keep as is
                pass
            elif any(col.endswith('_Region') for col in climate_df.columns):
                # Find and rename region column
                region_col = next(col for col in climate_df.columns if col.endswith('_Region'))
                climate_df.rename(columns={region_col: 'Region'}, inplace=True)
            
            # Add to the list
            all_climate_data.append(climate_df)
    
    # Combine all climate data
    if all_climate_data:
        return pd.concat(all_climate_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_wide_format(commodity_dfs, commodities):
    """Create a wide format with all data for each date."""
    # Start with the date range
    price_df = create_price_only_format(commodity_dfs, commodities)
    
    # Add climate data for each commodity
    for commodity in commodities:
        if commodity in commodity_dfs:
            df = commodity_dfs[commodity]
            
            # Identify climate variables
            date_cols = ['Date', 'Year', 'Month']
            price_col = f"{commodity}_Price"
            
            climate_cols = [col for col in df.columns 
                          if col not in date_cols and 
                          col != price_col and
                          col != 'Quarter' and
                          col != 'Quarter_climate' and
                          col != 'Quarter_commodity']
            
            # Rename climate columns to include commodity name
            rename_dict = {col: f"{commodity}_{col}" for col in climate_cols 
                        if not col.startswith(f"{commodity}_")}
            
            # Create a subset with renamed climate data
            climate_df = df[date_cols + climate_cols].copy()
            climate_df.rename(columns=rename_dict, inplace=True)
            
            # Get the new column names
            new_climate_cols = [rename_dict.get(col, col) for col in climate_cols]
            
            # Merge with the price DataFrame
            price_df = pd.merge(
                price_df, 
                climate_df[date_cols + new_climate_cols], 
                on=date_cols, how='left'
            )
    
    return price_df

def create_merged_format(commodity_dfs, commodities):
    """Create a complete merged dataset with all commodities."""
    # Start with the date range
    date_df = create_price_only_format(commodity_dfs, commodities)
    
    # Prepare to add commodity as a dimension
    records = []
    
    # For each date, create multiple records (one per commodity)
    for _, date_row in date_df.iterrows():
        date = date_row['Date']
        year = date_row['Year']
        month = date_row['Month']
        
        # Get price values for all commodities at this date
        price_values = {}
        for commodity in commodities:
            price_col = f"{commodity}_Price"
            if price_col in date_row and not pd.isna(date_row[price_col]):
                price_values[commodity] = date_row[price_col]
        
        # Create a record for each commodity
        for commodity in price_values.keys():
            if commodity in commodity_dfs:
                # Get the corresponding data row for this date
                df = commodity_dfs[commodity]
                matching_rows = df[df['Date'] == date]
                
                if len(matching_rows) > 0:
                    data_row = matching_rows.iloc[0].to_dict()
                    
                    # Create a clean record
                    record = {
                        'Date': date,
                        'Year': year,
                        'Month': month,
                        'Commodity': commodity,
                        f'{commodity}_Price': price_values[commodity]
                    }
                    
                    # Add climate data
                    for col, value in data_row.items():
                        if col not in ['Date', 'Year', 'Month', 'Commodity', f'{commodity}_Price']:
                            # Keep region columns as is
                            if col == 'Region':
                                record['Region'] = value
                            elif col.endswith('_Region'):
                                record[col] = value
                            # Add all climate variables
                            elif col not in ['Quarter', 'Quarter_climate', 'Quarter_commodity']:
                                record[col] = value
                    
                    records.append(record)
    
    # Convert to DataFrame
    if records:
        merged_df = pd.DataFrame(records)
        # Sort by date and commodity
        merged_df.sort_values(['Date', 'Commodity'], inplace=True)
        return merged_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    main()