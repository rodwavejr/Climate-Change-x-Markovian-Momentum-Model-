#!/usr/bin/env python3
"""
Commodity Growing Regions

This script defines the geographic regions where different commodities are grown.
It maps commodity codes to their names and growing regions.

Usage:
    Import this file to get access to the COMMODITY_REGIONS dictionary.
"""

# Commodity names mapping
COMMODITY_NAMES = {
    "PCOFFOTMUSDM": "Coffee",
    "PCOCOUSDM": "Cocoa",
    "PMAIZMTUSDM": "Maize",
    "PWHEAMTUSDM": "Wheat",
    "PSOYBUSDM": "Soybeans",
    "PCOTTINDUSDM": "Cotton",
    "PRICENPQUSDM": "Rice"
}

# Define commodity growing regions with bounding boxes [North, West, South, East]
COMMODITY_REGIONS = {
    # Coffee growing regions (already defined)
    "Coffee": {
        "COLOMBIA": [7.0, -77.0, 3.0, -73.0],              # Colombian coffee region
        "MINAS GERAIS (BRA)": [-18.0, -48.0, -22.0, -44.0], # Brazil - Minas Gerais
        # Additional coffee regions (not yet used)
        "VIETNAM": [16.0, 105.0, 10.0, 108.0],             # Vietnam Central Highlands
        "ETHIOPIA": [9.0, 35.0, 5.0, 39.0]                 # Ethiopian coffee regions
    },
    
    # Cocoa growing regions
    "Cocoa": {
        "GHANA": [8.0, -3.0, 5.0, 1.0],                   # Ghana cocoa region
        "IVORY COAST": [8.0, -8.0, 5.0, -4.0],            # Ivory Coast cocoa region
        "INDONESIA": [0.0, 116.0, -8.0, 119.0]            # Indonesia (Sulawesi)
    },
    
    # Maize (Corn) growing regions
    "Maize": {
        "US CORN BELT": [45.0, -95.0, 37.0, -85.0],        # US Corn Belt
        "MATO GROSSO (BRA)": [-12.0, -58.0, -17.0, -53.0], # Brazil - Mato Grosso
        "ARGENTINA": [-30.0, -65.0, -35.0, -60.0]          # Argentina corn region
    },
    
    # Wheat growing regions
    "Wheat": {
        "US GREAT PLAINS": [48.0, -105.0, 35.0, -95.0],    # US Great Plains
        "UKRAINE": [52.0, 25.0, 45.0, 40.0],               # Ukraine wheat region
        "AUSTRALIA": [-30.0, 135.0, -35.0, 145.0]          # Australia wheat belt
    },
    
    # Soybean growing regions
    "Soybeans": {
        "US MIDWEST": [45.0, -95.0, 37.0, -85.0],          # US Midwest
        "BRAZIL SOY BELT": [-20.0, -55.0, -25.0, -45.0],   # Brazil soy belt
        "ARGENTINA SOY": [-30.0, -65.0, -35.0, -60.0]      # Argentina soy region
    },
    
    # Cotton growing regions
    "Cotton": {
        "US COTTON BELT": [35.0, -100.0, 30.0, -90.0],     # US Cotton Belt
        "INDIA COTTON": [22.0, 70.0, 17.0, 78.0],          # India cotton region
        "CHINA COTTON": [42.0, 80.0, 38.0, 85.0]           # China (Xinjiang) cotton
    },
    
    # Rice growing regions
    "Rice": {
        "SOUTHEAST ASIA": [20.0, 95.0, 10.0, 105.0],       # Southeast Asia rice
        "INDIA RICE": [22.0, 75.0, 17.0, 85.0],            # India rice region
        "CHINA RICE": [30.0, 110.0, 20.0, 120.0]           # China rice region
    }
}

# Primary regions for each commodity (the main growing region we'll use first)
PRIMARY_REGIONS = {
    "Coffee": "COLOMBIA",
    "Cocoa": "GHANA",
    "Maize": "US CORN BELT",
    "Wheat": "US GREAT PLAINS",
    "Soybeans": "BRAZIL SOY BELT",
    "Cotton": "INDIA COTTON",
    "Rice": "SOUTHEAST ASIA"
}

# For backward compatibility - coffee regions in the old format
COFFEE_REGION_BOUNDS = {
    "COLOMBIA": [7.0, -77.0, 3.0, -73.0],              
    "MINAS GERAIS (BRA)": [-18.0, -48.0, -22.0, -44.0], 
    "VIETNAM": [16.0, 105.0, 10.0, 108.0],             
    "ETHIOPIA": [9.0, 35.0, 5.0, 39.0],                
    "INDONESIA": [0.0, 116.0, -8.0, 119.0],            
    "COSTA RICA": [11.0, -85.5, 8.0, -82.5],           
    "GUATEMALA": [16.0, -92.0, 13.5, -88.0],           
    "KENYA": [1.0, 36.0, -2.0, 39.0],                  
    "INDIA": [13.0, 75.0, 10.0, 78.0],                 
    "HONDURAS": [15.5, -89.0, 13.0, -86.0]             
}