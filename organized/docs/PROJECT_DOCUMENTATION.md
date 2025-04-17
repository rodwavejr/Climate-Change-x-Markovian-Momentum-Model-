# Climate and Commodity Analysis: Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Dataset Structure and Processing](#dataset-structure-and-processing)
4. [Processing Pipeline](#processing-pipeline)
5. [Analysis Methods](#analysis-methods)
6. [Results and Findings](#results-and-findings)
7. [Current Challenge](#current-challenge)
8. [Proposed Solution: Markovian Momentum Model](#proposed-solution-markovian-momentum-model)
9. [Next Steps](#next-steps)

## Project Overview

This project investigates the relationship between climate variables and agricultural commodity prices. It integrates high-resolution climate data from growing regions with global price data for key agricultural commodities:

- Coffee
- Cocoa
- Wheat
- Maize
- Rice
- Soybeans
- Cotton

The analysis spans from 2015 to 2022, providing insights into how temperature, precipitation, and derived climate metrics correlate with commodity price movements. The project includes data collection, processing, visualization, and exploratory analysis, with the foundation for predictive modeling.

## Directory Structure

The project is organized into the following directory structure:

```
Climate_Project/
├── QUICK_START_GUIDE.md           # Entry point for project users
├── REQUIREMENTS.md                # Project dependencies
├── Machine_Learning/              # Python virtual environment
├── archive_20250405_123807/       # Original project files (pre-organization)
├── organized/                     # Current organized project structure
│   ├── data_files/                # Processed data files in CSV format
│   ├── docs/                      # Documentation files
│   ├── notebooks/                 # Analysis Jupyter notebooks
│   ├── plots/                     # Generated visualizations
│   ├── raw_data/                  # Raw climate and commodity data
│   └── scripts/                   # Processing and utility scripts
├── remove_original_files.sh       # Script to clean up original files
└── setup_analysis_env.sh          # Environment setup script
```

## Dataset Structure and Processing

### Data Sources

#### Climate Data
The project uses the ERA5-Land reanalysis dataset from the Copernicus Climate Data Store, which provides hourly climate data on a 0.1° × 0.1° grid. The data is downloaded as NetCDF (.nc) files for specific growing regions of each commodity.

**Primary Growing Regions:**
- Coffee: Colombia and Minas Gerais (Brazil)
- Cocoa: Ghana
- Wheat: US Great Plains
- Maize: US Corn Belt
- Rice: Southeast Asia
- Soybeans: Brazil Soy Belt
- Cotton: India Cotton

For each region, the following climate variables are extracted:
- Temperature (2m temperature in °C)
- Precipitation (in meters)
- Dewpoint temperature (°C)
- Relative humidity (%)

#### Commodity Price Data
Price data comes from global commodity exchanges with the following symbols:
- Coffee: PCOFFOTMUSDM (ICE Futures, c/lb)
- Cocoa: PCOCOUSDM (ICE Futures, $/MT)
- Wheat: PWHEAMTUSDM (CBOT Futures, c/bu)
- Maize: PMAIZMTUSDM (CBOT Futures, c/bu)
- Rice: PRICENPQUSDM (CBOT Rough Rice, $/cwt)
- Soybeans: PSOYBUSDM (CBOT Futures, c/bu)
- Cotton: PCOTTINDUSDM (ICE Cotton, c/lb)

### Dataset Structure

The master dataset (`MASTER_climate_commodity_data.csv`) has the following structure:

- 95 rows (monthly data from February 2015 to December 2022)
- 86 columns including:
  - Date information (`Year`, `Month`)
  - Commodity prices (7 columns: `Wheat_Price`, `Cocoa_Price`, etc.)
  - Region information (7 columns: `Wheat_Region`, `Cocoa_Region`, etc.)
  - Climate variables for each commodity (10+ columns per commodity)

**Example of columns for one commodity (Coffee):**
```
Coffee_Price, Coffee_Region, Coffee_temperature_C, Coffee_precip_m, 
Coffee_dewpoint_C, Coffee_relative_humidity, Coffee_temp_anomaly, 
Coffee_precip_anomaly, Coffee_temp_3m_avg, Coffee_precip_3m_sum, 
Coffee_drought_index, Coffee_heat_stress
```

### Quarterly to Monthly Data Processing

A key aspect of the data processing pipeline is handling the temporal resolution:

1. **Raw Climate Data**: Originally downloaded at hourly resolution
2. **Quarterly Aggregation**: Data is first aggregated to quarterly values for initial processing
3. **Monthly Expansion**: Quarterly data is expanded to monthly resolution using interpolation techniques

The code responsible for this conversion is in `combine_climate_commodity.py`:

```python
def expand_quarterly_to_monthly(climate_df):
    """Expand quarterly climate data to monthly data using interpolation."""
    # For most variables, use linear interpolation
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
```

This approach ensures we have monthly data for analysis while preserving the integrity of the underlying climate patterns.

### Derived Climate Variables

The dataset includes several derived climate variables to capture meaningful patterns:

1. **Temperature and Precipitation Anomalies**: 
   - Deviations from monthly historical averages
   - Captures unusual weather patterns

2. **Moving Averages and Sums**:
   - 3-month moving average for temperature (`temp_3m_avg`)
   - 3-month cumulative precipitation (`precip_3m_sum`)
   - Captures longer-term climate trends

3. **Composite Indices**:
   - Drought index: Combination of precipitation deficit and temperature excess
   - Heat stress index: Combined effect of temperature and humidity
   - Represents complex climate conditions affecting crops

These derived variables are calculated in the `calculate_climate_signatures` function in `combine_climate_commodity.py` and similar scripts.

## Processing Pipeline

The data processing pipeline follows these steps:

1. **Raw Data Acquisition**
   - Download climate data from Copernicus Climate Data Store as NetCDF files
   - Collect commodity price data
   - Store raw data in `/organized/raw_data/`

2. **Climate Data Processing**
   - Extract variables for growing regions using `climate_data_loader.py`
   - Calculate quarterly aggregations using `climate_data_simple.py`
   - Create derived variables with `coffee_climate_monthly.py`
   - Generate initial climate datasets

3. **Quarterly to Monthly Expansion**
   - Expand quarterly data to monthly resolution using interpolation
   - Implemented in `combine_climate_commodity.py`
   - Preserves climate patterns while providing finer temporal granularity

4. **Data Integration**
   - Merge climate and price data by date using `combine_climate_commodity.py`
   - Create individual commodity datasets (e.g., `coffee_climate_joined.csv`)
   - Handle alignment between different data sources

5. **Master Dataset Creation**
   - Combine all commodity-climate datasets using `create_master_dataset.py`
   - Standardize column naming
   - Create final `MASTER_climate_commodity_data.csv`

This pipeline is designed to be repeatable and extensible as new data becomes available.

## Analysis Methods

### Data Exploration

The analysis begins with basic exploration of the master dataset:
- Trends in commodity prices over time
- Visualization of climate variables by region
- Seasonal patterns in both prices and climate
- Statistical summaries of price behavior

### Correlation Analysis

A key component is analyzing correlations between climate variables and prices:
- Direct correlations between climate variables and commodity prices
- Identification of strongest climate-price relationships
- Visualization using correlation heatmaps

### Time-Lagged Analysis

The analysis includes time-lagged effects of climate on prices:
- Calculation of correlations with lags from 0 to 6 months
- Identification of optimal lag periods for each climate variable
- Rationale: Climate impacts on crop yields often manifest after delays
- Implementation in `calculate_time_lagged_correlations` function

For example, with coffee, temperature anomalies show the strongest correlation with prices at 0-month and 2-month lags (+0.25).

### Machine Learning Approach

The project uses Random Forest as the primary modeling approach:
- Features: Climate variables and their lagged versions (1, 2, and 3 months)
- Target: Commodity price
- Train/Test Split: Chronological split (80/20) to respect time series nature
- Feature Importance: Identification of most predictive climate variables

The train/test split is implemented using a chronological approach:
```python
# Split the data chronologically
split_idx = int(len(X) * (1 - test_size))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

### Feature Engineering

The analysis includes several feature engineering approaches:
- Lagged features: Adding 1, 2, and 3-month lags of all climate variables
- Seasonal features: Adding sine and cosine transformations of month
- Scaling: Standardizing all features using StandardScaler
- Drought and heat stress indices: Composite climate indicators

## Results and Findings

### Price Trends
- Coffee shows the most dramatic price increase during 2021-2022, nearly doubling from ~$150 to ~$280
- Different commodities exhibit distinct volatility patterns
- Clear seasonal patterns exist in many commodity prices

### Climate-Price Correlations
- Temperature anomalies show the strongest correlation with coffee prices (+0.25)
- Drought index shows a negative correlation with coffee prices (-0.25)
- Precipitation variables show mixed correlations across commodities

### Regional Climate Patterns
The plots in `/organized/plots/` demonstrate key climate differences between regions:
- `temperature_comparison.png`: Shows Colombia maintains relatively stable temperatures (20-25°C) while Minas Gerais (Brazil) shows more extreme seasonal variations (19-29°C)
- `precipitation_comparison.png`: Shows different rainfall patterns between regions, affecting growing conditions

### Time-Lagged Effects
- Climate impacts often show time-lagged effects on prices
- For coffee, temperature anomalies with a 1-month lag showed consistent correlation with prices
- This suggests a delay between climate events and market price adjustments

### Model Performance
The Random Forest model shows:
- Strong performance on training data (R² = 0.82, RMSE = 6.60)
- Poor performance on test data (R² = -13.45, RMSE = 100.68)
- The significant drop in performance coincides with the COVID-19 period (post-2020)

### Feature Importance
The most important features for coffee price prediction:
1. Temperature anomaly with 1-month lag
2. Drought index with 1-month lag
3. Drought index with 3-month lag
4. Precipitation anomaly
5. Drought index with 2-month lag

This emphasizes the importance of lagged climate effects in price prediction.

## Current Challenge

After extensive analysis, a significant challenge has emerged: **The relationship between climate variables and commodity prices breaks down during the COVID-19 period (post-2020).**

The actual vs. predicted price plot clearly shows this breakdown:
- The model performs reasonably well from 2015-2019
- From 2021 onward, actual prices soar while predictions remain relatively flat
- Nearly every commodity experienced price spikes not explained by climate variables

This can be seen in the actual vs. predicted plot where:
- Training period (blue): Good alignment between actual and predicted prices
- Test period (red): Massive divergence between actual prices (~$280) and predicted prices (~$150)

This breakdown coincides with global disruptions from the COVID-19 pandemic, supply chain issues, and geopolitical events. The challenge reveals a fundamental limitation in climate-driven price modeling: external shocks can override typical climate-driven patterns, and these disruptions are becoming more common in our increasingly complex global system.

This raises critical questions:
1. How can we build models that center on climate but don't fall apart during global disruptions?
2. Can we design systems that recognize when their assumptions no longer apply?
3. How do we integrate non-climate factors while maintaining focus on climate relationships?

## Proposed Solution: Markovian Momentum Model (MMM)

To address the breakdown in model performance during periods of external disruption—most notably seen during the COVID-19 shock—we propose a hybrid modeling framework called the Markovian Momentum Model (MMM). This approach is designed to retain the core relationship between climate variables and commodity pricing, while dynamically responding to structural shifts and anomalous behavior in the market.

### Concept

The Markovian Momentum Model treats the commodity pricing system as a non-stationary Markov process, where the current state S_t is determined not only by the prior state S_{t-1} and exogenous climate features, but also by a dynamic set of shock-sensitive indicators. These indicators capture unusual behavior in the system that suggests the emergence of a new pricing regime—such as one driven by supply chain disruptions, demand shocks, or macroeconomic instability.

The system evolves according to the following structure:

$P(S_t | S_{t-1}) \cdot f(\Delta^2 P_t, \frac{d}{dt} \sigma_t, X_t^3, X_t^4, \dots, X_t^n)$

Where:
- $S_t$ represents the system state at time t, including climate and price features.
- $\Delta^2 P_t$: second-order price difference (i.e., price acceleration), used as a proxy for directional momentum.
- $\frac{d}{dt} \sigma_t$: rate of change in rolling price volatility, indicating growing market instability.
- $X_t^3, X_t^4, \ldots, X_t^n$: additional bootstrapped signals, which may include:
  - Global news sentiment volatility
  - Trade volume anomalies
  - Model residual shocks (unexpected error spikes)
  - Macro-economic indicators (e.g., global freight cost indexes)

Together, these signals form a shock detection vector that informs how the model should interpret the current environment.

### Model Architecture

1. **Base Climate Model**

   A traditional supervised learning model (e.g., random forest or linear regression) trained on:
   - Climate features (temperature, precipitation, anomalies)
   - Lagged price values (1–3 months)

   This model serves as the stable, interpretable foundation that reflects normal climate-driven pricing behavior.

2. **Adaptive Meta-Layer**

   An online layer that continuously monitors system dynamics:
   - Calculates the values of all momentum and volatility-based signals
   - Evaluates a shock score derived from the composite signal vector
   - Adjusts the weight or confidence of the base model prediction depending on regime classification

   This layer can either:
   - Re-weight the prediction output from the base model, or
   - Invoke a different model logic altogether during disrupted regimes

3. **Shock Indicator Function**

   Instead of binary dummy variables, the model uses a multi-signal, real-valued function to assess system abnormality:
   
   $\text{ShockScore}_t = g(\Delta^2 P_t, \frac{d}{dt} \sigma_t, \ldots, X_t^n)$
   
   This score can then be passed through a sigmoid or softmax to produce regime probabilities.

### Theoretical Foundation and Novelty

This framework draws on multiple established modeling paradigms:
- From financial econometrics: regime-switching logic and momentum-based volatility signals
- From macroeconomic forecasting: time-varying parameter models and latent state estimation
- From climate modeling: changepoint detection and anomaly indexing

What distinguishes this approach is its fusion of these disciplines into a unified, adaptive system grounded in real-world signals, capable of recognizing when external forces override internal dynamics.

Importantly, this model does not aim to predict when black-swan events will occur—but rather, to recognize early patterns that resemble known disruptions, and respond accordingly.

### Why It Matters

Rather than discarding post-2020 data as an anomaly, this model learns from those disruptions, treating them as training signals for building resilience into future forecasts. It respects the interpretability of climate-driven modeling while acknowledging the reality of modern market complexity.

The Markovian Momentum Model offers a flexible, explainable, and extendable solution for modeling climate-commodity interactions in an increasingly volatile world.

## Next Steps

To implement and validate the Markovian Momentum Model approach, the following steps are planned:

1. **Feature Engineering**
   - Implement price acceleration features (second-order differences)
   - Calculate rolling volatility metrics and their derivatives
   - Develop shock indicator functions based on historical patterns

2. **Base Model Development**
   - Train base climate models for each commodity
   - Evaluate performance during normal periods (pre-2020)
   - Establish baseline predictions and confidence intervals

3. **Meta-Model Development**
   - Create the adaptive layer to detect regime changes
   - Implement reweighting mechanisms for predictions
   - Define rules for model adjustment during shock periods

4. **Validation Framework**
   - Create a historical shock identification system
   - Test model performance across identified shock periods
   - Validate stability during normal periods

5. **Model Integration**
   - Combine base and meta models into a unified prediction system
   - Develop real-time monitoring for production use
   - Create interpretability measures to explain predictions

6. **Expanded Data Integration**
   - Incorporate additional external shock indicators
   - Include trade volume, inventory, and policy change data
   - Integrate global supply chain disruption metrics

This enhanced modeling approach aims to create a more resilient framework for understanding climate-commodity relationships, one that acknowledges the complex, non-stationary nature of global agricultural markets while still maintaining climate as a central explanatory factor.