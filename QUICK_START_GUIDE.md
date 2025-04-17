# Climate and Commodity Analysis Quick Start Guide

This guide provides a quick overview of how to get started with the Climate and Commodity Analysis project.

## Setup

1. Install required packages:
   
   The notebook template has been designed to gracefully handle missing packages, but for full functionality, install the recommended packages:
   
   ```
   # Basic data processing and visualization (required)
   pip install pandas numpy matplotlib seaborn
   
   # Machine learning (recommended)
   pip install scikit-learn
   
   # Time series analysis (optional)
   pip install statsmodels pmdarima
   
   # Deep learning (optional)
   pip install tensorflow
   ```
   
   See `REQUIREMENTS.md` for more details.

2. Activate the Machine Learning environment:
   ```
   source ./Machine_Learning/bin/activate
   ```

3. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

## Getting Started with the Data

The project includes a comprehensive dataset of climate variables and commodity prices for multiple commodities:

- Coffee
- Cocoa
- Wheat
- Maize
- Rice
- Soybeans
- Cotton

### Key Files

- **MASTER_climate_commodity_data.csv**: The complete dataset with all commodities and climate variables
- **Master_Dataset_Analysis.ipynb**: Notebook for analyzing the complete dataset
- **Master_Data_Analysis_Template.ipynb**: Template notebook for your own analyses

## Using the Template Notebook

The `Master_Data_Analysis_Template.ipynb` notebook is designed to help you quickly start your own analysis. It includes:

1. **Data Loading**: Code to load the master dataset
2. **Subsetting**: Functions to extract data for specific commodities
3. **Visualization**: Basic plots and correlation analyses
4. **Random Forest Model**: A complete setup for training and evaluating a Random Forest model
5. **Templates**: Code snippets for other data science methods

### How to Use the Template

1. Open the template notebook:
   ```
   jupyter notebook organized/notebooks/Master_Data_Analysis_Template.ipynb
   ```

2. Choose a commodity to analyze by changing the `commodity` variable:
   ```python
   commodity = 'Coffee'  # Change to 'Wheat', 'Cocoa', etc.
   ```

3. Run the cells to:
   - Load the data
   - Extract data for your chosen commodity
   - Visualize price trends and correlations
   - Train a Random Forest model
   - Analyze feature importance

4. Modify the code to implement your own analyses:
   - Try different models
   - Add new features
   - Test different time lags
   - Implement your own visualization

## Climate Variables

Each commodity in the dataset includes the following climate variables:

- `temperature_C`: Average temperature in Celsius
- `precip_m`: Precipitation in meters
- `dewpoint_C`: Dew point temperature in Celsius
- `relative_humidity`: Relative humidity (%)
- `temp_anomaly`: Temperature anomaly from historical average
- `precip_anomaly`: Precipitation anomaly from historical average
- `temp_3m_avg`: 3-month moving average of temperature
- `precip_3m_sum`: 3-month sum of precipitation
- `drought_index`: Calculated drought index
- `heat_stress`: Heat stress index

## Example Analysis Tasks

Here are some example analyses you can perform:

1. **Price Prediction**:
   - Predict commodity prices based on climate variables
   - Test different prediction horizons (1-month, 3-month, etc.)

2. **Climate Impact Analysis**:
   - Identify which climate variables have the strongest impact on prices
   - Compare climate impacts across different commodities

3. **Seasonal Patterns**:
   - Analyze how prices and climate variables change with seasons
   - Detect seasonal price patterns for different commodities

4. **Anomaly Detection**:
   - Identify unusual price movements
   - Correlate price anomalies with climate events

5. **Cross-Commodity Analysis**:
   - Analyze how climate affects different commodities
   - Find correlations between different commodity prices

## Advanced Topics

For more advanced analyses, consider:

1. **Deep Learning**:
   - Implement LSTM or other neural network architectures
   - Capture complex non-linear relationships

2. **Multivariate Time Series**:
   - Model multiple commodities together
   - Include external factors like economic indicators

3. **Causal Inference**:
   - Implement methods to infer causal relationships
   - Use techniques like Granger causality or causal forests

4. **Spatial Analysis**:
   - Incorporate more detailed geographic information
   - Create visualizations with geographic mapping

5. **Ensemble Methods**:
   - Combine multiple models for better predictions
   - Implement stacking or boosting methods

## Getting Help

If you need assistance or have questions, refer to the detailed documentation in the `organized/docs/` directory.