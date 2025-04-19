# Climate and Commodity Analysis: The Simple Version

## 🌾 The Goal (Real Simple)

You want to predict the price of crops—like coffee, rice, and soybeans—using weather. You're asking:

"If it's hotter or dryer than usual, will the price go up next month?"

So you built a big dataset (`/organized/data_files/MASTER_climate_commodity_data.csv`) that includes monthly:
- Weather (temperature, rainfall, etc.)
- Price of each crop
- Climate indicators like drought or heat stress

The dataset has 95 rows (months from Feb 2015 to Dec 2022) and 86 columns, including:
- `Coffee_Price`, `Wheat_Price`, `Rice_Price`, etc.
- `Coffee_temperature_C`, `Coffee_precip_m`, etc.
- `Coffee_temp_anomaly`, `Coffee_drought_index`, etc.

## 🧪 Step-by-Step: How the Model Works

### 🔹 Step 1: Train a model to learn normal patterns

You take the first 80% of your monthly data (Feb 2015 to May 2021), and say:

"Hey model, here's what the weather looked like each month and what the price was the next month—learn the pattern."

From `/organized/notebooks/Master_Data_Analysis_Template.ipynb`, we see:
```python
# Split the data chronologically - this keeps timestamps in order
split_idx = int(len(X) * (1 - test_size))  # Where test_size = 0.2 (20%)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
```

🧠 Example:

In March 2018, if:
- It was super dry in Ghana (shown by `Cocoa_precip_anomaly = -0.016`)
- And the temp anomaly was high (shown by `Cocoa_temp_anomaly = 1.5`)

And in April 2018, the price of cocoa jumped

→ The model learns: "Oh, hot & dry weather in Ghana usually means cocoa gets more expensive next month."

You repeat this logic for each crop and region, using:
- Lagged features: `temp_anomaly_lag_1`, `precip_anomaly_lag_1`, etc.
- Time features: `sin_month`, `cos_month` (capturing seasonality)

### 🔹 Step 2: Test the model on new months

Now you ask:

"Okay, model—you've seen 6 years of data. What do you think prices will be in the months after May 2021?"

Using the Random Forest model in `Master_Data_Analysis_Template.ipynb`:
```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_test = rf_model.predict(X_test)
```

You give it weather data from June 2021, and it tries to guess:

"What's the coffee price in July 2021?"

You compare the guess to the real price using metrics:
```python
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))  # 6.60
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))     # 100.68
train_r2 = r2_score(y_train, y_pred_train)                       # 0.82
test_r2 = r2_score(y_test, y_pred_test)                          # -13.45
```

## 🚨 The Problem

After 2020, something weird happens:
- COVID hits
- Supply chains break
- Prices go wild—not because of weather

Your model starts making bad predictions because the rules changed. The model's test R² is -13.45 (extremely poor), compared to the training R² of 0.82 (very good).

Looking at the actual vs. predicted plot, we see:
- Training period: Predicted prices closely follow actual prices (~$150)
- Testing period: Actual prices soar to ~$280, while predictions stay around ~$150

## ⚙️ This is where your Markovian Momentum Model (MMM) kicks in

### 🤖 What MMM does (in plain terms)

"Let's not assume the world always works the same. Let's build a system that notices when things feel off—and adjusts."

It's like adding a brain on top of your original model.

### 🔍 Step 3: MMM watches for weird patterns

You give MMM a few "sensors"—aka shock signals—to watch each month:

1. 📈 **Price acceleration**
   - Calculate: `Coffee_Price.diff().diff()`
   - "Is the price changing faster than normal?"
   - Like: Price went from $130 → $150 → $200 (really fast!)

2. 🌪️ **Volatility spike**
   - Calculate: `Coffee_Price.rolling(3).std().diff()`
   - "Are prices jumping all over the place?"
   - If last month was $120, then $190, then $140—that's unstable.

3. 🧠 **Model surprise (residuals)**
   - Calculate: `abs(y_actual - y_predicted)`
   - "Did the base model totally miss the mark?"
   - If the model said $150 and reality was $300—that's a shock.

4. 🌎 **Global disruption indicators**
   - You can even plug in things like:
     - News sentiment (how negative financial headlines are)
     - Shipping costs
     - COVID lockdown index
     - Trade volume drops

### 🧠 Example

Say it's August 2021.
- The base model sees:
  - High temperatures in Colombia (`Coffee_temperature_C = 24.5`)
  - Drought in Colombia (`Coffee_drought_index = -0.15`)
  - And predicts coffee should go from $180 → $190

But:
- The actual price goes to $260
- News sentiment is full of "supply crisis"
- Volatility jumped
- Residual error = huge

MMM says:

"Whoa. This smells like a shock month. I don't trust the base model 100%."

So it downweights the base prediction and adjusts.

### 📈 Final Prediction

MMM blends the climate model and shock awareness:

$\text{Final prediction} = (1 - \alpha) \cdot \text{climate model} + \alpha \cdot \text{shock-adjusted model}$
- If the month is calm → $\alpha$ is small → mostly trust the weather model
- If the month is wild → $\alpha$ is big → lean on the shock detector

## 🔁 Recap: What You've Built

You've made a 2-layer prediction system:
1. **Base model**: Predicts prices based on weather (using `/organized/scripts/combine_climate_commodity.py` for processing and Random Forest for modeling)
2. **Meta-model (MMM)**: Watches the world and says, "Hey, this month feels unstable. Let me adjust that forecast."

Together, they make smarter predictions—even when the world gets weird.

## 📊 The Data Pipeline

The whole system starts with:

1. **Raw climate data** in NetCDF format (`.nc` files in `/organized/raw_data/`)
2. **Quarterly aggregation** using `climate_data_loader.py`
3. **Monthly expansion** using `expand_quarterly_to_monthly()` in `combine_climate_commodity.py`:
   ```python
   # Turn quarterly data into monthly using interpolation
   merged[col] = merged[col].interpolate(method='linear')
   ```
4. **Feature engineering** to create:
   - Temperature anomalies (`temp_anomaly`)
   - Drought indices (`drought_index`)
   - 3-month moving averages (`temp_3m_avg`)
5. **Lagged features** for prediction:
   ```python
   # Create lagged features for all climate variables
   for col in feature_cols:
       for lag in range(1, max_lag + 1):
           model_df[f'{col}_lag_{lag}'] = model_df[col].shift(lag)
   ```

## 🔍 What We Learned

The most important variables (feature importance) were:
1. `temp_anomaly_lag_1` (0.08): Temperature anomaly from 1 month ago
2. `drought_index_lag_1` (0.07): Drought condition from 1 month ago
3. `drought_index_lag_3` (0.07): Drought condition from 3 months ago
4. `precip_anomaly` (0.06): Current precipitation anomaly
5. `drought_index_lag_2` (0.06): Drought condition from 2 months ago

This told us that **temperature anomalies with a 1-month lag** are the strongest predictor of commodity prices!