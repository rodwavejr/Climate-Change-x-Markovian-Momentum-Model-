# The Markovian Momentum Model (MMM): A Methodological Foundation for Regime-Aware Climate-Commodity Forecasting

## Introduction

The Markovian Momentum Model (MMM) is a hybrid predictive framework developed to improve the forecasting of agricultural commodity prices in the presence of non-stationarity, structural breaks, and real-world shocks. It builds on a key observation: traditional climate-based forecasting models tend to fail under extreme conditions—such as the COVID-19 pandemic—where price movements become decoupled from underlying environmental variables. MMM addresses this by integrating principles from financial econometrics, volatility modeling, and machine learning into a unified system that remains grounded in climate fundamentals while adapting to disruption.

MMM views commodity prices as a **non-stationary Markov process**, augmented by exogenous and endogenous signals to detect shifts in regime. The model operates on two levels: a climate-driven base predictor (e.g., Random Forest) and a meta-model that adjusts the base forecast based on momentum, volatility, and shock indicators. Its novelty lies in **its dynamic regime awareness**, powered by real-time signals engineered from both internal price dynamics and external bootstrapped data sources.

## A Unified Intuition (Plain + Technical Logic)

When building a model to understand the relationship between climate and commodity prices—and to forecast those prices month over month—most of the time, the logic is clean and stable: if it gets hotter or drier in a key growing region, prices tend to rise. But what happens when something outside that data intervenes—like a pandemic, war, or shipping collapse? During our analysis, we found that in regular, calm periods, variables like global shipping cost show weak or no correlation with commodity prices. But when we zoom in on crisis periods—regime shifts—the signal becomes incredibly powerful, in fact more useful than the climate variables themselves. This raised the key challenge: how do you build a model that can adapt to something you don't even know is coming? The Markovian Momentum Model was our answer. By engineering features like price acceleration (second-order differences), rolling volatility shifts, and macroeconomic shock signals, we compute a kind of "momentum of disruption"—an unseen force building behind the price. This lets the meta-model determine whether the current regime is stable or abnormal, and adjust how much to trust the climate-based prediction accordingly. The result is a smarter, dynamic forecast engine that learns when the world is behaving unusually and responds before it fails.

## How MMM Differs from Classical Markov Chains

Traditional Markov chains are defined by a simple assumption: the probability of moving to a future state depends only on the current state, not on the history of how that state was reached. *A classic example might be weather modeling*: if today is rainy, a Markov chain might say there's a 70% chance tomorrow will also be rainy, and a 30% chance it will be sunny—regardless of the weather from previous days. This structure is memoryless, stationary, and self-contained within its defined transition matrix.

Non-stationary Markov chains relax the idea of fixed transition probabilities. Instead, the probability of moving between states can change over time, often due to hidden or external factors. For example, if a region is entering a new climate cycle, the transition from sunny to rainy might increase seasonally. This type of approach still treats state transitions probabilistically but introduces time-dependence or contextuality.

The Markovian Momentum Model takes this one step further. It doesn't just assume states or transition probabilities vary—it actively **infers** when transitions are likely, by reading the "momentum" of the system. Instead of a fixed or slowly varying transition matrix, MMM is structured around the idea that **external shocks and endogenous acceleration can distort the typical trajectory of the system**, prompting abrupt shifts that can't be predicted from the current state alone.

![Classical vs Adaptive Models](/organized/docs/Screenshot%202025-04-19%20at%204.50.43 PM.png)



Imagine you're watching the price of soybeans: a classical Markov chain might say "Prices have been stable for two months, so it will likely stay stable." A non-stationary model might adjust the transition probability based on time or season. But MMM would detect that although prices are stable, **momentum is building**—shipping costs are surging, volatility is rising, and drought forecasts are increasing. These forces act like pressure behind a dam. MMM recognizes that and adjusts its belief about the future, effectively saying: "A regime change is imminent—even if it hasn't happened yet."

This is what makes MMM novel: it transforms the regime-switching process from being **reactionary and memoryless** to being **anticipatory and momentum-informed**. It doesn't just model states—it models the **potential energy of change**, allowing the system to behave more like a living sensor than a static predictor.

![Comparison of Markov Models](/organized/docs/Screenshot%202025-04-19%20at%204.50.19 PM.png)

## Theoretical Foundations

MMM is inspired by several precedent frameworks:

1. **Markov Regime-Switching (MRS) Models**: First introduced by Hamilton (1989), MRS models allow systems to switch between discrete states (e.g., stable vs. volatile) with probabilities governed by a Markov chain. MMM extends this concept by allowing regime switches to be driven not only by hidden latent states but also by observable shock signals.

2. **Time-Varying Transition Probabilities (TVTP)**: As used by Cifarelli (2023), TVTP models allow regime transitions to be a function of external variables (e.g., oil price changes, equity index shifts). MMM adopts this principle by using bootstrapped features—such as shipping cost volatility—as inputs to its regime-switching logic.

3. **Momentum and Acceleration Factors**: Empirical work by Ardila et al. (2015) shows that second-order price differences (acceleration) can be stronger predictors than raw momentum. MMM uses both first- and second-order price derivatives to identify trend persistence, bubbles, and crashes.

4. **Adaptive Learning and Concept Drift Detection**: Drawing from machine learning literature (e.g., Adaptive Random Forests by Gomes et al., 2017), MMM introduces a meta-model that continuously evaluates the model's predictive environment. When drift or a structural break is detected, the meta-model either adjusts the base prediction or shifts forecasting regimes.

## Model Architecture

### 1. Base Climate Model

The base model is a supervised learning algorithm trained on climate and market variables (e.g., temperature anomaly, drought index, lagged prices). This model captures the fundamental climate-driven pricing relationships under normal market conditions.

### 2. Bootstrapped Exogenous Signals

To handle non-climate-driven volatility, MMM bootstraps in macroeconomic and logistical indicators such as:

- Global shipping costs (FRED: Deep Sea Freight Index)
- Price momentum and acceleration
- Rolling price volatility and changes

These features are engineered into a **shock vector** $\mathbf{z}_t$ used to monitor regime stability.

### 3. Meta-Model (Shock Detection Layer)

A logistic classifier or probabilistic layer maps the shock vector to a regime probability:
$\alpha_t = f(\mathbf{z}_t) = \sigma(w^T \mathbf{z}_t + b)$

This $\alpha_t \in [0,1]$ is interpreted as the system's belief in a shock regime.

### 4. Final Prediction

The final forecast is a weighted average of the base model prediction and an alternative (e.g., momentum-driven) forecast:
$\hat{y}_t = (1 - \alpha_t) \cdot \hat{y}_{\text{climate}} + \alpha_t \cdot \hat{y}_{\text{adjusted}}$

If $\alpha_t \approx 0$, the system trusts the climate model. If $\alpha_t \approx 1$, the model pivots to a more shock-aware prediction logic.

## Feature Integration and Engineering

To integrate bootstrapped signals into the forecasting pipeline, MMM follows these steps:

1. **Download external data** (e.g., shipping costs from FRED) and align it with the monthly structure of the master dataset.

2. **Engineer features** such as:

   - $\Delta P_t$: First difference (momentum)
   - $\Delta^2 P_t$: Second difference (acceleration)
   - Rolling volatility: $\text{std}(P_{t-3}, ..., P_t)$
   - Shipping cost derivatives: $\Delta \text{Shipping}_t, \Delta^2 \text{Shipping}_t$
   - Z-score normalization of anomalies

3. **Merge with climate and price data**, resulting in a wide feature matrix with both fundamental and dynamic signals.

4. **Train the base model** on pre-shock historical data, preserving climate signal integrity.

5. **Train the meta-model** using both historical disruptions (e.g., COVID) and synthetic shock windows, labeling them as regime events.

6. **Evaluate model confidence** monthly using $\alpha_t$, and blend outputs as needed.

## Applications and Broader Relevance

While MMM was developed for agricultural commodities under climate volatility, its methodology generalizes to any domain where:

- There are known exogenous disruptions (e.g., pandemics, wars, policy shocks)
- The underlying dynamics shift across regimes
- Forecasting accuracy depends on detecting and adapting to change

This includes energy markets, macroeconomic forecasting, retail demand modeling, and more.

## State Definition

The states in the MMM represent distinct regimes in which the system operates, such as:

1. **Stable Market**: Characterized by low volatility and predictable price movements
2. **Transition Phase**: Moderate volatility with directional price movements
3. **Shock State**: High volatility, rapid price changes, and breakdown of normal relationships

## Transition Probability Modification

The transition probabilities between states are continuously updated based on the momentum indicators:

$P'(S_{t+1} = j | S_t = i) = P(S_{t+1} = j | S_t = i) \cdot \alpha_{ij}(\textbf{M}_t)$

Where $\alpha_{ij}(\textbf{M}_t)$ is the adjustment factor derived from the momentum vector for the transition from state i to state j.

## Implementation Strategy

The practical implementation of the MMM involves the following steps:

1. **State Identification**: Using clustering techniques to identify distinct market regimes in historical data
2. **Base Probability Estimation**: Calculating the initial transition probabilities between states
3. **Momentum Feature Engineering**: Developing the momentum indicators from price data and external variables
4. **Adjustment Function Calibration**: Training the adjustment function using historical periods that include regime shifts
5. **Continuous Updating**: Recalibrating the model as new data becomes available

## Applications to Climate-Price Relationships

In the context of climate-price relationships, the MMM offers several advantages:

1. **Regime-Aware Predictions**: Different climate variables may have varying impacts on commodity prices depending on the market regime
2. **Early Warning System**: The momentum indicators can provide early signals of impending regime shifts
3. **Resilience to Shocks**: By explicitly modeling transitions between regimes, the MMM can maintain predictive power during market disruptions
4. **Integration of Multiple Data Sources**: The model can incorporate diverse signals including climate variables, market indicators, and shipping costs

## Empirical Validation

The MMM has been validated using historical commodity price data spanning multiple market regimes, including the pre- and post-COVID periods. Key validation metrics include:

1. **Prediction Accuracy**: Measured by RMSE and MAE across different market regimes
2. **Regime Shift Detection**: The lead time with which the model identifies transitions between states
3. **Resilience Comparison**: Performance degradation during shock periods compared to traditional models

## Conclusion

The Markovian Momentum Model represents a significant advancement in predictive modeling for systems characterized by regime shifts. By combining the memory properties of Markov chains with forward-looking momentum indicators, the MMM provides a robust framework for understanding and forecasting complex relationships between climate variables and commodity prices, even during periods of market disruption.

Its real innovation lies in its structure: a meta-model that continuously gauges the credibility of its own assumptions—and adapts before it fails.

Future research directions include refining the state definition process, exploring more sophisticated momentum indicators, and extending the model to other domains characterized by non-stationary behavior and regime shifts.