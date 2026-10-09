# Rossmann store sales: a 48-day forecast

Forecasting daily sales for 1,115 Rossmann drug stores, 48 days ahead, using the [Kaggle Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) data (2.5 years of daily history). The project ends in a what-if app: *"what happens to this store's sales if we run a promo next week?"*

The focus is less on squeezing the leaderboard and more on **decisions backed by evidence**: every modelling choice traces back to a finding in the notebooks, and each decision is logged there.

## Results

Scored with RMSPE (root mean squared percentage error, the competition metric), averaged over three rolling 48-day validation windows (Mar to Jul 2015):

| Model | RMSPE | vs baseline |
|---|---|---|
| Store median | 0.310 | |
| Store × weekday median | 0.237 | |
| Store × weekday × promo median (**baseline**) | 0.154 | |
| **Gradient boosting** | **0.106** | **−31%** |

The model beats the baseline on every fold (by 23–37%), and its error **stays flat across the 48-day horizon**: about 0.098 in the first week and 0.106 over days 29–48.

For scale only: the winning Kaggle private-leaderboard score was about 0.100. That was scored on a different period, so it isn't directly comparable.

## Approach

1. **Exploratory analysis** ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)): question → finding → decision for each property of the data that affects modelling.
2. **Metric and validation** ([src/metrics.py](src/metrics.py), [src/validation.py](src/validation.py)): RMSPE, and rolling-origin folds that copy the real setup: train on the past, forecast the next 48 days.
3. **Baselines** ([src/models.py](src/models.py)): simple grouped medians, so the model's score has something to be measured against.
4. **Features** ([src/features.py](src/features.py)): built around the 48-day horizon (below).
5. **Model**: one scikit-learn `HistGradientBoostingRegressor` for all stores, trained on log(Sales).
6. **Error analysis** ([notebooks/02_modelling.ipynb](notebooks/02_modelling.ipynb)): where the model fails, whether it's biased, and what it relies on.
7. **What-if scenarios** ([src/scenarios.py](src/scenarios.py), [app/streamlit_app.py](app/streamlit_app.py)): change promos or opening days for a store and compare forecasts.

## Key decisions

| Decision | Why |
|---|---|
| **No short lag features** ("sales yesterday / last week") | At day 48 of the forecast, the newest known sales are 48 days old. Short lags would look great in validation and be unavailable for most of the real forecast. Instead, store-level history (each store's typical level by weekday and promo), fitted only on each fold's training period. The flat error across the horizon confirms the choice. |
| **Time-based validation, never random k-fold** | The test set is a 48-day block straight after training. Random folds would let the model train on days after the ones it predicts. The same reasoning ruled out scikit-learn's default early stopping, which holds out a *random* 10% of rows. |
| **`Customers` excluded as a same-day feature** | It correlates strongly with sales (≈0.82) but is absent from the test set: it's an outcome of the day, not known in advance. Its *historical average* per store is used, since past counts are known. |
| **Train only on open days; closed days forecast as 0 by rule** | Closed days always have zero sales (≈17% of rows). Training on them teaches "closed means zero", not anything about demand. |
| **Log(Sales) target** | RMSPE scores ratios; in log space a 10% miss costs the same for small and large stores. |
| **Prediction scaling tested and rejected** | Shrinking predictions slightly is a known RMSPE trick. Here the best factor flipped between folds and the average favoured 1.0. The model already under-forecasts by ≈1%. |
| **Missing values treated by cause** | Promo2 gaps are *structural* (stores not in Promo2) → sentinels. Competition distance is *genuinely unknown* for 3 stores → median plus a "was missing" flag. Missing `Open` in test is one store with a clear weekday pattern → filled as open. |
| **`Promo2Active` built from the raw fields** | The raw `Promo2` flag only means "this store sometimes runs Promo2". Whether it's active on a given day depends on the start date and the restart months, where `Sept` (not `Sep`) would silently break naive month parsing. Covered by a regression test. |

## Limitations

- **Refurbishment closures.** The worst misses are the last trading day before a multi-week closure, when sales collapse. No feature was added because the test window contains no long closures; a production system would need one.
- **State holidays and Sundays** are poorly predicted (RMSPE ≈0.2–0.4) but are under 1% of scored days.
- **Scenarios are associations, not causal effects.** Promos weren't assigned at random, so a predicted promo uplift mixes the promo's effect with whatever usually came with promo days. The app says this, and blocks weekend promos (none in the training data, so the model can't forecast them).
- **The 2014 gap.** 180 stores have no data from July to December 2014. The Aug–Sep 2014 "seasonal" check fold therefore scores only 935 stores, so it is used as a secondary check, not the headline score.

## Run it

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the data from the [competition page](https://www.kaggle.com/competitions/rossmann-store-sales/data) (you need to accept the competition rules) and put `train.csv`, `test.csv` and `store.csv` in `data/`.

```bash
pytest                                # 29 tests: metric, folds, leakage guards, features, scenarios
streamlit run app/streamlit_app.py    # first launch trains the final model (~1 minute)
```

## Project structure

```
src/
  data.py         load and clean; no feature engineering
  metrics.py      RMSPE
  validation.py   rolling 48-day folds, cross_validate()
  features.py     known-in-advance features + StoreHistory (fitted per fold)
  models.py       median baselines, GradientBoostingModel
  scenarios.py    what-if forecasts for one store
app/
  streamlit_app.py
notebooks/
  01_eda.ipynb        exploratory analysis and decision log
  02_modelling.ipynb  evaluation, error analysis, permutation importance
tests/
```
