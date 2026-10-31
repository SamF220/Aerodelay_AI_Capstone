# AeroDelay AI

AeroDelay AI estimates the chance that a US domestic flight arrives more than 15 minutes late. You enter an origin airport, a destination airport, a local departure date and time, and (optionally) an airline. It returns a risk band (LOW, ELEVATED or HIGH), a typical-case delay estimate in minutes, the weather it used for both airports, and the features that pushed the estimate up or down.

Graduate capstone project, M.S. Computer Science, Hofstra University, by Samuel F. Jhan.

**Live app:** https://huggingface.co/spaces/SamF220/Aerodelay_AI_Capstone

## How it works

- **Flights:** Bureau of Transportation Statistics (BTS) Reporting Carrier On-Time Performance data, October 2025 to July 2026. That is 5.88 million scheduled US domestic flights, sampled to 1,000,000 flights covering 347 airports, 14 airlines and 6,763 routes. Cancelled and diverted flights are excluded, so the model describes flights that actually operated.
- **Weather:** Open-Meteo hourly weather (ECMWF IFS model), matched to the origin at the scheduled departure hour and to the destination at the scheduled arrival hour.
- **Features:** 42 features built from the schedule, the route, smoothed historical delay averages (computed out-of-fold so a flight never sees its own delay), airport congestion in that hour, and origin and destination weather. The full list is stored in `aerodelay_artifacts.pkl` under `model_feature_columns`.
- **Models:** a Random Forest regressor for delay minutes and a gradient boosting classifier (`HistGradientBoostingClassifier`) for the probability of arriving more than 15 minutes late. Logistic regression was used only as a baseline.
- **Calibration:** the risk probability is Platt-calibrated on the May 2026 validation month (105,160 flights). The classifier was not retrained, so the AUC is unchanged at 0.723. The two calibration numbers are stored in `calibration.json`.
- **Explanations:** the app shows a SHAP **"Why this risk"** list on the gradient boosting classifier (in approximate percentage points of calibrated risk), plus **"What drives the minutes estimate"** from SHAP on the Random Forest.
- **Storage:** an SQLite database (`aerodelay.db`) holds the lookup tables the app needs.

## Weather

Weather is one of the first things people think about before a flight. People check flights ahead of time, so the app fetches weather for the date you choose:

1. Within about 15 days: Open-Meteo **forecast**.
2. Past dates: Open-Meteo **historical** weather.
3. Further out: the same dates in an earlier year (plus or minus 3 days). It steps back whole years until that week is at least 5 days in the past (the archive lags about 5 days), names the year used, and is labelled **"SEASONAL ANALOG, NOT a forecast"**.
4. If Open-Meteo is rate-limited or has a server error, the app retries twice. If it still cannot be reached: typical training values, labelled "weather forecast unavailable, typical values used".

## Results

Evaluated on flights the model never saw during fitting or tuning: June and July 2026 (210,696 flights). Training used October 2025 to April 2026, validation used May 2026, and the shipped models were refit on both.

| Measure | Result |
|---|---|
| Risk AUC (0.5 is guessing) | 0.723 |
| Precision-recall AUC | 0.475 |
| At the 0.20 risk cutoff: precision / recall / F1 | 0.417 / 0.652 / 0.509 |
| Accuracy at that cutoff vs always guessing on time | 0.670 vs 0.737 |
| Minutes MAE, Random Forest vs always guessing the median | 29.2 vs 29.8 |

Calibration on the same test months (Platt, fit on May 2026, AUC unchanged):

| Measure | Before | After | Actual |
|---|---|---|---|
| Average predicted risk | 20.1% | 20.4% | 26.3% |
| Top 10% riskiest flights | 47.5% | 55.4% | 57.6% |

Full numbers are in `final_metrics.json`. The calibration scripts and their results are in the `calibration/` folder.

## Limitations

- Delay minutes are hard to predict from schedule and weather alone. The minutes estimate is only slightly better than always guessing the median, so the risk probability is the useful output.
- The training months (Oct to Apr) have a 21.1% delay rate and the test months (Jun to Jul) have 26.3%. The risk model under-predicts on the summer test months (average predicted risk 20.1%). Calibration on May only moves this to 20.4%, because May itself is not as delayed as June and July.
- There is no August or September data.
- The model does not know about same-day events such as a late inbound aircraft or a ground stop.
- Forecast weather at prediction time can differ from the observed weather the model was trained on. That effect was not measured.
- US domestic flights only. Airports outside the 347 and airlines outside the 14 fall back to overall averages and the app shows a warning.

## Run it yourself

1. Install: `pip install -r requirements.txt` (the pickles were written with scikit-learn 1.9.1, so keep that version).
2. Start the app: `python app.py`, then open the local address Gradio prints.
3. To rebuild the models from scratch, open `AeroDelay_AI_retrain.ipynb` and run the cells top to bottom. All settings are in one configuration cell at the top, and no API keys are needed. A full run takes about an hour. Free Google Colab (about 12 GB of RAM) may run out of memory in the final-fit cell, so use a High-RAM runtime or a local machine.
4. The BTS and Open-Meteo data are re-downloaded on every run, so the numbers can shift slightly. The saved notebook outputs (run on Oct 30, 2026) show a test AUC of 0.729, compared with 0.723 in the report.

`AeroDelay_AI_1.ipynb` is the earlier version of the notebook and is kept only for history. It still runs, but its model was trained on an incorrect setup, so please do not use it.

## Data sources

- U.S. Department of Transportation, Bureau of Transportation Statistics. "Reporting Carrier On-Time Performance (1987-present)." https://www.transtats.bts.gov/
- Open-Meteo. Weather Forecast API and Historical Weather API. https://open-meteo.com/en/docs
- OurAirports airport table used for airport lookup (`airport_master` in the database).
