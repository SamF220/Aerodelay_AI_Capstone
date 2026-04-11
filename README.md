# AeroDelay AI - Hofstra MS Capstone

## Introduction

Instead of guessing, travelers get forecasted arrival hiccups through AeroDelay AI, a browser-accessible tool running predictive algorithms on live air travel information. By inputting a departure point, destination airport, scheduled day, and hour, the user receives estimated punctuality outcomes supported by contributing factors such as weather conditions, airport congestion proxies, and prior traffic patterns. Flight-related information is gathered through public aviation interfaces such as Aviationstack, while atmospheric conditions such as temperature, wind, and precipitation are collected from OpenWeatherMap. Airport metadata and coordinate support are also incorporated through AirportDB.

Behind the scenes, historical and structured flight-related records are used to train machine learning models, including a Random Forest approach built with scikit-learn. These models estimate likely delay behavior and support user-facing results through an interactive interface. AeroDelay AI reflects a hands-on application of artificial intelligence in flight operations, where combining multiple information streams is essential for producing realistic delay estimates and meaningful explanations.

---

## Overview

AeroDelay AI is a browser-based flight delay prediction project developed for my Hofstra University master’s capstone in computer science. The purpose of the project is to estimate expected flight delay minutes using airport data, flight data, weather data, machine learning, and prediction explainability.

The system uses:
- Aviationstack for route-level flight context
- OpenWeatherMap for weather conditions
- AirportDB for airport metadata and coordinate lookup
- a local airport master table for broader airport search
- SQLite for structured storage
- scikit-learn for machine learning
- SHAP for explainability
- Gradio for the final browser interface

The application allows a user to:
- choose an origin and destination airport
- type a full airport name, IATA code, or ICAO code
- enter a scheduled departure date and time
- receive a predicted delay in minutes
- receive a delay risk probability
- view live route and weather context
- view the top SHAP drivers behind the prediction

---

## Project Goal

The goal of this project is to demonstrate an end-to-end AI application pipeline that combines:
- live API integration
- structured SQL storage
- data cleaning and refinement
- feature engineering
- supervised learning
- model evaluation
- interpretable predictions
- public deployment through a browser-based interface

This is a project prototype and should be understood as a realistic predictive demo, not a production airline dispatch system.

---

## Methodology

Beginning with data assembly, historical alongside current flight details, including departure and arrival points, airline identifiers, and timing in universal coordinated time, are combined carefully with atmospheric conditions such as wind strength, rain occurrence, and visibility. Merged through uniform timestamps, these form structured datasets held within a standard SQL environment, where organized tables rely on defined relationships and rapid lookups via indexing.

Prior to storage, information undergoes refinement through a repeatable transformation workflow. Inconsistencies, repeated entries, and gaps are addressed systematically while integrity checks are applied at each stage. Attributes useful for analysis are then engineered, including portions of day, weekly position, annual cycles, encoded airports and carriers, estimated path lengths, and prior traffic-related measures.

From there, simple predictive structures arise first using logistic methods, followed by more adaptive algorithms including tree-based ensemble approaches aimed at estimating either the likelihood of late departures or the actual duration past scheduled times. When assessing performance, temporal partitions are applied, with measures such as AUC and F1 used for classification tasks, while MAE and RMSE are used for numerical delay predictions. To clarify individual predictions, SHAP is employed for localized interpretation so that the system provides not only outputs, but also an explanation of the factors influencing those outputs.

---

## Note on Implementation

The original methodology described outputs through a Flask-powered interface paired with a minimal viewing panel. In the implemented version of this Google Colab project, **Gradio** is used to deliver the final browser-based application. Gradio provides real-time input forms, model-driven outputs, and SHAP-based explanations in an interactive interface.

Although the earlier conceptual framing referenced Flask and React, the final working capstone implementation uses **Google Colab + Gradio** as the delivery environment.

---

## Main Components

### 1. Flight data
Aviationstack is used for route-level flight context and live route hints where available.

### 2. Weather data
OpenWeatherMap is used to gather weather conditions, especially at the origin airport.

### 3. Airport metadata
AirportDB is used to look up airport details and coordinates.

### 4. Airport search coverage
A broader airport master table is used so the interface can support:
- full airport names
- 3-letter IATA codes
- 4-letter ICAO codes

### 5. SQL storage
SQLite is used to store structured project data and intermediate outputs.

### 6. Machine learning
Two models are used:
- RandomForestRegressor for predicted delay minutes
- LogisticRegression for delay risk probability

### 7. Explainability
SHAP is used to explain which features most influenced the model output.

---

## Project Files

- `app.py` - final Gradio application
- `AeroDelay_AI_Flight_Delay_Predictor.ipynb` - notebook version of the project
- `requirements.txt` - Python dependencies
- `README.md` - documentation
- `aerodelay.db` - SQLite database
- `aerodelay_rf_model.pkl` - trained Random Forest model
- `aerodelay_log_model.pkl` - trained Logistic Regression model
- `aerodelay_artifacts.pkl` - saved feature lists, medians, and safe aggregate maps
- `aerodelay_encoding_maps.pkl` - saved encoding maps

---

## Data Pipeline Summary

1. Airport, flight, and weather data are collected.
2. Raw data is stored in SQLite.
3. Duplicate or code-share-like rows are reduced.
4. Features are engineered, including:
   - departure hour
   - departure minute
   - route identity
   - airport and carrier encodings
   - estimated path length
   - safe historical delay priors
   - flight count priors
   - weather severity indicators
5. The cleaned dataset is split temporally into train and test subsets.
6. The models are trained and saved.
7. The final Gradio app loads saved artifacts instead of retraining during launch.

---

## Evaluation Summary

The project evaluates the models on a held-out temporal test split.

Example metrics from one run:
- Random Forest MAE: 33.753
- Random Forest RMSE: 56.873
- Logistic Accuracy: 0.7143
- Logistic AUC: 0.6061
- Logistic F1: 0.8182

These values should be interpreted carefully because the cleaned training set is still relatively small.

---

## Important Note: Why PEK and PKX Can Return the Same Delay

A known limitation in the current version is that routes such as `JFK -> PEK` and `JFK -> PKX` may still return the same predicted delay minutes even after the fallback bug was fixed.

### Earlier issue
Earlier in development, unseen destinations and routes were collapsing into the same fallback feature values. That caused different destinations to become nearly identical at prediction time.

### Current situation
That collapse bug was patched so that unseen destinations and routes no longer share the exact same fallback encodings and priors.

However, the delay prediction model is a **RandomForestRegressor**, which produces piecewise-constant outputs. That means two different input rows can still land in the same leaf pattern and produce the same delay minutes.

For PEK and PKX specifically, the final feature rows can still be fairly close because:
- both are Beijing airports
- route distance is similar
- the cleaned training data is relatively small
- the safe aggregate priors for rare or unseen routes are still close
- live route-specific counts may be sparse

As a result, the model may output the same delay minutes while the risk probability changes slightly.

### Meaning
This is no longer just a coding bug. It is mainly a **model sensitivity and data coverage limitation**.

### Best next improvements
The best next improvements would be:
- train on more destination-specific rows
- include stronger destination-side features
- increase route-specific historical coverage
- reduce dependence on fallback priors for rare destinations
- consider alternative encodings or models for sparse routes

---

## Deployment Notes

### In Google Colab
The project can read secrets stored in Colab:
- `AVIATIONSTACK_API_KEY`
- `OPENWEATHER_API_KEY`
- `AIRPORTDB_API_KEY`

### In Other Environments
The same names should be set as environment variables:
- `AVIATIONSTACK_API_KEY`
- `OPENWEATHER_API_KEY`
- `AIRPORTDB_API_KEY`

The app is written to read either Colab secrets or standard environment variables, depending on where it is running.

---

## How to Run

1. Install dependencies from `requirements.txt`
2. Make sure the model and database files are present
3. Set the required secrets or environment variables
4. Run `app.py`

---

## Notes for Review

This project is intended to demonstrate the complete capstone workflow:
- data integration
- database usage
- machine learning
- explainability
- deployment

It should be evaluated as an end-to-end AI systems project rather than as a production-grade aviation forecasting platform.

---

## Author

Samuel Jhan  
Hofstra University  
MS in Computer Science Capstone