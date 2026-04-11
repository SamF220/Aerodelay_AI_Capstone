import gradio as gr
import joblib
import shap
import requests
import numpy as np
import pandas as pd
import sqlite3

from datetime import datetime, timezone
from math import radians, cos, sin, asin, sqrt
from collections import Counter

import os

try:
    from google.colab import userdata
    AVIATION_KEY = userdata.get("AVIATIONSTACK_API_KEY")
    OPENWEATHER_KEY = userdata.get("OPENWEATHER_API_KEY")
    AIRPORTDB_API_TOKEN = userdata.get("AIRPORTDB_API_KEY")
except Exception:
    AVIATION_KEY = os.getenv("AVIATIONSTACK_API_KEY")
    OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
    AIRPORTDB_API_TOKEN = os.getenv("AIRPORTDB_API_KEY")

if not AVIATION_KEY or not OPENWEATHER_KEY or not AIRPORTDB_API_TOKEN:
    raise ValueError(
        "Missing one or more Colab secrets: "
        "AVIATIONSTACK_API_KEY, OPENWEATHER_API_KEY, AIRPORTDB_API_KEY"
    )

class ConstantBinaryClassifier:
    def __init__(self, constant=0, positive_probability=None):
        self.constant = int(constant)
        self.positive_probability = positive_probability

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        import numpy as np
        return np.full(len(X), self.constant, dtype=int)

    def predict_proba(self, X):
        import numpy as np

        if self.positive_probability is None:
            p1 = float(self.constant)
        else:
            p1 = float(self.positive_probability)

        p1 = max(0.0, min(1.0, p1))
        p0 = 1.0 - p1
        return np.tile([p0, p1], (len(X), 1))

rf_model = joblib.load("aerodelay_rf_model.pkl")
log_model = joblib.load("aerodelay_log_model.pkl")
artifacts = joblib.load("aerodelay_artifacts.pkl")
encoding_maps = joblib.load("aerodelay_encoding_maps.pkl")

model_feature_columns = artifacts["model_feature_columns"]
train_medians = artifacts["train_medians"]

origin_mean_map = artifacts["origin_mean_map"]
destination_mean_map = artifacts["destination_mean_map"]
route_mean_map = artifacts["route_mean_map"]
carrier_mean_map = artifacts["carrier_mean_map"]

origin_count_map = artifacts["origin_count_map"]
destination_count_map = artifacts["destination_count_map"]
route_count_map = artifacts["route_count_map"]
carrier_count_map = artifacts["carrier_count_map"]

global_train_delay_mean = artifacts["global_train_delay_mean"]

origin_map = encoding_maps.get("origin_map", {})
destination_map = encoding_maps.get("destination_map", {})
route_map = encoding_maps.get("route_map", {})
carrier_map = encoding_maps.get("carrier_map", {})
part_of_day_map = encoding_maps.get("part_of_day_map", {})

# SQLite reference data
conn_app = sqlite3.connect("aerodelay.db")

def table_exists(connection, table_name):
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name=?
    """
    cur = connection.cursor()
    cur.execute(q, (table_name,))
    return cur.fetchone() is not None

try:
    airports_sql = pd.read_sql("SELECT * FROM airports_raw", conn_app) if table_exists(conn_app, "airports_raw") else pd.DataFrame()
except Exception:
    airports_sql = pd.DataFrame()

try:
    flights_clean_sql = pd.read_sql("SELECT * FROM flights_clean", conn_app) if table_exists(conn_app, "flights_clean") else pd.DataFrame()
except Exception:
    flights_clean_sql = pd.DataFrame()

# Full airport master table
# - use existing airport_master if already loaded
# - otherwise load from OurAirports CSV and cache in SQLite

airport_master_sql = pd.DataFrame()

try:
    if table_exists(conn_app, "airport_master"):
        airport_master_sql = pd.read_sql("SELECT * FROM airport_master", conn_app)
    else:
        airport_master_sql = pd.read_csv("https://ourairports.com/data/airports.csv")

        airport_master_sql = airport_master_sql[[
            "ident",
            "type",
            "name",
            "latitude_deg",
            "longitude_deg",
            "municipality",
            "iso_country",
            "iata_code"
        ]].copy()

        airport_master_sql["ident"] = airport_master_sql["ident"].fillna("").astype(str).str.upper().str.strip()
        airport_master_sql["iata_code"] = airport_master_sql["iata_code"].fillna("").astype(str).str.upper().str.strip()
        airport_master_sql["name"] = airport_master_sql["name"].fillna("").astype(str).str.strip()
        airport_master_sql["municipality"] = airport_master_sql["municipality"].fillna("").astype(str).str.strip()
        airport_master_sql["iso_country"] = airport_master_sql["iso_country"].fillna("").astype(str).str.upper().str.strip()

        airport_master_sql = airport_master_sql[
            airport_master_sql["type"].isin(["large_airport", "medium_airport", "small_airport"])
        ].copy()

        airport_master_sql = airport_master_sql.drop_duplicates(subset=["ident"]).reset_index(drop=True)
        airport_master_sql.to_sql("airport_master", conn_app, if_exists="replace", index=False)

except Exception as e:
    print(f"airport_master load failed: {e}")
    airport_master_sql = pd.DataFrame()

# Build display names for dropdown
if not airport_master_sql.empty:
    airport_master_sql["display_name"] = airport_master_sql.apply(
        lambda r: (
            f"{r['name']} | "
            f"IATA: {r['iata_code'] if str(r['iata_code']).strip() else 'N/A'} | "
            f"ICAO: {r['ident']} | "
            f"{r['municipality']}, {r['iso_country']}"
        ),
        axis=1
    )
else:
    print("airport_master_sql is empty, falling back to default airport choices.")

# Airport lookup helpers
DEFAULT_IATA_TO_ICAO = {
    "JFK": "KJFK",
    "LAX": "KLAX",
    "ORD": "KORD",
    "MIA": "KMIA",
    "ATL": "KATL",
    "LHR": "EGLL",
    "CDG": "LFPG",
    "MAD": "LEMD",
    "PEK": "ZBAA",
    "PKX": "ZBAD"
}

if not airports_sql.empty and {"iata_code", "icao_code"}.issubset(airports_sql.columns):
    sql_iata_to_icao = {
        str(row["iata_code"]).upper(): str(row["icao_code"]).upper()
        for _, row in airports_sql.dropna(subset=["iata_code", "icao_code"]).iterrows()
    }
else:
    sql_iata_to_icao = {}

if not airport_master_sql.empty and {"iata_code", "ident"}.issubset(airport_master_sql.columns):
    master_iata_to_icao = {
        str(row["iata_code"]).upper(): str(row["ident"]).upper()
        for _, row in airport_master_sql.iterrows()
        if str(row["iata_code"]).strip()
    }
else:
    master_iata_to_icao = {}

IATA_TO_ICAO = {**DEFAULT_IATA_TO_ICAO, **sql_iata_to_icao, **master_iata_to_icao}

def haversine_km(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return round(6371 * c, 2)

def parse_time_to_hour_float(time_str):
    s = str(time_str).strip().upper()

    if s in ("24:00", "2400"):
        return 0.0, 0, 0

    for fmt in ("%H:%M", "%H%M", "%I:%M %p"):
        try:
            dt = datetime.strptime(s, fmt)
            hour_float = dt.hour + dt.minute / 60.0
            return hour_float, dt.hour, dt.minute
        except ValueError:
            continue

    raise ValueError("Invalid time. Use formats like 14:30, 2400, or 2:30 PM.")

def get_part_of_day(hour):
    if pd.isna(hour):
        return "unknown"
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    return "night"

def get_airport_row_local(iata_code):
    if airports_sql.empty:
        return None

    temp = airports_sql[airports_sql["iata_code"].astype(str).str.upper() == iata_code.upper()]
    if temp.empty:
        return None

    row = temp.iloc[0].to_dict()
    return {
        "iata_code": str(row.get("iata_code", "")).upper(),
        "icao_code": str(row.get("icao_code", "")).upper(),
        "airport_name": row.get("airport_name"),
        "latitude": pd.to_numeric(row.get("latitude"), errors="coerce"),
        "longitude": pd.to_numeric(row.get("longitude"), errors="coerce"),
        "country_code": row.get("country_code"),
        "municipality": row.get("municipality")
    }

def fetch_airport_from_airportdb(iata_or_icao_code):
    code = str(iata_or_icao_code).strip().upper()

    if len(code) == 3:
        code = IATA_TO_ICAO.get(code, code)

    url = f"https://airportdb.io/api/v1/airport/{code}"
    params = {"apiToken": AIRPORTDB_API_TOKEN}

    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()

        lat_val = data.get("latitude", data.get("lat", data.get("latitude_deg")))
        lon_val = data.get("longitude", data.get("lon", data.get("longitude_deg")))

        return {
            "iata_code": str(data.get("iata_code", "")).upper(),
            "icao_code": str(data.get("icao_code", code)).upper(),
            "airport_name": data.get("name"),
            "latitude": pd.to_numeric(lat_val, errors="coerce"),
            "longitude": pd.to_numeric(lon_val, errors="coerce"),
            "country_code": data.get("iso_country"),
            "municipality": data.get("municipality")
        }
    except Exception:
        return None

def resolve_airport_input(user_text):
    """
    Accepts:
    - full airport display string
    - IATA code (3 letters)
    - ICAO code (4 letters)
    - full/partial airport name
    """
    text_raw = str(user_text).strip()
    text = text_raw.upper()

    # First try airport_master if available
    if not airport_master_sql.empty:
        df = airport_master_sql.copy()

        exact_iata = df[df["iata_code"].astype(str).str.upper() == text]
        if not exact_iata.empty:
            return exact_iata.iloc[0].to_dict()

        exact_ident = df[df["ident"].astype(str).str.upper() == text]
        if not exact_ident.empty:
            return exact_ident.iloc[0].to_dict()

        exact_display = df[df["display_name"].astype(str).str.upper() == text]
        if not exact_display.empty:
            return exact_display.iloc[0].to_dict()

        exact_name = df[df["name"].astype(str).str.upper() == text]
        if not exact_name.empty:
            return exact_name.iloc[0].to_dict()

        partial_name = df[df["name"].astype(str).str.upper().str.contains(text, na=False)]
        if not partial_name.empty:
            return partial_name.iloc[0].to_dict()

    local_row = get_airport_row_local(text)
    if local_row is not None:
        return {
            "ident": local_row["icao_code"],
            "type": "",
            "name": local_row["airport_name"],
            "latitude_deg": local_row["latitude"],
            "longitude_deg": local_row["longitude"],
            "municipality": local_row["municipality"],
            "iso_country": local_row["country_code"],
            "iata_code": local_row["iata_code"]
        }

    db_row = fetch_airport_from_airportdb(text)
    if db_row is not None:
        return {
            "ident": db_row["icao_code"],
            "type": "",
            "name": db_row["airport_name"],
            "latitude_deg": db_row["latitude"],
            "longitude_deg": db_row["longitude"],
            "municipality": db_row["municipality"],
            "iso_country": db_row["country_code"],
            "iata_code": db_row["iata_code"]
        }

    raise ValueError(f"Could not resolve airport input: {user_text}")

# Live API helpers
def get_weather_by_coords(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_KEY,
        "units": "metric"
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        w = r.json()

        return {
            "temperature_c": pd.to_numeric(w.get("main", {}).get("temp"), errors="coerce"),
            "feels_like_c": pd.to_numeric(w.get("main", {}).get("feels_like"), errors="coerce"),
            "pressure_hpa": pd.to_numeric(w.get("main", {}).get("pressure"), errors="coerce"),
            "humidity_pct": pd.to_numeric(w.get("main", {}).get("humidity"), errors="coerce"),
            "wind_strength": pd.to_numeric(w.get("wind", {}).get("speed"), errors="coerce"),
            "wind_deg": pd.to_numeric(w.get("wind", {}).get("deg"), errors="coerce"),
            "visibility": pd.to_numeric(w.get("visibility"), errors="coerce"),
            "rain_occurrence": pd.to_numeric(w.get("rain", {}).get("1h", 0.0), errors="coerce"),
            "weather_main": (w.get("weather") or [{}])[0].get("main"),
            "weather_description": (w.get("weather") or [{}])[0].get("description")
        }
    except Exception:
        return {
            "temperature_c": train_medians.get("temperature_c", 15.0),
            "feels_like_c": train_medians.get("feels_like_c", 15.0),
            "pressure_hpa": train_medians.get("pressure_hpa", 1013.0),
            "humidity_pct": train_medians.get("humidity_pct", 60.0),
            "wind_strength": train_medians.get("wind_strength", 5.0),
            "wind_deg": train_medians.get("wind_deg", 180.0),
            "visibility": train_medians.get("visibility", 10000.0),
            "rain_occurrence": train_medians.get("rain_occurrence", 0.0),
            "weather_main": "UNKNOWN",
            "weather_description": "unavailable"
        }

def fetch_live_route_context(origin_iata, destination_iata):
    if not origin_iata or not destination_iata:
        return {
            "live_route_count": 0,
            "top_carrier": None,
            "statuses": []
        }

    url = "https://api.aviationstack.com/v1/flights"
    params = {
        "access_key": AVIATION_KEY,
        "dep_iata": origin_iata,
        "arr_iata": destination_iata,
        "limit": 100
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()

        if "error" in payload:
            return {
                "live_route_count": 0,
                "top_carrier": None,
                "statuses": []
            }

        data = payload.get("data", [])
        carriers = []
        statuses = []

        for flight in data:
            airline = flight.get("airline", {}) or {}
            status = flight.get("flight_status")
            airline_iata = airline.get("iata")
            if airline_iata:
                carriers.append(str(airline_iata).upper())
            if status:
                statuses.append(status)

        top_carrier = Counter(carriers).most_common(1)[0][0] if carriers else None

        return {
            "live_route_count": len(data),
            "top_carrier": top_carrier,
            "statuses": sorted(list(set(statuses)))
        }
    except Exception:
        return {
            "live_route_count": 0,
            "top_carrier": None,
            "statuses": []
        }

# Encoding / aggregate helpers
def stable_unknown_code(value, mapping, start_offset=1000):
    if value in mapping:
        return mapping[value]
    return start_offset + (sum(ord(ch) for ch in str(value)) % 10000)

def encode_with_fallback(value, mapping, feature_name=None, start_offset=1000):
    return stable_unknown_code(value, mapping, start_offset=start_offset)

def get_safe_mean(value, mapping, distance_km=None):
    if value in mapping:
        return mapping[value]

    base = float(global_train_delay_mean)

    if distance_km is None or pd.isna(distance_km):
        return base

    median_distance = train_medians.get("estimated_path_length_km", 3500.0)
    adj = (float(distance_km) - float(median_distance)) / 2000.0
    adj = max(min(adj, 8), -8)

    return base + adj

def get_safe_count(value, mapping, live_count=None):
    if value in mapping:
        return mapping[value]

    if live_count is not None and not pd.isna(live_count):
        return max(int(live_count), 0)

    return 0

def choose_carrier(origin, destination, live_ctx, user_carrier=None):
    if user_carrier and str(user_carrier).strip():
        return str(user_carrier).strip().upper(), "user"

    if live_ctx["top_carrier"]:
        return live_ctx["top_carrier"], "aviationstack"

    if not flights_clean_sql.empty and {"route", "airline_id"}.issubset(flights_clean_sql.columns):
        route = f"{origin}_{destination}"
        subset = flights_clean_sql[flights_clean_sql["route"] == route]
        if not subset.empty:
            mode_series = subset["airline_id"].dropna().astype(str).str.upper().mode()
            if len(mode_series) > 0:
                return mode_series.iloc[0], "historical_route"

    return "UNKNOWN", "fallback"

explainer = shap.TreeExplainer(rf_model)

# Build model row
def build_prediction_row(origin, destination, scheduled_date, scheduled_time, user_carrier=None):
    origin_match = resolve_airport_input(origin)
    destination_match = resolve_airport_input(destination)

    origin_code = origin_match["iata_code"] if str(origin_match.get("iata_code", "")).strip() else str(origin_match["ident"]).upper()
    destination_code = destination_match["iata_code"] if str(destination_match.get("iata_code", "")).strip() else str(destination_match["ident"]).upper()

    origin_iata_for_api = str(origin_match.get("iata_code", "")).upper().strip()
    destination_iata_for_api = str(destination_match.get("iata_code", "")).upper().strip()

    if origin_code == destination_code:
        raise ValueError("Origin and destination must be different.")

    if scheduled_date and str(scheduled_date).strip():
        date_obj = pd.to_datetime(str(scheduled_date)).to_pydatetime()
    else:
        date_obj = datetime.now(timezone.utc)

    hour_float, hour_int, minute_int = parse_time_to_hour_float(scheduled_time)

    scheduled_dt = datetime(
        year=date_obj.year,
        month=date_obj.month,
        day=date_obj.day,
        hour=hour_int,
        minute=minute_int,
        tzinfo=timezone.utc
    )

    part_of_day = get_part_of_day(hour_int)
    route = f"{origin_code}_{destination_code}"

    origin_airport = {
        "iata_code": origin_iata_for_api,
        "icao_code": str(origin_match["ident"]).upper(),
        "airport_name": origin_match["name"],
        "latitude": pd.to_numeric(origin_match["latitude_deg"], errors="coerce"),
        "longitude": pd.to_numeric(origin_match["longitude_deg"], errors="coerce"),
        "country_code": origin_match["iso_country"],
        "municipality": origin_match["municipality"]
    }

    destination_airport = {
        "iata_code": destination_iata_for_api,
        "icao_code": str(destination_match["ident"]).upper(),
        "airport_name": destination_match["name"],
        "latitude": pd.to_numeric(destination_match["latitude_deg"], errors="coerce"),
        "longitude": pd.to_numeric(destination_match["longitude_deg"], errors="coerce"),
        "country_code": destination_match["iso_country"],
        "municipality": destination_match["municipality"]
    }

    route_distance = haversine_km(
        origin_airport["latitude"], origin_airport["longitude"],
        destination_airport["latitude"], destination_airport["longitude"]
    )

    if pd.isna(route_distance):
        if not flights_clean_sql.empty and "route" in flights_clean_sql.columns and "estimated_path_length_km" in flights_clean_sql.columns:
            subset = flights_clean_sql[flights_clean_sql["route"] == route]
            if not subset.empty:
                route_distance = pd.to_numeric(subset["estimated_path_length_km"], errors="coerce").median()

    if pd.isna(route_distance):
        route_distance = train_medians.get("estimated_path_length_km", 3500.0)

    weather = get_weather_by_coords(origin_airport["latitude"], origin_airport["longitude"])
    live_ctx = fetch_live_route_context(origin_iata_for_api, destination_iata_for_api)
    carrier, carrier_source = choose_carrier(origin_code, destination_code, live_ctx, user_carrier=user_carrier)

    encoded_origin = encode_with_fallback(origin_code, origin_map, "encoded_origin", start_offset=1000)
    encoded_destination = encode_with_fallback(destination_code, destination_map, "encoded_destination", start_offset=2000)
    encoded_route = encode_with_fallback(route, route_map, "encoded_route", start_offset=3000)
    encoded_carrier = encode_with_fallback(carrier, carrier_map, "encoded_carrier", start_offset=4000)
    encoded_part_of_day = encode_with_fallback(part_of_day, part_of_day_map, "encoded_part_of_day", start_offset=5000)

    origin_avg_delay_safe = get_safe_mean(origin_code, origin_mean_map, distance_km=route_distance)
    destination_avg_delay_safe = get_safe_mean(destination_code, destination_mean_map, distance_km=route_distance)
    route_avg_delay_safe = get_safe_mean(route, route_mean_map, distance_km=route_distance)
    carrier_avg_delay_safe = get_safe_mean(carrier, carrier_mean_map, distance_km=route_distance)

    origin_flight_count_safe = get_safe_count(origin_code, origin_count_map, live_count=live_ctx["live_route_count"])
    destination_flight_count_safe = get_safe_count(destination_code, destination_count_map, live_count=live_ctx["live_route_count"])
    route_flight_count_safe = get_safe_count(route, route_count_map, live_count=live_ctx["live_route_count"])
    carrier_flight_count_safe = get_safe_count(carrier, carrier_count_map, live_count=live_ctx["live_route_count"])

    high_wind_flag = int(float(weather["wind_strength"]) >= 10) if not pd.isna(weather["wind_strength"]) else 0
    rain_flag = int(float(weather["rain_occurrence"]) > 0) if not pd.isna(weather["rain_occurrence"]) else 0
    low_visibility_flag = int(float(weather["visibility"]) < 5000) if not pd.isna(weather["visibility"]) else 0
    weather_severity_score = high_wind_flag + rain_flag + low_visibility_flag

    feature_row = {
        "departure_hour": hour_int,
        "departure_minute": minute_int,
        "departure_hour_float": hour_float,
        "day_of_week": scheduled_dt.weekday(),
        "month": scheduled_dt.month,
        "day_of_month": scheduled_dt.day,
        "is_weekend": int(scheduled_dt.weekday() in [5, 6]),
        "encoded_origin": encoded_origin,
        "encoded_destination": encoded_destination,
        "encoded_route": encoded_route,
        "encoded_carrier": encoded_carrier,
        "encoded_part_of_day": encoded_part_of_day,
        "estimated_path_length_km": route_distance,
        "origin_avg_delay_safe": origin_avg_delay_safe,
        "destination_avg_delay_safe": destination_avg_delay_safe,
        "route_avg_delay_safe": route_avg_delay_safe,
        "carrier_avg_delay_safe": carrier_avg_delay_safe,
        "origin_flight_count_safe": origin_flight_count_safe,
        "destination_flight_count_safe": destination_flight_count_safe,
        "route_flight_count_safe": route_flight_count_safe,
        "carrier_flight_count_safe": carrier_flight_count_safe,
        "temperature_c": weather["temperature_c"],
        "feels_like_c": weather["feels_like_c"],
        "pressure_hpa": weather["pressure_hpa"],
        "humidity_pct": weather["humidity_pct"],
        "wind_strength": weather["wind_strength"],
        "wind_deg": weather["wind_deg"],
        "visibility": weather["visibility"],
        "rain_occurrence": weather["rain_occurrence"],
        "high_wind_flag": high_wind_flag,
        "rain_flag": rain_flag,
        "low_visibility_flag": low_visibility_flag,
        "weather_severity_score": weather_severity_score
    }

    final_row = {}
    for col in model_feature_columns:
        value = feature_row.get(col, train_medians.get(col, 0))
        if pd.isna(value):
            value = train_medians.get(col, 0)
        final_row[col] = value

    X_pred = pd.DataFrame([final_row])

    metadata = {
        "origin": origin_code,
        "destination": destination_code,
        "route": route,
        "carrier": carrier,
        "carrier_source": carrier_source,
        "scheduled_dt_utc": scheduled_dt,
        "origin_airport_name": origin_airport.get("airport_name", origin_code),
        "destination_airport_name": destination_airport.get("airport_name", destination_code),
        "route_distance_km": route_distance,
        "live_route_count": live_ctx["live_route_count"],
        "live_statuses": live_ctx["statuses"],
        "weather": weather
    }

    return X_pred, metadata

def debug_compare_routes():
    pek_row, _ = build_prediction_row("JFK", "PEK", "2026-04-04", "14:30", "")
    pkx_row, _ = build_prediction_row("JFK", "PKX", "2026-04-04", "14:30", "")

    compare = pd.concat(
        [
            pek_row.T.rename(columns={0: "PEK"}),
            pkx_row.T.rename(columns={0: "PKX"})
        ],
        axis=1
    )

    return compare.loc[
        [
            "encoded_destination",
            "encoded_route",
            "estimated_path_length_km",
            "destination_avg_delay_safe",
            "route_avg_delay_safe",
            "destination_flight_count_safe",
            "route_flight_count_safe",
            "encoded_carrier"
        ]
    ]

# Prediction function for Gradio
def predict_flight_delay(origin_input, destination_input, scheduled_date, scheduled_time, airline_input):
    try:
        X_pred, meta = build_prediction_row(
            origin=origin_input,
            destination=destination_input,
            scheduled_date=scheduled_date,
            scheduled_time=scheduled_time,
            user_carrier=airline_input
        )

        delay_minutes = float(rf_model.predict(X_pred)[0])
        delay_prob = float(log_model.predict_proba(X_pred)[0][1])

        delay_minutes = round(max(delay_minutes, 0), 1)

        if delay_minutes < 15:
            status = "On Time"
        elif delay_minutes < 45:
            status = "Minor Delay"
        else:
            status = "Significant Delay"

        shap_values = explainer.shap_values(X_pred)
        local_importance = pd.DataFrame({
            "feature": model_feature_columns,
            "shap_value": shap_values[0]
        })
        local_importance["abs_shap"] = local_importance["shap_value"].abs()
        local_importance = local_importance.sort_values("abs_shap", ascending=False).head(5)

        top_factors_text = "\n".join(
            [
                f"- **{row.feature}**: {round(row.shap_value, 2)}"
                for row in local_importance.itertuples(index=False)
            ]
        )

        weather = meta["weather"]
        live_statuses = ", ".join(meta["live_statuses"]) if meta["live_statuses"] else "none found"

        output = f"""
## {status}

**Predicted delay:** {delay_minutes} minutes
**Delay risk probability (>15 min):** {round(delay_prob * 100, 1)}%

**Origin:** {meta["origin"]} - {meta["origin_airport_name"]}
**Destination:** {meta["destination"]} - {meta["destination_airport_name"]}
**Scheduled departure (UTC):** {meta["scheduled_dt_utc"].strftime("%Y-%m-%d %H:%M")}
**Route distance:** {round(meta["route_distance_km"], 1)} km

**Carrier used for prediction:** {meta["carrier"]} ({meta["carrier_source"]})
**Live flights found on this route now:** {meta["live_route_count"]}
**Live route statuses seen:** {live_statuses}

### Live origin weather
- Temperature: {round(float(weather['temperature_c']), 1) if pd.notna(weather['temperature_c']) else 'N/A'} °C
- Wind: {round(float(weather['wind_strength']), 1) if pd.notna(weather['wind_strength']) else 'N/A'} m/s
- Rain (1h): {round(float(weather['rain_occurrence']), 2) if pd.notna(weather['rain_occurrence']) else 'N/A'}
- Visibility: {int(weather['visibility']) if pd.notna(weather['visibility']) else 'N/A'} m
- Condition: {weather.get('weather_main', 'UNKNOWN')} / {weather.get('weather_description', 'unavailable')}

### Top SHAP drivers
{top_factors_text}
"""
        return output

    except Exception as e:
        return f"Prediction failed: {e}"

# UI choices
if not airport_master_sql.empty and "display_name" in airport_master_sql.columns:
    airport_choices = airport_master_sql["display_name"].dropna().astype(str).tolist()
else:
    airport_choices = sorted(DEFAULT_IATA_TO_ICAO.keys())

today_utc_str = datetime.now(timezone.utc).date().isoformat()

default_origin = "John F Kennedy International Airport | IATA: JFK | ICAO: KJFK | New York, US"
default_destination = "Adolfo Suárez Madrid–Barajas Airport | IATA: MAD | ICAO: LEMD | Madrid, ES"

if airport_choices:
    if default_origin not in airport_choices:
        default_origin = airport_choices[0]
    if default_destination not in airport_choices:
        default_destination = airport_choices[1] if len(airport_choices) > 1 else airport_choices[0]

interface = gr.Interface(
    fn=predict_flight_delay,
    inputs=[
        gr.Dropdown(
            choices=airport_choices,
            value=default_origin if airport_choices else "JFK",
            allow_custom_value=True,
            label="Origin Airport (full name, IATA, or ICAO)"
        ),
        gr.Dropdown(
            choices=airport_choices,
            value=default_destination if airport_choices else "MAD",
            allow_custom_value=True,
            label="Destination Airport (full name, IATA, or ICAO)"
        ),
        gr.Textbox(
            value=today_utc_str,
            label="Scheduled Date (UTC, YYYY-MM-DD)"
        ),
        gr.Textbox(
            value="14:30",
            label="Scheduled Time (UTC, e.g. 14:30, 2400, 2:30 PM)"
        ),
        gr.Textbox(
            value="",
            label="Airline IATA (optional, e.g. AA, DL, BA)"
        )
    ],
    outputs=gr.Markdown(label="Prediction + Explanation"),
    title="AeroDelay AI - Flight Delay Predictor",
    description=(
        "Loads the saved model from your notebook, then enriches each prediction with live AirportDB, "
        "OpenWeather, and Aviationstack data. You can select a full airport name or type an IATA/ICAO code."
    ),
    examples=[
        ["John F Kennedy International Airport | IATA: JFK | ICAO: KJFK | New York, US", "Adolfo Suárez Madrid–Barajas Airport | IATA: MAD | ICAO: LEMD | Madrid, ES", today_utc_str, "14:30", ""],
        ["JFK", "PEK", today_utc_str, "14:30", ""],
        ["JFK", "PKX", today_utc_str, "14:30", ""],
        ["KJFK", "LEMD", today_utc_str, "24:00", ""],
        ["JFK", "LAX", today_utc_str, "14:30", "AA"]
    ],
    flagging_mode="never"
)

interface.launch(share=True)