"""AeroDelay AI inference helpers (shared by the Gradio app and the notebook).

Weather path for a prediction (all sources return the SAME schema/units as the training weather):
  1. Open-Meteo forecast API (ECMWF IFS, keyless) for -7 .. +15 days from today.   -> label "forecast"
  2. Open-Meteo archive API (same ECMWF IFS model) for older dates.                   -> label "historical"
  3. Beyond the forecast window: same calendar date one year earlier (+/-3 days)
     from the archive API, averaged by hour.                                          -> label "seasonal analog (NOT a forecast)"
  4. If Open-Meteo is unreachable: OpenWeather current weather, only if OPENWEATHER_API_KEY is set
     in the environment (key is never printed/logged).                                 -> label "OpenWeather current"
  5. Last resort: training medians.                                                    -> label "weather forecast unavailable, typical values used"
"""
import os, math, json, sqlite3, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, requests, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
OM_VARS = "temperature_2m,apparent_temperature,surface_pressure,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,precipitation,cloud_cover"
FORECAST_DAYS = 15

def load(path=HERE):
    A = joblib.load(os.path.join(path, "aerodelay_artifacts.pkl"))
    rf = joblib.load(os.path.join(path, "aerodelay_rf_model.pkl"))
    clf = joblib.load(os.path.join(path, "aerodelay_log_model.pkl"))
    con = sqlite3.connect(os.path.join(path, "aerodelay.db"))
    airports = pd.read_sql("SELECT * FROM airport_master", con)
    fc = pd.read_sql("SELECT origin,destination,route,airline_id FROM flights_clean", con)
    con.close()
    for c in ("iata_code", "ident", "name"):
        airports[c + "_u"] = airports[c].astype(str).str.upper().str.strip()
    return dict(A=A, rf=rf, clf=clf, airports=airports, flights=fc)

# ---------------------------------------------------------------- airports / time
def resolve_airport(text, ctx):
    t = str(text or "").strip().upper()
    if not t: raise ValueError("Please enter an airport (IATA like ATL, ICAO like KATL, or part of the name).")
    ap = ctx["airports"]
    for col in ("iata_code_u", "ident_u", "name_u"):
        m = ap[ap[col] == t]
        if not m.empty: return m.sort_values("type", key=lambda s: s.map({"large_airport": 0, "medium_airport": 1, "small_airport": 2}).fillna(3)).iloc[0].to_dict()
    m = ap[ap["name_u"].str.contains(t, na=False, regex=False)]
    if not m.empty:
        m = m.assign(_r=m["type"].map({"large_airport": 0, "medium_airport": 1, "small_airport": 2}).fillna(3), _i=(m["iata_code_u"] == "").astype(int))
        return m.sort_values(["_i", "_r"]).iloc[0].to_dict()
    raise ValueError(f"Could not find airport '{text}'.")

def airport_code(row):
    return row["iata_code_u"] if row.get("iata_code_u") else row["ident_u"]

def airport_tz(row, A):
    code = airport_code(row)
    if code in A["airport_tz"]: return A["airport_tz"][code], "training table"
    try:
        from timezonefinder import TimezoneFinder
        z = TimezoneFinder().timezone_at(lat=float(row["latitude_deg"]), lng=float(row["longitude_deg"]))
        if z: return z, "timezonefinder"
    except Exception: pass
    off = round(float(row["longitude_deg"]) / 15.0)
    return f"Etc/GMT{-off:+d}".replace("+-", "-") if off else "UTC", "approximate (longitude/15)"

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))

def parse_time(s):
    s = str(s or "").strip().upper().replace(" ", "")
    if not s: raise ValueError("Please enter a departure time like 14:30 or 2:30PM.")
    for fmt in ("%H:%M", "%H%M", "%I:%M%p", "%I%p"):
        try:
            t = datetime.strptime(s, fmt); return t.hour, t.minute
        except ValueError: pass
    raise ValueError("Invalid time. Use 14:30, 1430 or 2:30PM (local time at the departure airport).")

# ---------------------------------------------------------------- weather
WX_UNAVAILABLE = "weather forecast unavailable, typical values used"
WX_COLS = ["temperature_c", "feels_like_c", "pressure_hpa", "humidity_pct", "wind_strength", "wind_deg", "visibility", "rain_occurrence", "cloud_cover_pct", "wind_gust_ms"]

def _om_to_schema(h):
    """Open-Meteo hourly frame -> training schema/units (km/h -> m/s, precipitation mm/h)."""
    return pd.DataFrame({"time": pd.to_datetime(h["time"]), "temperature_c": h["temperature_2m"], "feels_like_c": h["apparent_temperature"], "pressure_hpa": h["surface_pressure"],
        "humidity_pct": h["relative_humidity_2m"], "wind_strength": pd.Series(h["wind_speed_10m"], dtype=float) / 3.6, "wind_deg": h["wind_direction_10m"], "visibility": h["visibility"],
        "rain_occurrence": h["precipitation"], "cloud_cover_pct": h["cloud_cover"], "wind_gust_ms": pd.Series(h["wind_gusts_10m"], dtype=float) / 3.6})

def _om(url, params, retries=2):
    """Open-Meteo request with 2 retries (short backoff) on 5xx / network errors. Errors never include the URL."""
    last = "unknown error"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            elif r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            else:
                j = r.json()
                if "hourly" not in j: raise RuntimeError(str(j.get("reason", "no hourly data"))[:60])
                return _om_to_schema(j["hourly"])
        except (requests.ConnectionError, requests.Timeout) as e:
            last = type(e).__name__
        if attempt < retries: time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(last)

def get_weather(lat, lon, when_utc, A):
    """when_utc: tz-aware UTC datetime. Returns (dict in training schema, label)."""
    hour = pd.Timestamp(when_utc).tz_convert("UTC").tz_localize(None).floor("h")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    base = dict(latitude=lat, longitude=lon, hourly=OM_VARS, timezone="UTC", models="ecmwf_ifs")
    days_ahead = (hour - now).total_seconds() / 86400
    try:
        if -7 <= days_ahead <= FORECAST_DAYS - 0.5:
            df = _om("https://api.open-meteo.com/v1/forecast", {**base, "past_days": 7, "forecast_days": FORECAST_DAYS + 1})
            row = df[df.time == hour]
            if not row.empty and row[WX_COLS[:6]].notna().all(axis=1).iloc[0]:
                return _fill(row.iloc[0], A), ("current conditions (Open-Meteo, now)" if abs(days_ahead) < 1/24 else ("forecast (Open-Meteo ECMWF IFS)" if days_ahead > 0 else "recent observed/analysis (Open-Meteo)"))
        elif days_ahead < -7:
            d0 = hour.strftime("%Y-%m-%d")
            df = _om("https://archive-api.open-meteo.com/v1/archive", {**base, "start_date": d0, "end_date": d0})
            row = df[df.time == hour]
            if not row.empty and row[WX_COLS[:6]].notna().all(axis=1).iloc[0]:
                return _fill(row.iloc[0], A), "historical (Open-Meteo archive)"
        # beyond forecast window (or empty hour): seasonal analog
        an = hour - pd.Timedelta(days=365)
        df = _om("https://archive-api.open-meteo.com/v1/archive", {**base, "start_date": (an - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), "end_date": (an + pd.Timedelta(days=3)).strftime("%Y-%m-%d")})
        df = df[df.time.dt.hour == hour.hour]
        if df.empty: raise RuntimeError("no analog data")
        s = df[WX_COLS].astype(float).mean()
        s["wind_deg"] = df["wind_deg"].astype(float).median()
        return _fill(s, A), "SEASONAL ANALOG: same date last year +/-3 days averaged. NOT a forecast"
    except Exception as e:
        pass
    key = os.environ.get("OPENWEATHER_API_KEY")
    if key and abs((hour - now).total_seconds()) < 86400 * 1.5:
        try:
            r = requests.get("https://api.openweathermap.org/data/2.5/weather", params={"lat": lat, "lon": lon, "appid": key, "units": "metric"}, timeout=15); r.raise_for_status(); w = r.json()
            s = pd.Series(dict(temperature_c=w["main"]["temp"], feels_like_c=w["main"]["feels_like"], pressure_hpa=w["main"].get("grnd_level", w["main"]["pressure"]), humidity_pct=w["main"]["humidity"],
                wind_strength=w.get("wind", {}).get("speed", np.nan), wind_deg=w.get("wind", {}).get("deg", np.nan), visibility=w.get("visibility", np.nan),
                rain_occurrence=w.get("rain", {}).get("1h", 0.0), cloud_cover_pct=w.get("clouds", {}).get("all", np.nan), wind_gust_ms=w.get("wind", {}).get("gust", np.nan)))
            return _fill(s, A), "current conditions (OpenWeather, Open-Meteo unavailable)"
        except Exception: pass
    return {c: A["train_medians"].get(c, 0.0) for c in WX_COLS}, WX_UNAVAILABLE

def _fill(s, A):
    out = {}
    for c in WX_COLS:
        v = s.get(c, np.nan)
        out[c] = float(v) if pd.notna(v) else float(A["train_medians"].get(c, 0.0))
    return out

def weather_flags(w, prefix=""):
    hw = int(w["wind_strength"] >= 10); rn = int(w["rain_occurrence"] > 0); lv = int(w["visibility"] < 5000)
    return hw, rn, lv

# ---------------------------------------------------------------- features
def build_row(origin, destination, date_str, time_str, carrier, ctx):
    A = ctx["A"]; warnings = []
    o = resolve_airport(origin, ctx); d = resolve_airport(destination, ctx)
    oc, dc = airport_code(o), airport_code(d)
    if oc == dc: raise ValueError("Origin and destination must be different.")
    otz, otz_src = airport_tz(o, A); dtz, _ = airport_tz(d, A)
    if otz_src.startswith("approx"): warnings.append(f"Time zone for {oc} is approximate.")
    hh, mm = parse_time(time_str)
    day = pd.to_datetime(date_str).date() if str(date_str or "").strip() else datetime.now(ZoneInfo(otz)).date()
    dep_local = datetime(day.year, day.month, day.day, hh, mm, tzinfo=ZoneInfo(otz)); dep_utc = dep_local.astimezone(timezone.utc)
    dist = haversine(float(o["latitude_deg"]), float(o["longitude_deg"]), float(d["latitude_deg"]), float(d["longitude_deg"]))
    route = f"{oc}_{dc}"
    in_o, in_d = oc in A["origin_mean_map"], dc in A["destination_mean_map"]
    if not in_o: warnings.append(f"{oc} is not in the training data (US domestic airports with 300+ flights). Origin effects use the overall average.")
    if not in_d: warnings.append(f"{dc} is not in the training data (US domestic airports with 300+ flights). Destination effects use the overall average.")
    if route not in A["route_mean_map"]: warnings.append(f"Route {route} was not seen in training. Route effect uses the overall average.")
    if dist > 6000 or not (in_o and in_d): warnings.append("Outside the US-domestic training coverage: treat this prediction as low confidence.")
    # carrier
    carrier = str(carrier or "").strip().upper(); csrc = "you entered"
    if not carrier:
        f = ctx["flights"]; sub = f[f.route == route].airline_id
        csrc = "most common on this route in training data"
        if sub.empty: sub = f[f.origin == oc].airline_id; csrc = "route not in training data; most common carrier at the origin (a guess, enter the airline if you know it)"
        if not len(sub): csrc = "unknown"
        carrier = sub.mode().iloc[0] if len(sub) else "UNKNOWN"
    if carrier not in A["carrier_mean_map"]:
        warnings.append(f"Carrier '{carrier}' is not in the training data (BTS US carriers: {', '.join(A['known_carriers'])}). Carrier effects use the overall average.")
    # weather: origin at departure hour, destination at scheduled arrival hour
    blk = A["route_block_minutes"].get(route, A["block_minutes_fit"][0] * dist + A["block_minutes_fit"][1]); arr_utc = dep_utc + timedelta(minutes=float(blk))
    wo, wo_src = get_weather(float(o["latitude_deg"]), float(o["longitude_deg"]), dep_utc, A)
    wd, wd_src = get_weather(float(d["latitude_deg"]), float(d["longitude_deg"]), arr_utc, A)
    hw, rn, lv = weather_flags(wo); dhw, drn, dlv = weather_flags({"wind_strength": wd["wind_strength"], "rain_occurrence": wd["rain_occurrence"], "visibility": wd["visibility"]})
    wk = int(dep_utc.weekday() >= 5); ldh = hh + mm / 60.0
    arr_local = arr_utc.astimezone(ZoneInfo(dtz))
    g = float(A["global_train_delay_mean"])
    f = {"departure_hour": dep_utc.hour, "departure_minute": dep_utc.minute, "departure_hour_float": dep_utc.hour + dep_utc.minute / 60.0, "day_of_week": dep_utc.weekday(), "is_weekend": wk,
         "encoded_origin": ctx["E"]["origin_map"].get(oc, -1), "encoded_destination": ctx["E"]["destination_map"].get(dc, -1), "encoded_route": ctx["E"]["route_map"].get(route, -1),
         "encoded_carrier": ctx["E"]["carrier_map"].get(carrier, -1), "encoded_part_of_day": ctx["E"]["part_of_day_map"].get(_pod(dep_utc.hour), -1), "estimated_path_length_km": dist,
         "origin_avg_delay_safe": A["origin_mean_map"].get(oc, g), "destination_avg_delay_safe": A["destination_mean_map"].get(dc, g), "route_avg_delay_safe": A["route_mean_map"].get(route, g), "carrier_avg_delay_safe": A["carrier_mean_map"].get(carrier, g),
         "origin_flight_count_safe": A["origin_count_map"].get(oc, 0), "destination_flight_count_safe": A["destination_count_map"].get(dc, 0), "route_flight_count_safe": A["route_count_map"].get(route, 0), "carrier_flight_count_safe": A["carrier_count_map"].get(carrier, 0),
         "high_wind_flag": hw, "rain_flag": rn, "low_visibility_flag": lv, "weather_severity_score": hw + rn + lv,
         "local_departure_hour_float": ldh, "origin_hour_flights": A["congestion_origin"].get(f"{oc}|{hh}|{wk}", A["train_medians"]["origin_hour_flights"]),
         "dest_hour_arrivals": A["congestion_dest"].get(f"{dc}|{arr_local.hour}|{int(arr_utc.weekday() >= 5)}", A["train_medians"]["dest_hour_arrivals"]),
         "dest_temperature_c": wd["temperature_c"], "dest_wind_strength": wd["wind_strength"], "dest_visibility": wd["visibility"], "dest_rain_occurrence": wd["rain_occurrence"], "dest_cloud_cover_pct": wd["cloud_cover_pct"], "dest_weather_severity_score": dhw + drn + dlv}
    f.update({k: wo[k] for k in WX_COLS})
    row = {c: f.get(c, A["train_medians"].get(c, 0)) for c in A["model_feature_columns"]}
    X = pd.DataFrame([row])
    meta = dict(origin=oc, destination=dc, origin_name=o["name"], destination_name=d["name"], route=route, carrier=carrier, carrier_source=csrc, dep_local=dep_local, dep_utc=dep_utc, distance_km=dist, block_minutes=float(blk),
                weather_origin=wo, weather_origin_src=wo_src, weather_dest=wd, weather_dest_src=wd_src, warnings=warnings, origin_tz=otz)
    return X, meta

def _pod(h): return "morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 17 else "evening" if 17 <= h < 21 else "night"

def band(p):
    return ("LOW", "<20%") if p < 0.20 else (("ELEVATED", "20-35%") if p < 0.35 else ("HIGH", ">=35%"))

def predict(origin, destination, date_str, time_str, carrier, ctx, explainer=None, top_k=5):
    X, meta = build_row(origin, destination, date_str, time_str, carrier, ctx)
    minutes = float(ctx["rf"].predict(X)[0]); prob = float(ctx["clf"].predict_proba(X)[0][1])   # real classifier: no constant fallback
    out = dict(minutes=minutes, risk=prob, band=band(prob), meta=meta, X=X)
    if explainer is not None:
        sv = explainer.shap_values(X)[0]
        s = pd.DataFrame({"feature": X.columns, "shap_minutes": sv, "value": X.iloc[0].values}); s["abs"] = s.shap_minutes.abs()
        out["shap"] = s.sort_values("abs", ascending=False).head(top_k).drop(columns="abs").reset_index(drop=True)
        out["shap_base"] = float(np.ravel(explainer.expected_value)[0])
    return out

def setup(path=HERE):
    import shap
    ctx = load(path); ctx["E"] = joblib.load(os.path.join(path, "aerodelay_encoding_maps.pkl")); ctx["explainer"] = shap.TreeExplainer(ctx["rf"]); return ctx
