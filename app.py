"""AeroDelay AI (retrained) - Gradio app. Run: python app.py   (needs the files next to it)."""
import os, gradio as gr
from datetime import datetime
import aerodelay_infer as ai

CTX = ai.setup(os.path.dirname(os.path.abspath(__file__)))
M = CTX["A"]["metrics"]

NL = chr(10)          # real line breaks (no backslash escapes in the output text)
ICON = {"LOW": chr(0x1F7E2), "ELEVATED": chr(0x1F7E1), "HIGH": chr(0x1F534)}

def wx_line(w, src):
    if src == ai.WX_UNAVAILABLE:
        return ai.WX_UNAVAILABLE
    return f"{w['temperature_c']:.1f} C, wind {w['wind_strength']:.1f} m/s, visibility {w['visibility']/1000:.1f} km, rain {w['rain_occurrence']:.1f} mm/h, cloud {w['cloud_cover_pct']:.0f}% ({src})"

def run(origin, destination, date_str, time_str, carrier):
    try:
        r = ai.predict(origin, destination, date_str, time_str, carrier, CTX, CTX["explainer"])
    except Exception as e:
        return "**Could not predict:** " + str(e)[:200]
    m = r["meta"]; band, rng = r["band"]
    P = []   # paragraphs, joined with a blank line so every field renders on its own line
    P.append(f"## {ICON[band]} Delay risk: {band} ({r['risk']*100:.1f}%)")
    P.append(f"**Delay risk:** {r['risk']*100:.1f}% chance of arriving more than 15 min late. Band: {band} ({rng}).")
    P.append(f"**Typical-case delay estimate:** {max(r['minutes'],0):.1f} min (model average for flights like this; individual flights vary a lot)")
    P.append(f"**Route:** {m['origin']} - {m['origin_name']} to {m['destination']} - {m['destination_name']} ({m['distance_km']:.0f} km, about {m['block_minutes']:.0f} min scheduled)")
    P.append(f"**Departure:** {m['dep_local'].strftime('%Y-%m-%d %H:%M')} local ({m['origin_tz']}), {m['dep_utc'].strftime('%Y-%m-%d %H:%M')} UTC")
    P.append(f"**Carrier:** {m['carrier']} ({m['carrier_source']})")
    P.append("**Origin weather:** " + wx_line(m["weather_origin"], m["weather_origin_src"]))
    P.append("**Destination weather:** " + wx_line(m["weather_dest"], m["weather_dest_src"]))
    if "SEASONAL" in m["weather_origin_src"] or "SEASONAL" in m["weather_dest_src"]:
        P.append("> **Weather is a seasonal analog, not a forecast** (the date is beyond the 15-day forecast window).")
    P.append(f"**Top factors** (SHAP: minutes added or removed from the model average of {r['shap_base']:.1f} min):" + NL + NL + NL.join(f"- **{row.feature}** = {row.value:.3g}: {row.shap_minutes:+.2f} min" for row in r["shap"].itertuples()))
    if m["warnings"]:
        P.append("**Warnings:**" + NL + NL + NL.join("- " + w for w in m["warnings"]))
    P.append(f"*Held-out test (Jun-Jul 2026, US domestic): risk AUC {M['clf_test']['AUC']:.2f}; minutes MAE {M['rf_test']['MAE']:.1f} vs {M['base_median_test']['MAE']:.1f} for always guessing the median. Risk bands: LOW below 20%, ELEVATED 20-35%, HIGH 35% and up.*")
    return (NL + NL).join(P)

demo = gr.Interface(fn=run,
    inputs=[gr.Textbox(value="ATL", label="Origin airport (IATA, ICAO or name)"), gr.Textbox(value="LAX", label="Destination airport"),
            gr.Textbox(value=datetime.now().strftime("%Y-%m-%d"), label="Departure date (local, YYYY-MM-DD; any date)"), gr.Textbox(value="14:25", label="Scheduled departure time (LOCAL time at origin, e.g. 14:25)"),
            gr.Textbox(value="DL", label="Airline code (optional, e.g. DL, AA, UA, WN)")],
    outputs=gr.Markdown(label="Prediction"), title="AeroDelay AI",
    description="Predicts delay risk for US domestic flights. Trained on 10 months of BTS on-time data with time-matched weather. Uses forecast weather for the chosen airport and time.",
    examples=[["ATL", "LAX", datetime.now().strftime("%Y-%m-%d"), "14:25", "DL"], ["JFK", "MAD", datetime.now().strftime("%Y-%m-%d"), "14:30", ""]], flagging_mode="never")
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=os.environ.get("GRADIO_SHARE") == "1")
