import io
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

from PIL import Image as PILImage, ImageDraw

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage


# ------------------- YOUR PDF NOTES TEXT -------------------
REPORT_TEXT = """
The map depicts the values of the estimated Landslide Susceptibility Index (LSI). LSI values range from 0 to 1. The larger is the LSI value, the higher is the susceptibility to landslide phenomenon. The LSI has been calculated using the Machine Learning (ML) algorithm called Random Forest (RF). The RF model was trained based on an inventory of active, dormant and relict landslides located in the southwestern part of Cyprus, provisioned by the Geological Survey Department of Cyprus (GSD). Further information about the methodology and the performance of the RF model can be found in the Technical Report of the research project LIRA. The LSI values are not meant to be a study of fixed but perceived as exact estimates of the probability of existence of a landslide (active or inactive). Moreover, they do not represent probability of failure. The LSI values are aimed to function as indicators to need for caution and further, more detailed geological and geotechnical investigation. The Landslide Susceptibility Map provided herein is meant to complement the Map of Geological Suitability Zones provided in the web geoportal of the Geological Survey Department of Cyprus (GSD) at https://www.moa.gov.cy/moa/gsd/gsd.nsf/page17_en/page17_en?openDocument. The Landslide Susceptibility Map, along with input data that is made available through this web application, can assist practitioners in the preparation of the geohazard-focused geological/geotechnical studies provisioned by the pertinent circulars of the GSD, Department of Town Planning and Housing at https://www.etek.org.cy/uploads/5057f4b82.pdf, and the Scientific and Technical Chamber of Cyprus (ETEK) at https://www.etek.org.cy/uploads/Egkiklio/2023/297bbe3c0f.pdf
""".strip()

MODEL_PATH = "models/rf_model.pkl"
FEATURES_PATH = "models/feature_names.json"

st.set_page_config(page_title="Landslide Susceptibility (RF)", layout="wide")


# ------------------- SUSCEPTIBILITY CLASSES (YOUR LEGEND BREAKS) -------------------
# Breaks from your legend image:
# 0 - 0.11796
# 0.11797 - 0.30196
# 0.30197 - 0.52941
# 0.52942 - 0.80392
# 0.80393 - 1
SUSC_CLASSES = [
    {"name": "Very low susceptibility", "lo": 0.0,      "hi": 0.11796, "color": "#0b3d0b"},  # dark green
    {"name": "Low susceptibility",      "lo": 0.11797,  "hi": 0.30196, "color": "#2ecc71"},  # green
    {"name": "Moderate susceptibility", "lo": 0.30197,  "hi": 0.52941, "color": "#f1c40f"},  # yellow
    {"name": "High susceptibility",     "lo": 0.52942,  "hi": 0.80392, "color": "#e67e22"},  # orange
    {"name": "Very high susceptibility","lo": 0.80393,  "hi": 1.0,     "color": "#e74c3c"},  # red
]


def classify_susceptibility(p: float) -> Dict[str, Any]:
    """
    Classify probability/LSI p in [0,1] into 5 susceptibility classes (your legend).
    Returns the matching class dict plus a nice range label.
    """
    # clamp to [0,1]
    p = max(0.0, min(1.0, float(p)))

    for c in SUSC_CLASSES:
        lo, hi = c["lo"], c["hi"]
        # include endpoints
        if lo <= p <= hi:
            out = dict(c)
            out["range_label"] = f"{lo:.5f} – {hi:.5f}"
            return out

    # fallback (shouldn't happen)
    out = dict(SUSC_CLASSES[-1])
    out["range_label"] = f"{out['lo']:.5f} – {out['hi']:.5f}"
    return out


def legend_html(selected_name: str | None = None) -> str:
    """
    Small HTML legend for Streamlit.
    Highlights selected class with a border.
    """
    rows = []
    for c in SUSC_CLASSES:
        border = "2px solid #ffffff" if selected_name and c["name"] == selected_name else "1px solid rgba(255,255,255,0.25)"
        rows.append(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin:6px 0;">
              <div style="width:18px;height:14px;background:{c['color']};border:{border};border-radius:3px;"></div>
              <div style="font-size:13px;">
                <b>{c['name']}</b><br/>
                <span style="opacity:0.85">{c['lo']:.5f} – {c['hi']:.5f}</span>
              </div>
            </div>
            """
        )
    return "<div>" + "".join(rows) + "</div>"


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    if not isinstance(feature_names, list) or len(feature_names) != 17:
        raise ValueError("feature_names.json must be a JSON array of exactly 17 strings.")
    return model, feature_names


def infer_positive_class(model) -> Any:
    # landslide class preference: 1, else contains "land", else last
    if not hasattr(model, "classes_"):
        return 1
    classes = list(model.classes_)
    if 1 in classes:
        return 1
    for c in classes:
        if isinstance(c, str) and "land" in c.lower():
            return c
    return classes[-1]


def label_from_class(pred_class: Any, positive_class: Any) -> str:
    return "landslide" if pred_class == positive_class else "non-landslide"


def parse_geojson(uploaded_bytes: bytes) -> Dict[str, Any]:
    try:
        obj = json.loads(uploaded_bytes.decode("utf-8"))
    except Exception:
        raise ValueError("File is not valid JSON/GeoJSON.")

    if not isinstance(obj, dict) or obj.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection.")

    feats = obj.get("features")
    if not isinstance(feats, list) or len(feats) == 0:
        raise ValueError("FeatureCollection.features must be a non-empty list.")

    # Take first feature (single-point workflow)
    feat = feats[0]
    if feat.get("type") != "Feature":
        raise ValueError("First item in features must be a GeoJSON Feature.")
    if (feat.get("geometry") or {}).get("type") != "Point":
        raise ValueError("Only Point geometry is supported.")
    coords = (feat.get("geometry") or {}).get("coordinates")
    if not (isinstance(coords, list) and len(coords) >= 2):
        raise ValueError("Point.coordinates must be [lon, lat].")

    props = feat.get("properties") or {}
    if not isinstance(props, dict):
        raise ValueError("Feature.properties must be an object.")

    return obj


def extract_single_point(fc: Dict[str, Any], feature_names: List[str]) -> Tuple[float, float, Dict[str, Any], np.ndarray]:
    feat = fc["features"][0]
    lon, lat = feat["geometry"]["coordinates"][0], feat["geometry"]["coordinates"][1]
    props = feat.get("properties") or {}

    missing = [k for k in feature_names if k not in props]
    if missing:
        raise ValueError(f"Missing properties: {missing}")

    row = []
    for k in feature_names:
        v = props.get(k)
        try:
            row.append(float(v))
        except Exception:
            raise ValueError(f"Property '{k}' must be numeric. Got: {v}")

    X = np.array([row], dtype=float)
    return float(lat), float(lon), props, X


# ------------------- MAP IMAGE FOR PDF (ESRI EXPORT + FALLBACK) -------------------
def _lonlat_to_webmerc(lon: float, lat: float) -> tuple[float, float]:
    # EPSG:3857
    R = 6378137.0
    x = R * math.radians(lon)
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def fetch_esri_export_png(lat: float, lon: float, w: int = 900, h: int = 420, half_width_m: float = 6000) -> bytes | None:
    """
    Reliable basemap image from ArcGIS Online basemap export endpoint.
    """
    try:
        cx, cy = _lonlat_to_webmerc(lon, lat)
        half_height_m = half_width_m * (h / w)
        bbox = (cx - half_width_m, cy - half_height_m, cx + half_width_m, cy + half_height_m)

        url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/export"
        params = {
            "bbox": ",".join(f"{v:.3f}" for v in bbox),
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": f"{w},{h}",
            "format": "png",
            "transparent": "false",
            "f": "image",
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200 and r.content and r.headers.get("content-type", "").lower().startswith("image/"):
            return r.content
    except Exception:
        pass
    return None


def fallback_map_png(lat: float, lon: float, w: int = 900, h: int = 420, marker_color=(231, 76, 60)) -> bytes:
    """
    Always-works fallback: simple grid + point + coords (no basemap).
    """
    img = PILImage.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    grid = (210, 210, 210)
    for gx in range(0, w, 60):
        draw.line([(gx, 0), (gx, h)], fill=grid, width=1)
    for gy in range(0, h, 60):
        draw.line([(0, gy), (w, gy)], fill=grid, width=1)

    draw.text((14, 14), "Basemap unavailable (fallback map)", fill=(60, 60, 60))
    draw.text((14, 38), f"lat={lat:.6f}, lon={lon:.6f}", fill=(60, 60, 60))

    cx, cy = w // 2, h // 2
    r = 10
    draw.ellipse((cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2), fill=(255, 255, 255))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=marker_color)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def hex_to_rgb_tuple(hx: str) -> tuple[int, int, int]:
    hx = hx.lstrip("#")
    return int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)


def build_pdf(
    lat: float,
    lon: float,
    props: Dict[str, Any],
    feature_names: List[str],
    susc_class: Dict[str, Any],
    prob_landslide: float | None,
    model_binary_label: str | None = None,
) -> bytes:
    """
    Build a PDF:
      - Susceptibility class (5-category)
      - P(landslide) displayed as 0.xxxx
      - Map with point
      - Notes text
      - Table of 17 variables
      - Legend (5 categories)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Landslide Susceptibility Report",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=6))

    story = []
    story.append(Paragraph("Landslide Susceptibility Report (Single Point)", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Small"]))
    story.append(Paragraph(f"Location (lat, lon): {lat:.6f}, {lon:.6f}", styles["Small"]))
    story.append(Spacer(1, 8))

    prob_txt = "N/A" if prob_landslide is None else f"{float(prob_landslide):.4f}"
    story.append(Paragraph(f"<b>Susceptibility:</b> {susc_class['name']}", styles["Normal"]))
    story.append(Paragraph(f"<b>P(landslide):</b> {prob_txt}", styles["Normal"]))
    story.append(Paragraph(f"<b>Class range:</b> {susc_class['range_label']}", styles["Normal"]))

    # optional: keep binary model label in small text (doesn't replace susceptibility)
    if model_binary_label:
        story.append(Paragraph(f"<i>Model binary label:</i> {model_binary_label}", styles["Small"]))

    story.append(Spacer(1, 10))

    # Map image (ESRI export -> fallback)
    story.append(Paragraph("Map", styles["H2"]))
    map_bytes = fetch_esri_export_png(lat, lon)  # reliable on Streamlit Cloud
    if not map_bytes:
        marker_rgb = hex_to_rgb_tuple(susc_class["color"])
        map_bytes = fallback_map_png(lat, lon, marker_color=marker_rgb)

    img = RLImage(io.BytesIO(map_bytes))
    img.drawWidth = 170 * mm
    img.drawHeight = 90 * mm
    story.append(img)
    story.append(Paragraph("<i>Basemap: Esri World Street Map (if available).</i>", styles["Small"]))
    story.append(Spacer(1, 10))

    # Legend (5 classes)
    story.append(Paragraph("Legend (Susceptibility classes)", styles["H2"]))
    legend_rows = [["Class", "Range (LSI / P(landslide))"]]
    for c in SUSC_CLASSES:
        legend_rows.append([c["name"], f"{c['lo']:.5f} – {c['hi']:.5f}"])

    legend_table = Table(legend_rows, colWidths=[75 * mm, 95 * mm])
    # Style: color swatches by painting the class cell background
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    # Color each class cell
    for i, c in enumerate(SUSC_CLASSES, start=1):
        ts.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(c["color"])))
        ts.append(("TEXTCOLOR", (0, i), (0, i), colors.white))
        if c["name"] == susc_class["name"]:
            ts.append(("BOX", (0, i), (-1, i), 1.2, colors.black))  # highlight selected class row

    legend_table.setStyle(TableStyle(ts))
    story.append(legend_table)
    story.append(Spacer(1, 10))

    # Notes
    story.append(Paragraph("Notes", styles["H2"]))
    story.append(Paragraph(REPORT_TEXT.replace("\n", " "), styles["Small"]))
    story.append(Spacer(1, 10))

    # Feature table
    story.append(Paragraph("Input Features (17 variables)", styles["H2"]))
    rows = []
    for name in feature_names:
        v = props.get(name, "")
        try:
            fv = float(v)
            if name == "rain_topo":
                v_str = f"{fv:.2f}"
            else:
                v_str = f"{fv:.4f}"
        except Exception:
            v_str = str(v)
        rows.append([name, v_str])

    table = Table([["Feature", "Value"]] + rows, colWidths=[60 * mm, 110 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    story.append(table)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ---------------- UI ----------------
st.title("Landslide Susceptibility (Random Forest)")

with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("Upload GeoJSON (FeatureCollection with 1 Point)", type=["geojson", "json"])
    st.caption("The point must have the 17 properties used by the RF model.")

try:
    model, feature_names = load_assets()
except Exception as e:
    st.error(f"Failed to load model/assets: {e}")
    st.stop()

if not up:
    st.info("Upload a GeoJSON to begin.")
    st.stop()

try:
    fc = parse_geojson(up.getvalue())
    if len(fc["features"]) > 1:
        st.warning(f"Your file has {len(fc['features'])} features. This demo uses ONLY the first one.")
    lat, lon, props, X = extract_single_point(fc, feature_names)
except Exception as e:
    st.error(str(e))
    st.stop()

# Predict
preds = model.predict(X)
pred_class = preds[0]

prob_landslide = None
binary_label = None

positive_class = None
if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
    positive_class = infer_positive_class(model)
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    pos_idx = classes.index(positive_class) if positive_class in classes else (proba.shape[1] - 1)
    prob_landslide = float(proba[0, pos_idx])
    binary_label = label_from_class(pred_class, positive_class)
else:
    # If model has no proba, we can’t compute susceptibility bins from probability.
    # You can decide to treat regression output as LSI if that’s your case.
    try:
        prob_landslide = float(pred_class)  # if regression output already in [0,1]
    except Exception:
        prob_landslide = None

if prob_landslide is None:
    st.error("This model did not provide a usable probability/LSI value. Susceptibility classes require P(landslide) in [0,1].")
    st.stop()

susc = classify_susceptibility(prob_landslide)

# Layout: Map + Results
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Map")
    m = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB positron")

    popup_html = f"""
    <div style="font-family:Arial;font-size:13px;">
      <b>Susceptibility:</b> {susc['name']}<br/>
      <b>P(landslide):</b> {prob_landslide:.4f}<br/>
      <b>Range:</b> {susc['range_label']}<br/>
    """
    if binary_label:
        popup_html += f"<b>Binary label:</b> {binary_label}<br/>"
    popup_html += "</div>"

    folium.CircleMarker(
        location=[lat, lon],
        radius=9,
        color="#111827",
        weight=2,
        fill=True,
        fill_opacity=0.95,
        fill_color=susc["color"],
        popup=folium.Popup(popup_html, max_width=280),
    ).add_to(m)

    st_folium(m, height=520, width=None)

with col_right:
    st.subheader("Classification Result")
    st.markdown(
        f"""
        <div style="padding:12px;border-radius:12px;border:1px solid rgba(255,255,255,0.15);background:rgba(255,255,255,0.03);">
          <div style="font-size:18px;font-weight:900;">
            Susceptibility: {susc['name']}
            <span style="display:inline-block;margin-left:8px;width:14px;height:14px;border-radius:3px;background:{susc['color']};border:1px solid rgba(255,255,255,0.4);"></span>
          </div>
          <div style="opacity:0.9;margin-top:6px;">
            P(landslide): <b>{prob_landslide:.4f}</b><br/>
            Class range: <b>{susc['range_label']}</b>
          </div>
          {f"<div style='opacity:0.85;margin-top:6px;'><i>Binary model label:</i> {binary_label}</div>" if binary_label else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Legend")
    st.markdown(legend_html(selected_name=susc["name"]), unsafe_allow_html=True)

    st.subheader("Input features")
    df = pd.DataFrame([{k: props.get(k) for k in feature_names}]).T
    df.columns = ["value"]
    st.dataframe(df, use_container_width=True)

    st.subheader("PDF report")
    pdf_bytes = build_pdf(
        lat=lat,
        lon=lon,
        props=props,
        feature_names=feature_names,
        susc_class=susc,
        prob_landslide=prob_landslide,
        model_binary_label=binary_label,  # optional small line
    )

    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=f"landslide_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
