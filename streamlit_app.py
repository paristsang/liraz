import io
import json
import math
import os
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
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)


# ------------------- CONFIG -------------------
MODEL_PATH = "models/rf_model.pkl"
FEATURES_PATH = "models/feature_names.json"
COVER_IMAGE_PATH = "assets/cover.png"  # <- add this file to your repo

st.set_page_config(page_title="Landslide Susceptibility (RF)", layout="wide")

REPORT_TEXT = """
The map depicts the values of the estimated Landslide Susceptibility Index (LSI). LSI values range from 0 to 1. The larger is the LSI value, the higher is the susceptibility to landslide phenomenon. The LSI has been calculated using the Machine Learning (ML) algorithm called Random Forest (RF). The RF model was trained based on an inventory of active, dormant and relict landslides located in the southwestern part of Cyprus, provisioned by the Geological Survey Department of Cyprus (GSD). Further information about the methodology and the performance of the RF model can be found in the Technical Report of the research project LIRA. The LSI values are not meant to be a study of fixed but perceived as exact estimates of the probability of existence of a landslide (active or inactive). Moreover, they do not represent probability of failure. The LSI values are aimed to function as indicators to need for caution and further, more detailed geological and geotechnical investigation. The Landslide Susceptibility Map provided herein is meant to complement the Map of Geological Suitability Zones provided in the web geoportal of the Geological Survey Department of Cyprus (GSD) at https://www.moa.gov.cy/moa/gsd/gsd.nsf/page17_en/page17_en?openDocument. The Landslide Susceptibility Map, along with input data that is made available through this web application, can assist practitioners in the preparation of the geohazard-focused geological/geotechnical studies provisioned by the pertinent circulars of the GSD, Department of Town Planning and Housing at https://www.etek.org.cy/uploads/5057f4b82.pdf, and the Scientific and Technical Chamber of Cyprus (ETEK) at https://www.etek.org.cy/uploads/Egkiklio/2023/297bbe3c0f.pdf
""".strip()


# ------------------- SUSCEPTIBILITY CLASSES (YOUR BREAKS) -------------------
SUSC_CLASSES = [
    {"name": "Very low susceptibility", "lo": 0.0,      "hi": 0.11796, "color": "#0b3d0b"},
    {"name": "Low susceptibility",      "lo": 0.11797,  "hi": 0.30196, "color": "#1f8a2a"},
    {"name": "Moderate susceptibility", "lo": 0.30197,  "hi": 0.52941, "color": "#f1c40f"},
    {"name": "High susceptibility",     "lo": 0.52942,  "hi": 0.80392, "color": "#e67e22"},
    {"name": "Very high susceptibility","lo": 0.80393,  "hi": 1.0,     "color": "#e74c3c"},
]

# ------------------- FR -> ORIGINAL CLASS MAPPING (FOR PDF TABLE) -------------------
# Each entry: (class_label, FR_value)
FEATURE_FR_CLASSES: Dict[str, List[Tuple[str, float]]] = {
    # Hydrology (your "Hydrology" table) -> likely fr_river
    "Distance to River": [
        ("< 100", 0.2842),
        ("100–200", 0.2796),
        ("200–300", 0.2430),
        ("300–400", 0.1235),
        ("NoData (9999)", 0.0697),
    ],

    # Tectonic -> fr_tect
    "Distance to Faults": [
        ("< 250", 0.2333),
        ("250–500", 0.2012),
        ("> 750", 0.1991),
        ("> 1000", 0.2050),
        ("NoData (9999)", 0.1614),
    ],

    # Plan curvature -> fr_plan
    "Plan Curvature": [
        ("< -0.2", 0.4631),
        ("-0.2 to 0.2 (±0.2)", 0.2332),
        ("> 0.2", 0.3037),
    ],

    # Profile curvature -> fr_prof
    "fr_prof": [
        ("< -0.15", 0.3264),
        ("-0.15 to 0.15 (≈0)", 0.2535),
        ("> 0.15", 0.4201),
    ],

    # Aspect -> fr_asp
    "fr_asp": [
        ("North (0–22.5, 337.5–360) = 1", 0.1601),
        ("Northeast (22.5–67.5) = 2", 0.1520),
        ("East (67.5–112.5) = 3", 0.1316),
        ("Southeast (112.5–157.5) = 4", 0.1129),
        ("South (157.5–202.5) = 5", 0.1016),
        ("Southwest (202.5–247.5) = 6", 0.0796),
        ("West (247.5–292.5) = 7", 0.1058),
        ("Northwest (292.5–337.5) = 8", 0.1564),
        ("Flat / No aspect = 9", 0.0000),
    ],

    # Slope -> fr_slop
    "fr_slop": [
        ("0–5° = 1", 0.0147),
        ("5–15° = 2", 0.1016),
        ("15–25° = 3", 0.1303),
        ("25–35° = 4", 0.0538),
        ("35–45° = 5", 0.0769),
        ("> 45° = 6", 0.6226),
    ],

    # Elevation -> fr_elev
    "fr_elev": [
        ("0–350", 0.2519),
        ("350–700", 0.4138),
        ("700–1050", 0.2286),
        ("1050–1400", 0.1057),
    ],

    # SPI -> fr_SPI
    "fr_SPI": [
        ("< 2", 0.1624),
        ("2–4", 0.2529),
        ("4–6", 0.2959),
        ("> 6", 0.2888),
    ],

    # TWI -> fr_TWI
    "fr_TWI": [
        ("< 4", 0.2129),
        ("4–6", 0.2122),
        ("6–8", 0.2345),
        ("8–10", 0.1634),
        ("> 10", 0.1771),
    ],

    # Land use -> fr_land
    "fr_land": [
        ("Artificial surfaces (class 1)", 0.0942),
        ("Agricultural areas (class 2)", 0.4777),
        ("Forest & semi-natural (class 3)", 0.2339),
        ("Wetlands (class 4)", 0.0000),
        ("Water bodies (class 5)", 0.1942),
    ],

    # Lithology -> fr_gew
    "fr_gew": [
        ("Quaternary formations", 0.0054),
        ("Nicosia Formation", 0.0103),
        ("Evaporites", 0.0058),
        ("Pachna Formation", 0.0086),
        ("Lefkara Formation", 0.0736),
        ("Talus", 0.1830),
        ("Kannaviou Formation", 0.1859),
        ("Agia Varvara Formation", 0.0272),
        ("Pera pedi", 0.0000),
        ("Melange", 0.1210),
        ("Mesozoic fine grained clastic", 0.1460),
        ("Limestones", 0.0370),
        ("Silica sandstones", 0.1007),
        ("Lava group", 0.0353),
        ("Subvolcanic", 0.0014),
        ("Serpentinite", 0.0478),
        ("Plutonic", 0.0108),
        ("Flysch", 0.0),
    ],

    # Road network -> fr_road
    "fr_road": [
        ("< 100", 0.110768893),
        ("100–200", 0.185879598),
        ("200–300", 0.227070635),
        ("300–400", 0.278489831),
        ("> 400", 0.197791042),
    ],

    # Plasticity Index -> fr_pi
    "fr_pi": [
        ("NoData = 1", 0.0520),
        ("0–20 = 2", 0.1576),
        ("20–40 = 3", 0.0858),
        ("40–60 = 4", 0.2103),
        ("> 60 = 5", 0.4943),
    ],

    # Clay content -> fr_clay
    "fr_clay": [
        ("NoData = 1", 0.0836),
        ("0–15 = 2", 0.0254),
        ("15–30 = 3", 0.1651),
        ("30–45 = 4", 0.2056),
        ("> 45 = 5", 0.5204),
    ],

    # GSI -> fr_gsi
    "fr_gsi": [
        ("0–40 = 1", 0.5936),
        ("40–60 = 2", 0.3096),
        ("60–90 = 3", 0.0099),
        ("NoData (9999) = 4", 0.0869),
    ],

    # Dip direction difference -> fr_dip
    "fr_dip": [
        ("0–30 = 1", 0.256515093),
        ("30–60 = 2", 0.198573182),
        ("60–90 = 3", 0.249106252),
        ("> 90 = 4", 0.295805473),
    ],
}


def fr_value_to_original_class(feature_name: str, fr_value: Any, *, tol: float = 1e-3) -> str:
    """
    Map a feature's FR value back to the original-value class label.
    Uses nearest match with tolerance to handle rounding (e.g. 0.1108 vs 0.110768893).
    """
    try:
        x = float(fr_value)
    except Exception:
        return "N/A"

    mapping = FEATURE_FR_CLASSES.get(feature_name)
    if not mapping:
        # e.g. rain_topo (raw value) or unknown feature
        return f"{x:.2f}" if feature_name == "rain_topo" else f"{x:.4f}"

    best_label, best_fr = min(mapping, key=lambda it: abs(it[1] - x))
    if abs(best_fr - x) <= tol:
        return best_label

    # If values are slightly off but still closest, return closest (with hint)
    return f"{best_label} (nearest)"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def classify_susceptibility(p: float) -> Dict[str, Any]:
    p = clamp01(p)
    for c in SUSC_CLASSES:
        if c["lo"] <= p <= c["hi"]:
            out = dict(c)
            out["range_label"] = f"{c['lo']:.5f} – {c['hi']:.5f}"
            return out
    out = dict(SUSC_CLASSES[-1])
    out["range_label"] = f"{out['lo']:.5f} – {out['hi']:.5f}"
    return out


def hex_to_rgb_tuple(hx: str) -> tuple[int, int, int]:
    hx = hx.lstrip("#")
    return int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)


# ------------------- LOAD MODEL & FEATURES -------------------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    if not isinstance(feature_names, list) or len(feature_names) != 17:
        raise ValueError("feature_names.json must be a JSON array of exactly 17 strings.")
    return model, feature_names


def infer_positive_class(model) -> Any:
    if not hasattr(model, "classes_"):
        return 1
    classes = list(model.classes_)
    if 1 in classes:
        return 1
    for c in classes:
        if isinstance(c, str) and "land" in c.lower():
            return c
    return classes[-1]


# ------------------- GEOJSON PARSING -------------------
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


# ------------------- PDF MAP IMAGE (ESRI EXPORT + OVERLAYS) -------------------
def _lonlat_to_webmerc(lon: float, lat: float) -> tuple[float, float]:
    R = 6378137.0
    x = R * math.radians(lon)
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def fetch_esri_export_png(
    lat: float,
    lon: float,
    w: int = 900,
    h: int = 420,
    half_width_m: float = 6000
) -> tuple[bytes | None, tuple[float, float, float, float] | None]:
    """
    Returns (png_bytes, bbox_3857) or (None, None).
    bbox_3857 = (xmin, ymin, xmax, ymax)
    """
    try:
        cx, cy = _lonlat_to_webmerc(lon, lat)
        half_height_m = half_width_m * (h / w)
        xmin, ymin, xmax, ymax = (cx - half_width_m, cy - half_height_m, cx + half_width_m, cy + half_height_m)

        url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/export"
        params = {
            "bbox": f"{xmin:.3f},{ymin:.3f},{xmax:.3f},{ymax:.3f}",
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": f"{w},{h}",
            "format": "png",
            "transparent": "false",
            "f": "image",
        }
        r = requests.get(url, params=params, timeout=15)
        ct = (r.headers.get("content-type") or "").lower()
        if r.status_code == 200 and r.content and ct.startswith("image/"):
            return r.content, (xmin, ymin, xmax, ymax)
    except Exception:
        pass
    return None, None


def _choose_nice_scale_length_m(target_m: float) -> float:
    if target_m <= 0:
        return 100.0
    exp = math.floor(math.log10(target_m))
    base = target_m / (10 ** exp)
    candidates = [1, 2, 5, 10]
    best = min(candidates, key=lambda c: abs(c - base))
    return best * (10 ** exp)


def _format_scale_label(meters: float) -> str:
    if meters >= 1000:
        return f"{meters/1000:.0f} km" if (meters % 1000 == 0) else f"{meters/1000:.1f} km"
    return f"{meters:.0f} m"


def draw_north_arrow(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    margin = 18
    cx = w - margin - 18
    cy = margin + 18
    draw.line([(cx, cy + 18), (cx, cy - 12)], fill=(0, 0, 0, 255), width=3)
    draw.polygon([(cx, cy - 16), (cx - 8, cy - 2), (cx + 8, cy - 2)], fill=(0, 0, 0, 255))
    draw.text((cx - 5, cy + 22), "N", fill=(0, 0, 0, 255))


def draw_scale_bar(draw: ImageDraw.ImageDraw, w: int, h: int, meters_per_pixel: float) -> None:
    if meters_per_pixel <= 0:
        return

    desired_px = 160
    target_m = meters_per_pixel * desired_px
    nice_m = _choose_nice_scale_length_m(target_m)
    bar_px = int(round(nice_m / meters_per_pixel))

    max_px = int(w * 0.35)
    if bar_px > max_px:
        nice_m = _choose_nice_scale_length_m(meters_per_pixel * max_px)
        bar_px = int(round(nice_m / meters_per_pixel))

    margin = 18
    y = h - margin - 14
    x0 = margin
    x1 = x0 + bar_px

    pad = 6
    label = _format_scale_label(nice_m)
    box_w = bar_px + 90
    box_h = 34
    draw.rectangle((x0 - pad, y - 22, x0 - pad + box_w, y - 22 + box_h),
                   fill=(255, 255, 255, 200), outline=(0, 0, 0, 120))

    draw.line([(x0, y), (x1, y)], fill=(0, 0, 0, 255), width=4)
    draw.line([(x0, y - 7), (x0, y + 7)], fill=(0, 0, 0, 255), width=2)
    draw.line([(x1, y - 7), (x1, y + 7)], fill=(0, 0, 0, 255), width=2)

    draw.text((x1 + 10, y - 10), label, fill=(0, 0, 0, 255))


def draw_coordinates_box(draw: ImageDraw.ImageDraw, w: int, h: int, lat: float, lon: float) -> None:
    margin = 18
    box_w = 250
    box_h = 40
    x0 = w - margin - box_w
    y0 = h - margin - box_h
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h),
                   fill=(255, 255, 255, 200), outline=(0, 0, 0, 120))
    draw.text((x0 + 10, y0 + 6), f"Lat: {lat:.6f}", fill=(0, 0, 0, 255))
    draw.text((x0 + 10, y0 + 22), f"Lon: {lon:.6f}", fill=(0, 0, 0, 255))


def overlay_point_and_decorations(
    png_bytes: bytes,
    lat: float,
    lon: float,
    bbox_3857: tuple[float, float, float, float],
    w: int,
    h: int,
    marker_hex: str,
) -> bytes:
    xmin, ymin, xmax, ymax = bbox_3857
    px_m, py_m = _lonlat_to_webmerc(lon, lat)

    x = int(round((px_m - xmin) / (xmax - xmin) * w))
    y = int(round((ymax - py_m) / (ymax - ymin) * h))

    img = PILImage.open(io.BytesIO(png_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img)

    fill = (*hex_to_rgb_tuple(marker_hex), 255)
    ring = (255, 255, 255, 255)
    r = 10
    draw.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3), fill=ring)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    meters_per_pixel = abs((xmax - xmin) / float(w))
    draw_north_arrow(draw, w, h)
    draw_scale_bar(draw, w, h, meters_per_pixel)
    draw_coordinates_box(draw, w, h, lat, lon)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def fallback_map_png(
    lat: float,
    lon: float,
    w: int = 900,
    h: int = 420,
    marker_hex: str = "#e74c3c",
    half_width_m: float = 6000
) -> bytes:
    img = PILImage.new("RGBA", (w, h), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img)
    grid = (210, 210, 210, 255)

    for gx in range(0, w, 60):
        draw.line([(gx, 0), (gx, h)], fill=grid, width=1)
    for gy in range(0, h, 60):
        draw.line([(0, gy), (w, gy)], fill=grid, width=1)

    draw.text((14, 14), "Basemap unavailable (fallback map)", fill=(60, 60, 60, 255))
    draw.text((14, 38), f"lat={lat:.6f}, lon={lon:.6f}", fill=(60, 60, 60, 255))

    cx, cy = w // 2, h // 2
    r = 10
    fill = (*hex_to_rgb_tuple(marker_hex), 255)
    draw.ellipse((cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3), fill=(255, 255, 255, 255))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)

    meters_per_pixel = (2 * half_width_m) / float(w)
    draw_north_arrow(draw, w, h)
    draw_scale_bar(draw, w, h, meters_per_pixel)
    draw_coordinates_box(draw, w, h, lat, lon)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ------------------- PDF BUILDER (WITH COVER PAGE) -------------------
def build_pdf(
    lat: float,
    lon: float,
    props: Dict[str, Any],
    feature_names: List[str],
    susc_class: Dict[str, Any],
    p_landslide: float,
) -> bytes:
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

    # --------- Cover page (Page 1) ---------
    if os.path.exists(COVER_IMAGE_PATH):
        cover = RLImage(COVER_IMAGE_PATH)

        avail_w = A4[0] - doc.leftMargin - doc.rightMargin
        avail_h = A4[1] - doc.topMargin - doc.bottomMargin

        iw, ih = float(cover.imageWidth), float(cover.imageHeight)
        scale = min(avail_w / iw, avail_h / ih)

        cover.drawWidth = iw * scale
        cover.drawHeight = ih * scale
        cover.hAlign = "CENTER"

        story.append(cover)
        story.append(PageBreak())
    else:
        story.append(Paragraph("Cover image missing: assets/cover.png", styles["Small"]))
        story.append(PageBreak())

    # --------- Report pages (Page 2+) ---------
    story.append(Paragraph("Landslide Susceptibility Report", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Small"]))
    story.append(Paragraph(f"Location (lat, lon): {lat:.6f}, {lon:.6f}", styles["Small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"<b>Susceptibility:</b> {susc_class['name']}", styles["Normal"]))
    story.append(Paragraph(f"<b>P(landslide):</b> {p_landslide:.4f}", styles["Normal"]))
    story.append(Paragraph(f"<b>Class range:</b> {susc_class['range_label']}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Map with point + north + scale + coords
    story.append(Paragraph("Map", styles["H2"]))
    W, H = 900, 420
    half_width_m = 6000  # smaller = more zoom-in
    png, bbox = fetch_esri_export_png(lat, lon, w=W, h=H, half_width_m=half_width_m)

    if png and bbox:
        map_bytes = overlay_point_and_decorations(
            png_bytes=png,
            lat=lat,
            lon=lon,
            bbox_3857=bbox,
            w=W,
            h=H,
            marker_hex=susc_class["color"],
        )
        story.append(RLImage(io.BytesIO(map_bytes), width=170 * mm, height=90 * mm))
        story.append(Paragraph("<i>Basemap: Esri World Street Map.</i>", styles["Small"]))
    else:
        map_bytes = fallback_map_png(lat, lon, w=W, h=H, marker_hex=susc_class["color"], half_width_m=half_width_m)
        story.append(RLImage(io.BytesIO(map_bytes), width=170 * mm, height=90 * mm))
        story.append(Paragraph("<i>Basemap unavailable, showing fallback map.</i>", styles["Small"]))

    story.append(Spacer(1, 10))

    # Notes
    story.append(Paragraph("Notes", styles["H2"]))
    story.append(Paragraph(REPORT_TEXT.replace("\n", " "), styles["Small"]))
    story.append(Spacer(1, 10))

    
    # Feature table (show ORIGINAL CLASS instead of FR value)
    story.append(Paragraph("Input Features (17 variables)", styles["H2"]))

    rows = []
    for name in feature_names:
        cls = fr_value_to_original_class(name, props.get(name, ""))
        rows.append([name, cls])

    table = Table([["Feature", "Class"]] + rows, colWidths=[60 * mm, 110 * mm])
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


# ------------------- STREAMLIT UI -------------------
st.title("LIRA Project - Landslide Susceptibility in Cyprus")

with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("Upload GeoJSON (FeatureCollection with 1 Point)", type=["geojson", "json"])
    st.caption("The first point is used. Properties must include the 17 variables.")

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

# Predict probability
preds = model.predict(X)
pred_class = preds[0]

p_landslide = None
if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
    pos_class = infer_positive_class(model)
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    pos_idx = classes.index(pos_class) if pos_class in classes else (proba.shape[1] - 1)
    p_landslide = float(proba[0, pos_idx])
else:
    # If regression output is already LSI in [0,1]
    try:
        p_landslide = float(pred_class)
    except Exception:
        p_landslide = None

if p_landslide is None:
    st.error("Model did not provide a usable probability/LSI value in [0,1].")
    st.stop()

p_landslide = clamp01(p_landslide)
susc = classify_susceptibility(p_landslide)

# Layout: map + results
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Map")
    m = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB positron")
    folium.CircleMarker(
        location=[lat, lon],
        radius=9,
        color="#111827",
        weight=2,
        fill=True,
        fill_opacity=0.95,
        fill_color=susc["color"],
        popup=folium.Popup(
            f"<b>{susc['name']}</b><br/>P(landslide): {p_landslide:.4f}<br/>Range: {susc['range_label']}",
            max_width=300,
        ),
    ).add_to(m)
    st_folium(m, height=520, width=None)

with col_right:
    st.subheader("Landslide Susceptibility Result")
    st.markdown(
        f"""
        <div style="padding:12px;border-radius:12px;border:1px solid rgba(255,255,255,0.15);background:rgba(255,255,255,0.03);">
          <div style="font-size:18px;font-weight:900;">
            Susceptibility: {susc['name']}
            <span style="display:inline-block;margin-left:8px;width:14px;height:14px;border-radius:3px;background:{susc['color']};border:1px solid rgba(255,255,255,0.4);"></span>
          </div>
          <div style="opacity:0.95;margin-top:6px;">
            P(landslide): <b>{p_landslide:.4f}</b><br/>
            Class range: <b>{susc['range_label']}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Input features")
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "fr_value": [props.get(k) for k in feature_names],
            "class": [fr_value_to_original_class(k, props.get(k, "")) for k in feature_names],
        }
    ).set_index("feature")

    st.dataframe(df, use_container_width=True)


    st.subheader("PDF report")
    pdf_bytes = build_pdf(
        lat=lat,
        lon=lon,
        props=props,
        feature_names=feature_names,
        susc_class=susc,
        p_landslide=p_landslide,
    )

    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=f"landslide_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
