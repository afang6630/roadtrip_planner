import math
import os
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple

import requests
import streamlit as st
import pandas as pd
import pydeck as pdk


# =======================
# Data Models
# =======================

@dataclass
class Park:
    park_id: str
    name: str
    state: str
    lat: float
    lon: float
    best_start_month: int  # 1-12
    best_end_month: int    # 1-12
    phenomenon: str
    entrance_fee: float        # per vehicle
    typical_daily_cost: float  # lodging + food rough estimate near the park
    notes: str


@dataclass
class City:
    city_id: str
    name: str
    lat: float
    lon: float


# =======================
# Static Data
# =======================

PARKS: Dict[str, Park] = {
    "YOSE": Park(
        "YOSE", "Yosemite National Park", "CA",
        37.8651, -119.5383,
        best_start_month=5, best_end_month=6,
        phenomenon="Waterfalls & spring runoff",
        entrance_fee=35.0,
        typical_daily_cost=90.0,
        notes="Very crowded in June; snow/road closures possible in May at higher elevations."
    ),
    "SEKI": Park(
        "SEKI", "Sequoia & Kings Canyon National Parks", "CA",
        36.4864, -118.5658,
        best_start_month=5, best_end_month=9,
        phenomenon="Giant sequoias & high-country hikes",
        entrance_fee=35.0,
        typical_daily_cost=85.0,
        notes="Snow at higher elevations into early summer; hot in lower canyons mid-summer."
    ),
    "ZION": Park(
        "ZION", "Zion National Park", "UT",
        37.2982, -113.0263,
        best_start_month=4, best_end_month=10,
        phenomenon="Red-rock canyons & iconic hikes",
        entrance_fee=35.0,
        typical_daily_cost=80.0,
        notes="Very hot in summer afternoons; shuttle crowds in peak months."
    ),
    "BRCA": Park(
        "BRCA", "Bryce Canyon National Park", "UT",
        37.5930, -112.1871,
        best_start_month=5, best_end_month=9,
        phenomenon="Hoodoos & high-elevation views",
        entrance_fee=35.0,
        typical_daily_cost=75.0,
        notes="Cooler temps due to elevation; snow possible in shoulder seasons."
    ),
    "ARCH": Park(
        "ARCH", "Arches National Park", "UT",
        38.7331, -109.5925,
        best_start_month=4, best_end_month=6,
        phenomenon="Red-rock arches & night skies",
        entrance_fee=30.0,
        typical_daily_cost=80.0,
        notes="Very hot mid-summer; timed entry reservations certain seasons."
    ),
    "RMNP": Park(
        "RMNP", "Rocky Mountain National Park", "CO",
        40.3428, -105.6836,
        best_start_month=7, best_end_month=9,
        phenomenon="Wildflowers & alpine hiking",
        entrance_fee=30.0,
        typical_daily_cost=95.0,
        notes="Trail Ridge Road often closed until late June due to snow."
    ),
    "GRSM": Park(
        "GRSM", "Great Smoky Mountains National Park", "TN/NC",
        35.6118, -83.4895,
        best_start_month=5, best_end_month=6,
        phenomenon="Lush forests & synchronous fireflies (lottery)",
        entrance_fee=0.0,
        typical_daily_cost=75.0,
        notes="Very popular in June and October; humidity and storms in summer."
    ),
    "ACAD": Park(
        "ACAD", "Acadia National Park", "ME",
        44.3386, -68.2733,
        best_start_month=9, best_end_month=10,
        phenomenon="Coastal foliage & crisp air",
        entrance_fee=35.0,
        typical_daily_cost=100.0,
        notes="Fog common; peak foliage varies year to year."
    ),
}

CITIES: Dict[str, City] = {
    "LA": City("LA", "Los Angeles, CA", 34.0522, -118.2437),
    "DEN": City("DEN", "Denver, CO", 39.7392, -104.9903),
    "LV": City("LV", "Las Vegas, NV", 36.1699, -115.1398),
    "ATL": City("ATL", "Atlanta, GA", 33.7490, -84.3880),
    "BOS": City("BOS", "Boston, MA", 42.3601, -71.0589),
}


# =======================
# Ollama Chat Backend
# =======================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1")  # or another pulled model


def call_chat_model(messages: List[Dict[str, str]]) -> str:
    """
    Call a local Ollama chat model with OpenAI-style messages.
    Requires Ollama running locally and the model pulled.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL_NAME,
                "messages": messages,
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "I couldn't generate a response.")
    except Exception as e:
        return f"(Ollama error: {e})"


# =======================
# Styling
# =======================

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Hide default menu & footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Page background */
        .main {
            background: radial-gradient(circle at top left, #e0f2fe 0, #f9fafb 45%, #ffffff 100%);
        }

        /* Layout width & padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 60%, #111827 100%);
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {
            color: #e5e7eb !important;
        }

        .stButton>button {
            background: #06b6d4;
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.4rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 12px 30px rgba(8, 47, 73, 0.28);
        }
        .stButton>button:hover {
            background: #0e7490;
            transform: translateY(-1px);
        }

        /* Hero */
        .hero-pill {
            display: inline-block;
            padding: 0.15rem 0.8rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0369a1;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            color: #0f172a;
            margin-top: 0.6rem;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            color: #4b5563;
            max-width: 26rem;
        }

        /* Cards */
        .glass-card {
            background: rgba(255,255,255,0.94);
            border-radius: 18px;
            padding: 1.1rem 1.35rem;
            box-shadow: 0 20px 55px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(226, 232, 240, 0.9);
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .section-kicker {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6b7280;
            margin-bottom: 0.2rem;
        }

        .stat-chip {
            background: #ffffff;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.8rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
            border: 1px solid #e5e7eb;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }
        .stat-chip-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.7rem;
            color: #9ca3af;
        }
        .stat-chip-value {
            font-weight: 700;
            color: #0f172a;
        }

        /* Dataframe container */
        .stDataFrame div[data-testid="stVerticalBlock"] {
            background: white;
            border-radius: 16px;
            padding: 0.5rem;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =======================
# Utility Functions
# =======================

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def estimate_drive_hours(distance_miles: float, avg_speed_mph: float = 55.0) -> float:
    return distance_miles / avg_speed_mph


def phenomenon_alignment_score(park: Park, trip_month: int) -> float:
    start, end = park.best_start_month, park.best_end_month
    if start <= end:
        in_window = start <= trip_month <= end
        distance = 0 if in_window else min(abs(trip_month - start), abs(trip_month - end))
    else:
        in_window = trip_month >= start or trip_month <= end
        if in_window:
            distance = 0
        else:
            d1 = (start - trip_month) % 12
            d2 = (trip_month - end) % 12
            distance = min(d1, d2)

    if distance == 0:
        return 1.0
    return max(0.0, 1.0 - distance / 4.0)


def predict_peak_window(park: Park, year: int) -> Tuple[date, date]:
    mid_month = (park.best_start_month + park.best_end_month) / 2
    mid_month_int = int(round(mid_month))
    start = date(year, mid_month_int, 10)
    end = date(year, mid_month_int, 20)
    return start, end


def plan_segment(start_lat: float, start_lon: float,
                 end_lat: float, end_lon: float) -> Tuple[float, float, int]:
    dist = haversine_miles(start_lat, start_lon, end_lat, end_lon)
    hours = estimate_drive_hours(dist)
    rest_stops = int(hours // 3)
    return dist, hours, rest_stops


def compute_route(start_city_id: str, park_ids: List[str]) -> Dict:
    if start_city_id not in CITIES:
        raise ValueError("Unknown start city")
    if not park_ids:
        raise ValueError("No parks provided")

    segments = []
    total_miles = 0.0
    total_hours = 0.0

    city = CITIES[start_city_id]
    first_park = PARKS[park_ids[0]]
    dist, hours, stops = plan_segment(city.lat, city.lon, first_park.lat, first_park.lon)
    segments.append({
        "from": city.name,
        "to": first_park.name,
        "distance_miles": dist,
        "drive_hours": hours,
        "rest_stops": stops,
    })
    total_miles += dist
    total_hours += hours

    for i in range(len(park_ids) - 1):
        p1 = PARKS[park_ids[i]]
        p2 = PARKS[park_ids[i + 1]]
        dist, hours, stops = plan_segment(p1.lat, p1.lon, p2.lat, p2.lon)
        segments.append({
            "from": p1.name,
            "to": p2.name,
            "distance_miles": dist,
            "drive_hours": hours,
            "rest_stops": stops,
        })
        total_miles += dist
        total_hours += hours

    return {
        "segments": segments,
        "total_miles": total_miles,
        "total_hours": total_hours,
    }


def estimate_budget(
    park_ids: List[str],
    trip_days: int,
    total_miles: float,
    mpg: float,
    gas_price: float,
    lodging_style: str = "camping",
    food_per_day: float = 40.0,
) -> Dict:
    if not park_ids:
        raise ValueError("No parks")

    fuel_cost = (total_miles / mpg) * gas_price

    if lodging_style == "camping":
        nightly_rate = 30.0
    elif lodging_style == "budget":
        nightly_rate = 80.0
    else:
        nightly_rate = 150.0

    lodging_cost = nightly_rate * max(trip_days - 1, 0)
    food_cost = food_per_day * trip_days
    park_fees = sum(PARKS[p].entrance_fee for p in set(park_ids))
    misc_cost = 0.15 * (fuel_cost + lodging_cost + food_cost + park_fees)
    total_cost = fuel_cost + lodging_cost + food_cost + park_fees + misc_cost

    return {
        "fuel_cost": fuel_cost,
        "lodging_cost": lodging_cost,
        "food_cost": food_cost,
        "park_fees": park_fees,
        "misc_cost": misc_cost,
        "total_cost": total_cost,
    }


def risk_score_for_segment(hours: float, month: int) -> float:
    base = 20.0
    if hours > 4:
        base += (hours - 4) * 5
    if month in (12, 1, 2):
        base += 20
    return max(0.0, min(100.0, base))


def overall_trip_risk(route_info: Dict, trip_month: int, max_daily_hours: float) -> Dict:
    segment_risks = []
    for seg in route_info["segments"]:
        seg_hours = seg["drive_hours"]
        seg_risk = risk_score_for_segment(seg_hours, trip_month)
        if seg_hours > max_daily_hours:
            seg_risk += 15
        seg_risk = min(seg_risk, 100.0)
        segment_risks.append(seg_risk)
        seg["risk_score"] = seg_risk

    avg_risk = sum(segment_risks) / len(segment_risks)
    worst_risk = max(segment_risks) if segment_risks else 0.0

    return {
        "avg_risk": avg_risk,
        "worst_risk": worst_risk,
        "segments": route_info["segments"],
    }


def generate_trip_summary(
    start_city_id: str,
    park_ids: List[str],
    year: int,
    month: int,
    trip_days: int,
    route_info: Dict,
    budget_info: Dict,
    risk_info: Dict,
) -> str:
    city = CITIES[start_city_id]
    park_names = [PARKS[p].name for p in park_ids]
    total_miles = route_info["total_miles"]
    total_hours = route_info["total_hours"]

    risk_text = "low"
    if risk_info["avg_risk"] > 70:
        risk_text = "high"
    elif risk_info["avg_risk"] > 40:
        risk_text = "medium"

    best_park_id = max(park_ids, key=lambda pid: phenomenon_alignment_score(PARKS[pid], month))
    best_park = PARKS[best_park_id]
    peak_start, peak_end = predict_peak_window(best_park, year)

    lines = []
    lines.append(f"For a {trip_days}-day roadtrip starting from {city.name}, you'll visit:")
    for name in park_names:
        lines.append(f"  • {name}")
    lines.append("")
    lines.append(f"Total driving is about {total_miles:.0f} miles over {total_hours:.1f} hours.")
    lines.append(f"Average segment risk is {risk_info['avg_risk']:.0f}/100 ({risk_text} risk).")
    lines.append("")
    lines.append(f"In month {month}, the best seasonal timing on this route is at {best_park.name},")
    lines.append(
        f"with an expected peak for {best_park.phenomenon.lower()} around "
        f"{peak_start.strftime('%b %d')}–{peak_end.strftime('%b %d')}."
    )
    lines.append("")
    lines.append("Estimated budget breakdown:")
    lines.append(f"  • Fuel: ${budget_info['fuel_cost']:.0f}")
    lines.append(f"  • Lodging: ${budget_info['lodging_cost']:.0f}")
    lines.append(f"  • Food: ${budget_info['food_cost']:.0f}")
    lines.append(f"  • Park entrance fees: ${budget_info['park_fees']:.0f}")
    lines.append(f"  • Miscellaneous buffer: ${budget_info['misc_cost']:.0f}")
    lines.append(f"  → Total estimated cost: ${budget_info['total_cost']:.0f}")
    lines.append("")
    lines.append("Longer drives are where most of your risk comes from, so consider:")
    lines.append("  • Starting driving days early to avoid night driving;")
    lines.append("  • Building in rest stops every 2–3 hours on segments flagged as higher risk;")
    lines.append("  • Having a weather and closure backup plan for mountain passes.")
    return "\n".join(lines)


def ai_recommendation(summary_text: str, park_ids: List[str], trip_month: int) -> str:
    month_name = date(2000, trip_month, 1).strftime("%B")
    park_names = ", ".join(PARKS[pid].name for pid in park_ids)

    system_msg = {
        "role": "system",
        "content": (
            "You are an AI travel planner for U.S. national parks roadtrips. "
            "You give concise, practical advice about routes, seasons, and safety."
        ),
    }

    user_msg = {
        "role": "user",
        "content": f"""
The user is planning this roadtrip:

{summary_text}

Additional context:
- Parks on the route: {park_names}
- Trip month: {month_name}

Write a friendly 3–5 sentence recommendation that:
- Highlights the best seasonal experience on this route,
- Mentions any driving risk or pacing concerns,
- Gives 1–2 practical tips (timing, hydration, weather/closures, reservations).

Make it concise and conversational (no bullet points).
""",
    }

    return call_chat_model([system_msg, user_msg])


def build_route_points_and_path(start_city_id: str, park_ids: List[str]) -> Tuple[pd.DataFrame, list]:
    city = CITIES[start_city_id]
    points = [{
        "name": city.name,
        "type": "Start",
        "order": 0,
        "lat": city.lat,
        "lon": city.lon,
    }]

    for i, pid in enumerate(park_ids, start=1):
        p = PARKS[pid]
        points.append({
            "name": p.name,
            "type": "Park",
            "order": i,
            "lat": p.lat,
            "lon": p.lon,
        })

    df_points = pd.DataFrame(points)
    path = [[row["lon"], row["lat"]] for _, row in df_points.sort_values("order").iterrows()]
    return df_points, path


# =======================
# Streamlit App
# =======================

def main():
    st.set_page_config(
        page_title="Roadtrip Planner",
        page_icon="🗺️",
        layout="wide",
    )

    inject_custom_css()

    # ---------- HERO ----------
    hero_col1, hero_col2 = st.columns([2.1, 1])

    with hero_col1:
        st.markdown('<div class="hero-pill">Planning a better travel</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-title">Design your dream<br>national parks roadtrip.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-subtitle">'
            'Pick a starting city and a chain of national parks. I’ll map the route, estimate your budget, '
            'score driving risk, and let an AI copilot help you fine-tune the plan.'
            '</div>',
            unsafe_allow_html=True,
        )

        stat_c1, stat_c2, stat_c3 = st.columns(3)
        stat_c1.markdown(
            '<div class="stat-chip">'
            '<div class="stat-chip-label">Parks</div>'
            '<div class="stat-chip-value">8 featured</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        stat_c2.markdown(
            '<div class="stat-chip">'
            '<div class="stat-chip-label">Roadtrip style</div>'
            '<div class="stat-chip-value">Budget → comfy</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        stat_c3.markdown(
            '<div class="stat-chip">'
            '<div class="stat-chip-label">AI copilot</div>'
            '<div class="stat-chip-value">Chat built-in</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with hero_col2:
        st.image(
            "https://images.pexels.com/photos/618833/pexels-photo-618833.jpeg",
            use_column_width=True,
            caption="National parks, scenic routes, and wide-open roads.",
        )

    st.markdown("")  # spacing

    # ---------- SIDEBAR ----------
    st.sidebar.header("Trip Settings")

    city_ids = list(CITIES.keys())
    city_labels = {cid: CITIES[cid].name for cid in city_ids}
    start_city = st.sidebar.selectbox(
        "Start city",
        options=city_ids,
        format_func=lambda cid: f"{city_labels[cid]} ({cid})",
        index=1,
    )

    st.sidebar.subheader("Parks to visit (order = selection order)")
    park_ids = list(PARKS.keys())
    default_parks = ["RMNP", "ARCH", "ZION", "BRCA"]
    default_parks = [p for p in default_parks if p in park_ids]

    parks_selected = st.sidebar.multiselect(
        "Choose parks",
        options=park_ids,
        default=default_parks,
        format_func=lambda pid: f"{PARKS[pid].name} ({pid})",
        help="The order you select parks is the order they'll be visited.",
    )

    with st.expander("Selected parks details", expanded=False):
        if parks_selected:
            for order, pid in enumerate(parks_selected, start=1):
                p = PARKS[pid]
                st.markdown(
                    f"**{order}. {p.name}** ({pid}, {p.state})  \n"
                    f"*Best months:* {p.best_start_month}–{p.best_end_month}  \n"
                    f"*Phenomenon:* {p.phenomenon}  \n"
                    f"*Notes:* {p.notes}"
                )
        else:
            st.write("No parks selected yet.")

    trip_year = st.sidebar.number_input("Trip year", min_value=2024, max_value=2100, value=2026, step=1)
    trip_month = st.sidebar.number_input("Trip month (1–12)", min_value=1, max_value=12, value=7, step=1)
    trip_days = st.sidebar.number_input("Trip length (days)", min_value=1, max_value=60, value=10, step=1)
    mpg = st.sidebar.number_input("Vehicle MPG", min_value=5.0, max_value=80.0, value=25.0, step=0.5)
    gas_price = st.sidebar.number_input("Gas price ($/gallon)", min_value=1.0, max_value=10.0, value=4.0, step=0.1)

    lodging_style = st.sidebar.selectbox(
        "Lodging style",
        options=["camping", "budget", "nice"],
        format_func=lambda x: {
            "camping": "Camping (cheapest)",
            "budget": "Budget motels",
            "nice": "Nice hotels",
        }[x],
        index=1,
    )

    max_daily_hours = st.sidebar.number_input(
        "Max daily driving hours you're comfortable with",
        min_value=2.0,
        max_value=16.0,
        value=7.0,
        step=0.5,
    )

    plan_button = st.sidebar.button("🚗 Plan my roadtrip")

    # ---------- MAIN CONTENT ----------
    if plan_button:
        if not parks_selected:
            st.error("Please select at least one park.")
            return

        try:
            route = compute_route(start_city, parks_selected)
        except ValueError as e:
            st.error(str(e))
            return

        budget = estimate_budget(
            parks_selected,
            int(trip_days),
            route["total_miles"],
            float(mpg),
            float(gas_price),
            lodging_style,
        )
        risk = overall_trip_risk(route, int(trip_month), float(max_daily_hours))
        summary = generate_trip_summary(
            start_city,
            parks_selected,
            int(trip_year),
            int(trip_month),
            int(trip_days),
            route,
            budget,
            risk,
        )

        ai_text = ai_recommendation(summary, parks_selected, int(trip_month))

        main_col, map_col = st.columns([1.4, 1])

        # ----- LEFT: cards -----
        with main_col:
            # Trip overview
            st.markdown(
                '<div class="glass-card">'
                '<div class="section-kicker">Trip overview</div>'
                '<div class="section-title">Your roadtrip at a glance</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<pre style='white-space: pre-wrap; font-family: inherit; margin-top: 0.6rem;'>{summary}</pre>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")

            # AI insight
            st.markdown(
                '<div class="glass-card" style="margin-top: 0.25rem;">'
                '<div class="section-kicker">AI copilot</div>'
                '<div class="section-title">Trip insight</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='margin-top:0.6rem; font-size:0.95rem;'>{ai_text}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")

            # Segment details
            st.markdown(
                '<div class="glass-card" style="margin-top: 0.4rem;">'
                '<div class="section-kicker">Driving plan</div>'
                '<div class="section-title">Segment details</div>',
                unsafe_allow_html=True,
            )
            seg_rows = []
            for seg in risk["segments"]:
                seg_rows.append({
                    "From": seg["from"],
                    "To": seg["to"],
                    "Distance (miles)": round(seg["distance_miles"], 1),
                    "Drive time (hrs)": round(seg["drive_hours"], 2),
                    "Rest stops": seg["rest_stops"],
                    "Risk (0–100)": round(seg["risk_score"], 1),
                })
            if seg_rows:
                st.dataframe(seg_rows, width="stretch")
            else:
                st.write("No segments to show.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")

            # Chat assistant
            st.markdown(
                '<div class="glass-card" style="margin-top: 0.4rem;">'
                '<div class="section-kicker">Conversation</div>'
                '<div class="section-title">Chat with your trip assistant</div>',
                unsafe_allow_html=True,
            )

            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = [
                    {
                        "role": "assistant",
                        "content": (
                            "Hi! I’m your roadtrip assistant. Ask me anything about this itinerary, "
                            "alternate parks, packing, or driving safety."
                        ),
                    }
                ]

            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Ask a question about your trip...")
            if user_input:
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                parks_text = ", ".join(PARKS[pid].name for pid in parks_selected)
                system_msg = {
                    "role": "system",
                    "content": (
                        "You are a helpful U.S. national parks roadtrip assistant. "
                        "You know about driving safety, basic costs, park seasons, and logistics. "
                        f"The current trip summary is:\n\n{summary}\n\n"
                        f"Selected parks: {parks_text}\n\n"
                        "Give specific, practical advice. If the user asks for something "
                        "outside this trip, you can still answer but stay travel-focused."
                    ),
                }

                chat_history_for_model = [system_msg] + st.session_state.chat_messages

                with st.chat_message("assistant"):
                    with st.spinner("Thinking about your trip..."):
                        reply = call_chat_model(chat_history_for_model)
                    st.markdown(reply)

                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

            st.markdown("</div>", unsafe_allow_html=True)

        # ----- RIGHT: map -----
        with map_col:
            st.markdown(
                '<div class="glass-card">'
                '<div class="section-kicker">Route</div>'
                '<div class="section-title">Map & featured destinations</div>',
                unsafe_allow_html=True,
            )

            df_points, path = build_route_points_and_path(start_city, parks_selected)

            if not df_points.empty:
                midpoint_lat = df_points["lat"].mean()
                midpoint_lon = df_points["lon"].mean()

                route_layer = pdk.Layer(
                    "PathLayer",
                    data=[{"path": path}],
                    get_path="path",
                    get_width=4,
                    get_color=[80, 80, 200],
                    width_min_pixels=2,
                )

                points_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_points,
                    get_position="[lon, lat]",
                    get_radius=40000,
                    get_fill_color="[type == 'Start' ? 0 : 200, type == 'Start' ? 200 : 50, 80]",
                    pickable=True,
                )

                view_state = pdk.ViewState(
                    latitude=midpoint_lat,
                    longitude=midpoint_lon,
                    zoom=4,
                    bearing=0,
                    pitch=0,
                )

                st.pydeck_chart(
                    pdk.Deck(
                        layers=[route_layer, points_layer],
                        initial_view_state=view_state,
                        tooltip={"text": "{name} ({type})"},
                    )
                )
            else:
                st.write("No points to display on the map.")

            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

