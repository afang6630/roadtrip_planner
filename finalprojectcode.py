import math
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import pydeck as pdk


# ---------- Data Models ----------

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


# ---------- Static Data (sample parks & cities) ----------

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


# ---------- Utility Functions ----------

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points in miles.
    This is an approximation of road distance, not exact driving distance.
    """
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def estimate_drive_hours(distance_miles: float, avg_speed_mph: float = 55.0) -> float:
    return distance_miles / avg_speed_mph


# ---------- Phenomenon Timing / "AI" Logic ----------

def phenomenon_alignment_score(park: Park, trip_month: int) -> float:
    """
    Score from 0–1 for how well the chosen month lines up with the park's best window.
    """
    start, end = park.best_start_month, park.best_end_month
    if start <= end:
        in_window = start <= trip_month <= end
        distance = 0 if in_window else min(abs(trip_month - start), abs(trip_month - end))
    else:
        # window wraps (e.g., Nov–Feb)
        in_window = trip_month >= start or trip_month <= end
        if in_window:
            distance = 0
        else:
            d1 = (start - trip_month) % 12
            d2 = (trip_month - end) % 12
            distance = min(d1, d2)

    if distance == 0:
        return 1.0
    # decay score: each month away reduces alignment
    return max(0.0, 1.0 - distance / 4.0)


def predict_peak_window(park: Park, year: int) -> Tuple[date, date]:
    """
    Simple peak window prediction: middle of the best months, +/- a few days.
    """
    mid_month = (park.best_start_month + park.best_end_month) / 2
    mid_month_int = int(round(mid_month))
    start = date(year, mid_month_int, 10)
    end = date(year, mid_month_int, 20)
    return start, end


# ---------- Routing / Mileage / Rest Stops ----------

def plan_segment(start_lat: float, start_lon: float,
                 end_lat: float, end_lon: float) -> Tuple[float, float, int]:
    """
    Compute one segment: (distance_miles, drive_hours, suggested_rest_stops)
    """
    dist = haversine_miles(start_lat, start_lon, end_lat, end_lon)
    hours = estimate_drive_hours(dist)
    # Simple rest-stop logic: 1 stop every 3 hours of driving
    rest_stops = int(hours // 3)
    return dist, hours, rest_stops


def compute_route(start_city_id: str, park_ids: List[str]) -> Dict:
    """
    Compute distances, hours, and rest stops for a route from a start city
    through parks in the given order.
    """
    if start_city_id not in CITIES:
        raise ValueError("Unknown start city")
    if not park_ids:
        raise ValueError("No parks provided")

    segments = []
    total_miles = 0.0
    total_hours = 0.0

    # start city to first park
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

    # between parks
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


# ---------- Budgeting Logic ----------

def estimate_budget(
    park_ids: List[str],
    trip_days: int,
    total_miles: float,
    mpg: float,
    gas_price: float,
    lodging_style: str = "camping",  # "camping", "budget", "nice"
    food_per_day: float = 40.0,
) -> Dict:
    """
    Estimate total budget for the trip.
    """
    if not park_ids:
        raise ValueError("No parks")

    # Fuel
    fuel_cost = (total_miles / mpg) * gas_price

    # Lodging
    if lodging_style == "camping":
        nightly_rate = 30.0
    elif lodging_style == "budget":
        nightly_rate = 80.0
    else:
        nightly_rate = 150.0

    lodging_cost = nightly_rate * max(trip_days - 1, 0)

    # Food
    food_cost = food_per_day * trip_days

    # Park entrance fees (unique parks)
    park_fees = sum(PARKS[p].entrance_fee for p in set(park_ids))

    # Misc buffer (gear, snacks, etc.)
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


# ---------- Risk Scoring ("Flight Risk" → Roadtrip Risk) ----------

def risk_score_for_segment(hours: float, month: int) -> float:
    """
    Very simple risk scoring:
    - Long drives increase risk
    - Winter months add extra risk (snow/ice)
    """
    base = 20.0

    # add risk for long drives
    if hours > 4:
        base += (hours - 4) * 5  # +5 per extra hour

    # winter risk (snow/ice)
    if month in (12, 1, 2):
        base += 20

    # clamp 0–100
    return max(0.0, min(100.0, base))


def overall_trip_risk(route_info: Dict, trip_month: int, max_daily_hours: float) -> Dict:
    """
    Compute per-segment and overall risk.
    """
    segment_risks = []
    for seg in route_info["segments"]:
        seg_hours = seg["drive_hours"]
        seg_risk = risk_score_for_segment(seg_hours, trip_month)

        # penalty if above user's preferred max
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


# ---------- "AI" Trip Summary Generator ----------

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
    """
    Generate a human-friendly narrative summary of the roadtrip.
    This mimics an AI assistant explaining your plan.
    """
    city = CITIES[start_city_id]
    park_names = [PARKS[p].name for p in park_ids]
    total_miles = route_info["total_miles"]
    total_hours = route_info["total_hours"]

    # risk text
    risk_text = "low"
    if risk_info["avg_risk"] > 70:
        risk_text = "high"
    elif risk_info["avg_risk"] > 40:
        risk_text = "medium"

    # pick the most seasonally aligned park for the chosen month
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


# ---------- Map Helper ----------

def build_route_points_and_path(start_city_id: str, park_ids: List[str]) -> Tuple[pd.DataFrame, list]:
    """
    Build a dataframe of points (start + parks) and a path list for mapping.
    path is a list of [lon, lat] in order.
    """
    city = CITIES[start_city_id]
    points = []

    # start city
    points.append({
        "name": city.name,
        "type": "Start",
        "order": 0,
        "lat": city.lat,
        "lon": city.lon,
    })

    # parks in order
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


# ---------- Streamlit App ----------

def main():
    st.set_page_config(page_title="Roadtrip Planner", layout="wide")
    st.title("🛣️ National Parks Roadtrip Planner")

    st.markdown(
        "Pick your **start city** and the **parks** you want to visit (in order), "
        "and I'll estimate driving, budget, risk — and show your route on a map."
    )

    # Sidebar inputs
    st.sidebar.header("Trip Settings")

    # Start city
    city_ids = list(CITIES.keys())
    city_labels = {cid: CITIES[cid].name for cid in city_ids}
    start_city = st.sidebar.selectbox(
        "Start city",
        options=city_ids,
        format_func=lambda cid: f"{city_labels[cid]} ({cid})",
        index=1,  # default to Denver
    )

    # Visual park picker with ordering
    st.sidebar.subheader("Parks to visit (order = selection order)")

    park_ids = list(PARKS.keys())
    parks_default = ["RMNP", "ARCH", "ZION", "BRCA"]
    parks_default = [p for p in parks_default if p in park_ids]

    parks_selected = st.sidebar.multiselect(
        "Choose parks",
        options=park_ids,
        default=parks_default,
        format_func=lambda pid: f"{PARKS[pid].name} ({pid})",
        help="The order you click/select parks is the order they'll be visited.",
    )

    # Show details of selected parks
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

    # Other trip params
    trip_year = st.sidebar.number_input("Trip year", min_value=2024, max_value=2100, value=2026, step=1)
    trip_month = st.sidebar.number_input("Trip month (1-12)", min_value=1, max_value=12, value=7, step=1)
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

        # Layout: summary + table + map
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("Trip Summary")
            st.text(summary)

            st.subheader("Segment Details")
            seg_rows = []
            for seg in risk["segments"]:
                seg_rows.append({
                    "From": seg["from"],
                    "To": seg["to"],
                    "Distance (miles)": round(seg["distance_miles"], 1),
                    "Drive time (hrs)": round(seg["drive_hours"], 2),
                    "Rest stops": seg["rest_stops"],
                    "Risk (0-100)": round(seg["risk_score"], 1),
                })
            if seg_rows:
                st.dataframe(seg_rows, use_container_width=True)
            else:
                st.write("No segments to show.")

        with col2:
            st.subheader("Route Map")
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


if __name__ == "__main__":
    main()
