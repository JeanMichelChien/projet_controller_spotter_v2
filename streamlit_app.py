from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

# App-level constants and file locations.
APP_TITLE = "Zürich controller spotter"
DATA_PATH = Path("data/station_events.csv")
GITHUB_REPO_URL = "https://github.com/JeanMichelChien/projet_controller_spotter_v2"
WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

DISCLAIMER_TEXT = (
    "These randomized data are presented for educational purposes only and are not "
    "representative of reality. The author of these visualisations is not responsible "
    "for the use of this information by third parties."
)
BASEMAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


@st.cache_data(show_spinner=False)
def load_events(path: str) -> pd.DataFrame:
    """Load and type-normalize the extracted station events table."""
    df = pd.read_csv(path)

    # Fail fast if the extraction schema changed unexpectedly.
    expected_columns = {
        "sent_at_utc",
        "sent_at_ch",
        "weekday_ch",
        "hour_ch",
        "station",
        "match_status",
        "match_method",
        "match_score",
        "longitude",
        "latitude",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in datasource: {sorted(missing)}")

    # All app filters run in local Zurich time.
    df["sent_at_ch"] = (
        pd.to_datetime(df["sent_at_ch"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Zurich")
    )
    df["hour_ch"] = pd.to_numeric(df["hour_ch"], errors="coerce").fillna(-1).astype(int)

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    score = pd.to_numeric(df["match_score"], errors="coerce")
    df["match_score"] = score.astype("Int64")

    weekday_map = {name: idx for idx, name in enumerate(WEEKDAY_ORDER)}
    df["weekday_idx"] = df["weekday_ch"].map(weekday_map)

    return df


def apply_filters(df: pd.DataFrame, day_range: tuple[int, int], hour_range: tuple[int, int]) -> pd.DataFrame:
    """Apply inclusive weekday and hour filters."""
    day_min, day_max = day_range
    hour_min, hour_max = hour_range

    return df[
        (df["weekday_idx"] >= day_min)
        & (df["weekday_idx"] <= day_max)
        & (df["hour_ch"] >= hour_min)
        & (df["hour_ch"] <= hour_max)
    ]


def aggregate_map_points(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filtered events into one map point per station."""
    matched = df[df["match_status"] == "matched"].copy()
    matched = matched.dropna(subset=["longitude", "latitude", "station"])
    # Guard against malformed coordinates in source data that can break map centering.
    matched = matched[
        matched["longitude"].between(5.0, 12.0)
        & matched["latitude"].between(45.0, 49.5)
    ]

    if matched.empty:
        return matched

    grouped = (
        matched.groupby(["station", "latitude", "longitude"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    max_count = max(int(grouped["count"].max()), 1)
    # Keep markers compact for readability.
    grouped["radius"] = (4 + (grouped["count"] / max_count) * 14).round(1)
    ratio = grouped["count"] / max_count

    # Color ramp: green (small) -> orange (medium) -> red (large).
    low = ratio <= 0.5
    hi = ~low

    t_low = (ratio / 0.5).clip(0, 1)
    t_hi = ((ratio - 0.5) / 0.5).clip(0, 1)

    grouped["color_r"] = 0
    grouped["color_g"] = 0
    grouped["color_b"] = 0

    # Green (#22c55e) -> Orange (#f97316)
    grouped.loc[low, "color_r"] = (34 + (249 - 34) * t_low[low]).round(0).astype(int)
    grouped.loc[low, "color_g"] = (197 + (115 - 197) * t_low[low]).round(0).astype(int)
    grouped.loc[low, "color_b"] = (94 + (22 - 94) * t_low[low]).round(0).astype(int)

    # Orange (#f97316) -> Red (#dc2626)
    grouped.loc[hi, "color_r"] = (249 + (220 - 249) * t_hi[hi]).round(0).astype(int)
    grouped.loc[hi, "color_g"] = (115 + (38 - 115) * t_hi[hi]).round(0).astype(int)
    grouped.loc[hi, "color_b"] = (22 + (38 - 22) * t_hi[hi]).round(0).astype(int)

    return grouped


def make_deck(points: pd.DataFrame) -> pdk.Deck:
    """Build the pydeck object with dots and zoom-level station labels."""
    if points.empty:
        # Zurich center fallback
        view_state = pdk.ViewState(
            latitude=47.3769,
            longitude=8.5417,
            zoom=11.5,
            pitch=0,
            bearing=0,
        )
        return pdk.Deck(
            map_style=BASEMAP_STYLE_URL,
            initial_view_state=view_state,
            layers=[],
            tooltip={"text": "No matched points for selected filters."},
        )

    view_state = pdk.ViewState(
        latitude=float(points["latitude"].median()),
        longitude=float(points["longitude"].median()),
        zoom=11.8,
        pitch=0,
        bearing=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[longitude, latitude]",
        get_fill_color="[color_r, color_g, color_b, 180]",
        get_radius="radius",
        radius_units="pixels",
        pickable=True,
        radius_min_pixels=2,
        radius_max_pixels=20,
        stroked=True,
        get_line_color=[30, 30, 30, 200],
        line_width_min_pixels=1,
    )
    label_layer = pdk.Layer(
        "TextLayer",
        data=points,
        get_position="[longitude, latitude]",
        get_text="station",
        get_size=12,
        size_units="pixels",
        get_color=[240, 240, 240, 210],
        get_alignment_baseline="'top'",
        get_pixel_offset="[0, 8]",
        pickable=False,
        min_zoom=12.2,
    )

    return pdk.Deck(
        map_style=BASEMAP_STYLE_URL,
        initial_view_state=view_state,
        layers=[layer, label_layer],
        tooltip={"text": "Station: {station}\nReports: {count}"},
    )


def build_analytics_frames(filtered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build chart-ready aggregates for matched control events."""
    flagged = filtered[filtered["match_status"] == "matched"].copy()
    if flagged.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_month = (
        flagged.assign(
            month_ch=flagged["sent_at_ch"]
            .dt.tz_localize(None)
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        .groupby("month_ch", as_index=False)
        .size()
        .rename(columns={"size": "controls_flagged"})
        .sort_values("month_ch")
    )
    by_weekday = (
        flagged.groupby("weekday_idx", as_index=False)
        .size()
        .rename(columns={"size": "controls_flagged"})
    )
    by_weekday = (
        pd.DataFrame({"weekday_idx": range(7), "weekday_ch": WEEKDAY_ORDER})
        .merge(by_weekday, on="weekday_idx", how="left")
    )
    by_weekday["controls_flagged"] = by_weekday["controls_flagged"].fillna(0).astype(int)

    return by_month, by_weekday


def build_top_stations_frame(filtered: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Return the top matched stations by flagged controls."""
    flagged = filtered[filtered["match_status"] == "matched"].copy()
    if flagged.empty:
        return pd.DataFrame()

    return (
        flagged.groupby("station", as_index=False)
        .size()
        .rename(columns={"size": "controls_flagged"})
        .sort_values("controls_flagged", ascending=False)
        .head(top_n)
    )


def build_hour_weekday_frame(filtered: pd.DataFrame) -> pd.DataFrame:
    """Build an hour x weekday matrix source for heatmap rendering."""
    flagged = filtered[filtered["match_status"] == "matched"].copy()
    if flagged.empty:
        return pd.DataFrame()

    by_hour_day = (
        flagged.groupby(["weekday_idx", "weekday_ch", "hour_ch"], as_index=False)
        .size()
        .rename(columns={"size": "controls_flagged"})
    )

    all_days = pd.DataFrame({"weekday_idx": range(7), "weekday_ch": WEEKDAY_ORDER})
    all_hours = pd.DataFrame({"hour_ch": range(24)})
    grid = all_days.merge(all_hours, how="cross")

    merged = grid.merge(
        by_hour_day,
        on=["weekday_idx", "weekday_ch", "hour_ch"],
        how="left",
    )
    merged["controls_flagged"] = merged["controls_flagged"].fillna(0).astype(int)
    return merged


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Small style block for a readable app header.
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
        .app-header {
            background: linear-gradient(90deg, #0f172a 0%, #1d4ed8 100%);
            color: white;
            padding: 1rem 1.2rem;
            border-radius: 0.8rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="app-header">
            <h2 style="margin:0;">{APP_TITLE}</h2>
            <div style="opacity:0.9; margin-top:0.25rem;">Aggregated station signals derived from Telegram messages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"[GitHub repository]({GITHUB_REPO_URL})")

    if not DATA_PATH.exists():
        st.error(
            "Datasource missing. Build it first with: "
            "`python scripts/build_station_events.py --input exports/messages.jsonl "
            "--stations data/zurich_stations.csv --output data/station_events.csv`"
        )
        st.stop()

    df = load_events(str(DATA_PATH))

    st.sidebar.header("Filters")
    day_labels = st.sidebar.select_slider(
        "Day of week",
        options=WEEKDAY_ORDER,
        value=(WEEKDAY_ORDER[0], WEEKDAY_ORDER[-1]),
    )
    day_range = (WEEKDAY_ORDER.index(day_labels[0]), WEEKDAY_ORDER.index(day_labels[1]))
    hour_range = st.sidebar.slider(
        "Hour of day",
        min_value=0,
        max_value=23,
        value=(0, 23),
    )

    selected_days = WEEKDAY_ORDER[day_range[0] : day_range[1] + 1]
    st.sidebar.caption(f"Selected days: {', '.join(selected_days)}")

    # Apply filters before both metrics and map so all views stay consistent.
    filtered = apply_filters(df, day_range, hour_range)
    points = aggregate_map_points(filtered)

    total_filtered = len(filtered)
    matched_filtered = int((filtered["match_status"] == "matched").sum())
    unmatched_filtered = total_filtered - matched_filtered
    stations_shown = int(points["station"].nunique()) if not points.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events (filtered)", f"{total_filtered:,}")
    c2.metric("Matched", f"{matched_filtered:,}")
    c3.metric("Unmatched", f"{unmatched_filtered:,}")
    c4.metric("Stations on map", f"{stations_shown:,}")

    day_text = (
        selected_days[0]
        if len(selected_days) == 1
        else f"{selected_days[0]} to {selected_days[-1]}"
    )
    hour_text = f"{hour_range[0]:02d}:00 to {hour_range[1]:02d}:59"
    latest_dt = filtered["sent_at_ch"].dropna().max() if total_filtered > 0 else None
    if latest_dt is not None:
        st.write(
            "Showing control events for "
            f"{day_text}, between {hour_text}. "
            f"Latest event in selection: {latest_dt.strftime('%Y-%m-%d %H:%M')}."
        )
    else:
        st.write(
            "Showing control events for "
            f"{day_text}, between {hour_text}. "
            "No events match the current filters."
        )

    tab_map, tab_analytics = st.tabs(["Map", "Analytics"])

    with tab_map:
        st.pydeck_chart(make_deck(points), use_container_width=True)

        show_table = st.checkbox("Show filtered events table", value=False)
        if show_table:
            display_cols = [
                "sent_at_ch",
                "weekday_ch",
                "hour_ch",
                "station",
                "match_status",
                "match_method",
                "match_score",
                "longitude",
                "latitude",
            ]
            st.dataframe(filtered[display_cols], use_container_width=True, height=420)

    with tab_analytics:
        st.caption("Controls flagged are counted from matched events after applying current filters.")
        by_month, by_weekday = build_analytics_frames(filtered)
        if by_month.empty:
            st.info("No matched control events for the current filters.")
        else:
            st.subheader("Controls flagged per month")
            st.line_chart(
                by_month.set_index("month_ch")["controls_flagged"],
                use_container_width=True,
            )

            st.subheader("Controls flagged per day of week")
            weekday_chart = (
                alt.Chart(by_weekday)
                .mark_bar()
                .encode(
                    x=alt.X("weekday_ch:N", title="Day of week", sort=WEEKDAY_ORDER),
                    y=alt.Y("controls_flagged:Q", title="Controls flagged"),
                    tooltip=[
                        alt.Tooltip("weekday_ch:N", title="Day"),
                        alt.Tooltip("controls_flagged:Q", title="Controls"),
                    ],
                    color=alt.Color(
                        "controls_flagged:Q",
                        title="Controls flagged",
                        scale=alt.Scale(scheme="orangered"),
                    ),
                )
                .properties(height=280)
            )
            st.altair_chart(weekday_chart, use_container_width=True)

            st.subheader("Top control stations")
            top_stations = build_top_stations_frame(filtered, top_n=15)
            top_chart = (
                alt.Chart(top_stations)
                .mark_bar()
                .encode(
                    x=alt.X("controls_flagged:Q", title="Controls flagged"),
                    y=alt.Y("station:N", sort="-x", title="Station"),
                    tooltip=[
                        alt.Tooltip("station:N", title="Station"),
                        alt.Tooltip("controls_flagged:Q", title="Controls"),
                    ],
                    color=alt.Color(
                        "controls_flagged:Q",
                        title="Controls flagged",
                        scale=alt.Scale(scheme="orangered"),
                    ),
                )
                .properties(height=420)
            )
            st.altair_chart(top_chart, use_container_width=True)

            st.subheader("Top control time per day")
            hour_weekday = build_hour_weekday_frame(filtered)
            heatmap = (
                alt.Chart(hour_weekday)
                .mark_rect(cornerRadius=1)
                .encode(
                    x=alt.X("hour_ch:O", title="Hour of day", sort=list(range(24))),
                    y=alt.Y("weekday_ch:O", title="Day of week", sort=WEEKDAY_ORDER),
                    color=alt.Color(
                        "controls_flagged:Q",
                        title="Controls flagged",
                        scale=alt.Scale(scheme="orangered"),
                    ),
                    tooltip=[
                        alt.Tooltip("weekday_ch:N", title="Day"),
                        alt.Tooltip("hour_ch:Q", title="Hour"),
                        alt.Tooltip("controls_flagged:Q", title="Controls"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(heatmap, use_container_width=True)

    st.divider()
    st.caption(DISCLAIMER_TEXT)


if __name__ == "__main__":
    main()
