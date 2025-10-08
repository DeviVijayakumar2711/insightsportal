# app.py — single, fast HA/HB/HC app with a visible sidebar chatbot (cloud-first, smarter assistant)
import os
import re
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

APP_VERSION = "v-weekpicker-generic-alarm-2025-09-22-final-clean-ui-thresholds-assistant-v2"
st.set_page_config(page_title="Alarm Insights Dashboard", page_icon="📊", layout="wide")

ALARM_MAP = {
    "HA": {"long": "Harsh Acceleration", "short": "HA"},
    "HB": {"long": "Harsh Braking",      "short": "HB"},
    "HC": {"long": "Harsh Cornering",    "short": "HC"},
}

# ---------- config helpers (ENV-first, no TOML warning) ----------
_SECRETS_CANDIDATES = [
    Path(".streamlit/secrets.toml"),
    Path.home() / ".streamlit" / "secrets.toml",
]

def _has_secrets_file() -> bool:
    return any(p.exists() for p in _SECRETS_CANDIDATES)

def _cfg(key: str, default: str = "") -> str:
    """
    1) Environment variables (Azure App Settings / .env)
    2) .streamlit/secrets.toml (ONLY if file exists; avoids Streamlit warning)
    3) provided default
    """
    v = os.environ.get(key)
    if v not in (None, ""):
        return str(v).strip()

    if _has_secrets_file():
        try:
            return str(st.secrets.get(key, default)).strip()
        except Exception:
            pass

    return str(default).strip()

def _maybe_mtime(_: str) -> float:
    # Avoids cache busting from mtime when using URLs
    return 0.0


# ---------- AI (optional) ----------
try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    AzureChatOpenAI = None

@st.cache_resource
def initialize_llm():
    endpoint = _cfg("AZURE_ENDPOINT").rstrip("/")
    api_key  = _cfg("OPENAI_API_KEY")
    deploy   = _cfg("AZURE_DEPLOYMENT")
    api_ver  = _cfg("AZURE_API_VERSION", "2024-02-15-preview")

    if not (endpoint and api_key and deploy) or AzureChatOpenAI is None:
        return None
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            openai_api_key=api_key,
            azure_deployment=deploy,
            api_version=api_ver,
            temperature=0.1,
            max_retries=2,
        )
        _ = llm.predict("ping")
        return llm
    except Exception:
        return None


# ---------- Load & normalize once ----------
@st.cache_data
def load_and_process_data(telematics_file, exclusions_file, headcounts_file, file_mtime, _v=APP_VERSION):
    df_raw = pd.read_csv(telematics_file)
    df_excl = pd.read_csv(exclusions_file)
    df_head = pd.read_csv(headcounts_file)

    # Normalize text
    for c in ["bus_no","depot_id","svc_no","driver_id","alarm_type"]:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].astype(str).str.strip()
    if "driver_id" in df_raw.columns:
        df_raw["driver_id"] = df_raw["driver_id"].str.replace(r"\.0$", "", regex=True)

    # Trip id norm
    if "trip_id" not in df_raw.columns:
        df_raw["trip_id"] = None
    df_raw["trip_id_norm"] = df_raw["trip_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    # Exclusions
    excl = set(df_excl["bus_no"].astype(str).str.strip()) if not df_excl.empty else set()
    if "bus_no" in df_raw.columns and excl:
        df_raw = df_raw[~df_raw["bus_no"].astype(str).str.strip().isin(excl)]

    # Dates & durations
    df_raw["trip_departure_dt"] = pd.to_datetime(df_raw.get("trip_departure_datetime"), dayfirst=True, errors="coerce")
    df_raw["trip_arrival_dt"]   = pd.to_datetime(df_raw.get("trip_arrival_datetime"),    dayfirst=True, errors="coerce")
    df_raw["alarm_date"]        = pd.to_datetime(df_raw.get("alarm_calendar_date"),      dayfirst=True, errors="coerce")
    df_raw["trip_duration_hr"]  = ((df_raw["trip_arrival_dt"] - df_raw["trip_departure_dt"]).dt.total_seconds()/3600).fillna(0.0)
    df_raw["driver_id_num"]     = pd.to_numeric(df_raw.get("driver_id"), errors="coerce")

    # Completed flag
    completed_col = next((c for c in ["is_week_completed","is_week"] if c in df_raw.columns), None)
    if completed_col:
        df_raw["is_week_completed_flag"] = df_raw[completed_col].astype(str).str.strip().str.upper().isin(["Y","YES","TRUE","1"])
    else:
        df_raw["is_week_completed_flag"] = False

    # Year/Week (alarm)
    if "week_of_year" in df_raw.columns:
        df_raw["alarm_week"] = pd.to_numeric(df_raw["week_of_year"], errors="coerce").astype("Int64")
    else:
        df_raw["alarm_week"] = pd.Series([pd.NA]*len(df_raw), dtype="Int64")
    alarm_iso = df_raw["alarm_date"].dt.isocalendar()
    df_raw["alarm_year"] = alarm_iso.year.astype("Int64")
    df_raw.loc[df_raw["alarm_week"].isna(), "alarm_week"] = alarm_iso.week.astype("Int64")

    # Helpers + categoricals
    df_raw["alarm_up"]      = df_raw["alarm_type"].astype(str).str.strip().str.upper().astype("category")
    df_raw["depot_id_norm"] = df_raw["depot_id"].astype str).str.strip().astype("category")  # noqa: fixed below
    # ^^ (typo guard) we'll reassign correctly:
    df_raw["depot_id_norm"] = df_raw["depot_id"].astype(str).str.strip().astype("category")
    df_raw["bus_no"]        = df_raw["bus_no"].astype("category")
    df_raw["svc_no"]        = df_raw["svc_no"].astype("category")
    df_raw["driver_id"]     = df_raw["driver_id"].astype("category")

    # Headcounts
    if "depot_id" in df_head.columns:
        df_head["depot_id"] = df_head["depot_id"].astype(str).str.strip()

    # Pre-aggregate weekly for all alarms (fast switching)
    weekly_all = (
        df_raw[df_raw["alarm_date"].notna()]
        .groupby(["alarm_up","depot_id_norm","alarm_year","alarm_week"], as_index=False)["trip_id_norm"]
        .count().rename(columns={"trip_id_norm": "alarm_sum"})
    )
    return df_head, df_raw, weekly_all


# ---------- Fast slices ----------
@st.cache_data
def slice_by_filters(df_raw: pd.DataFrame, weekly_all: pd.DataFrame,
                     depots_tuple: tuple, alarm_choice: str,
                     only_completed: bool, exclude_null_driver: bool):
    depots_set = set(depots_tuple)

    mask = (
        (df_raw["alarm_up"] == alarm_choice) &
        (df_raw["depot_id_norm"].isin(depots_set)) &
        (df_raw["alarm_date"].notna())
    )
    df_alarm = df_raw.loc[mask, ["trip_id_norm","driver_id","bus_no","svc_no",
                                 "depot_id_norm","alarm_year","alarm_week",
                                 "is_week_completed_flag","trip_duration_hr"]]

    if only_completed:
        df_alarm = df_alarm.loc[df_alarm["is_week_completed_flag"]]
    if exclude_null_driver:
        df_alarm = df_alarm[df_alarm["driver_id"].notna() & (df_alarm["driver_id"].astype(str) != "")]

    # Week map
    if df_alarm.empty:
        week_map = pd.DataFrame(columns=["alarm_year","alarm_week","label"])
    else:
        week_map = (df_alarm[["alarm_year","alarm_week"]]
                    .dropna().drop_duplicates().sort_values(["alarm_year","alarm_week"]))
        week_map["label"] = week_map.apply(lambda r: f"W{int(r['alarm_week'])} · {int(r['alarm_year'])}", axis=1)

    # Weekly agg
    wk_mask = (weekly_all["alarm_up"] == alarm_choice) & (weekly_all["depot_id_norm"].isin(depots_set))
    weekly = (weekly_all.loc[wk_mask, ["alarm_year","alarm_week","alarm_sum"]]
              .groupby(["alarm_year","alarm_week"], as_index=False)["alarm_sum"].sum())

    return df_alarm, week_map, weekly


@st.cache_data
def per_week_kpis(weekly: pd.DataFrame, headcounts: pd.DataFrame, depots_tuple: tuple):
    head_sel = headcounts[headcounts["depot_id"].isin(depots_tuple)]
    total_headcount = float(head_sel["headcount"].sum()) if not head_sel.empty else 0.0
    if total_headcount <= 0:
        return pd.DataFrame(), total_headcount
    weekly_sum = weekly.copy()
    weekly_sum["per_bc"] = weekly_sum["alarm_sum"] / total_headcount
    weekly_sum["start_of_week"] = pd.to_datetime(
        weekly_sum["alarm_year"].astype(int).astype(str) + weekly_sum["alarm_week"].astype(int).astype(str) + "1",
        format="%G%V%w", errors="coerce"
    )
    return weekly_sum, total_headcount


# ---------- Metrics ----------
def calc_metrics(ev_df: pd.DataFrame, trips_df: pd.DataFrame, category: str) -> pd.DataFrame:
    if ev_df.empty:
        return pd.DataFrame()

    counts = ev_df.groupby(category, sort=False).size().rename("Alarm Count")
    trips_unique = (
        ev_df[["trip_id_norm", category]]
        .drop_duplicates()
        .groupby(category, sort=False)["trip_id_norm"]
        .nunique()
        .rename("Alarm Trips")
    )

    if not trips_df.empty:
        td = trips_df.drop_duplicates(subset=["trip_id_norm"])[[category, "trip_id_norm", "trip_duration_hr"]]
        dur = (
            td.groupby(category, sort=False)["trip_duration_hr"]
            .sum()
            .rename("Total Duration (hr)")
        )
    else:
        dur = pd.Series(0.0, index=counts.index, name="Total Duration (hr)")

    df = pd.concat([counts, trips_unique, dur], axis=1).fillna({"Alarm Trips": 0, "Total Duration (hr)": 0.0})
    df = df.reset_index().rename(columns={'index': category})
    
    trips_nonzero = df["Alarm Trips"].replace({0: pd.NA})
    df["Alarms per Trip"] = (df["Alarm Count"] / trips_nonzero).fillna(0.0)
    dur_nz = df["Total Duration (hr)"].where(df["Total Duration (hr)"] > 0)
    df["Alarms per Hour"] = (df["Alarm Count"] / dur_nz).fillna(0.0)
    return df


# ---------- Trend chart (dynamic thresholds) ----------
def trend_chart(df_agg: pd.DataFrame, y_col: str, title_metric: str, alarm_code: str):
    fig = go.Figure()
    if df_agg.empty:
        return fig

    # thresholds by alarm
    if str(alarm_code).upper() == "HB":
        lo, hi = 2.0, 4.0
    else:  # HA, HC
        lo, hi = 3.0, 5.0

    max_y = max(hi + 1, float(df_agg[y_col].max()) * 1.1)

    # background bands
    fig.add_hrect(y0=0,   y1=lo, line_width=0, fillcolor="green",  opacity=0.10, layer="below")
    fig.add_hrect(y0=lo,  y1=hi, line_width=0, fillcolor="yellow", opacity=0.10, layer="below")
    fig.add_hrect(y0=hi,  y1=max_y, line_width=0, fillcolor="red", opacity=0.10, layer="below")

    # series
    fig.add_trace(go.Scatter(
        x=df_agg["week_label"], y=df_agg[y_col],
        mode="lines+markers+text", name=title_metric,
        line=dict(color="#0072C6", width=3),
        text=df_agg[y_col].round(2), textposition="top center",
        textfont=dict(color="black", size=12),
        hovertemplate="Week: %{x}<br>"+title_metric+": %{y:.2f}<extra></extra>"
    ))

    # dotted threshold lines
    fig.add_hline(y=lo, line_width=1.5, line_dash="dash", line_color="green")
    fig.add_hline(y=hi, line_width=1.5, line_dash="dash", line_color="red")

    fig.update_layout(
        title_text="12-Week Trend: " + title_metric,
        yaxis_title=title_metric, xaxis_title="Week",
        yaxis_range=[0, max_y], showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20), height=400
    )
    return fig


# ---------- AI deep dive ----------
@st.cache_data(show_spinner=False)
def generate_ai_deep_dive(_llm, alarm_code, w1_metric, delta, driver_perf, bus_perf, svc_perf, fleet_avg_per_trip, _v=APP_VERSION):
    if _llm is None:
        return "AI model is not available."

    def slim(df, keep_cols, n=5):
        if df is None or df.empty:
            return []
        d = df[keep_cols].copy()
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                d[c] = d[c].round(3)
        return d.head(n).to_dict(orient="records")

    payload = {
        "alarm_code": alarm_code,
        "overall_avg_per_bc": round(float(w1_metric), 3),
        "delta_vs_prev_week": round(float(delta), 3) if delta is not None else None,
        "fleet_avg_per_trip": round(float(fleet_avg_per_trip), 3) if fleet_avg_per_trip is not None else None,
        "top_drivers_by_per_trip": slim(driver_perf.sort_values("Alarms per Trip", ascending=False),
                                         ["driver_id","Alarms per Trip","Alarm Count","Alarm Trips"]),
        "top_buses_by_per_trip":    slim(bus_perf.sort_values("Alarms per Trip",   ascending=False),
                                         ["bus_no","Alarms per Trip","Alarm Count","Alarm Trips"]),
        "top_services_by_per_trip": slim(svc_perf.sort_values("Alarms per Trip",   ascending=False),
                                         ["svc_no","Alarms per Trip","Alarm Count","Alarm Trips"]),
    }
    try:
        return _llm.predict(
            f"""
You are an expert Bus Operations Analyst. Alarm type: {alarm_code}.
Using ONLY this JSON summary, produce insights.

DATA:
{payload}

Write markdown with EXACT sections:
### Executive Summary
### Actionable Anomaly Detection
- Nexus of Risk (cross-category)
- Category-Specific Insights (compare to fleet_avg_per_trip)
- Low Workload, High Rate (if any)
### Prioritized Recommendations
Return a 3-row table: Priority | Recommended Action | Data-Driven Rationale.
""".strip()
        )
    except Exception as e:
        return f"AI error: {e}"


# ---------- Chatbot helpers ----------
def _parse_query(q: str) -> Tuple[str, Optional[str], int, Optional[int], Optional[int]]:
    """
    Returns (etype, entity_id_or_text, n_weeks, week, year)
    etype ∈ {'driver','bus','svc','drivers','buses','services','generic'}
    """
    if not q:
        return "generic", None, 12, None, None

    ql = q.strip().lower()

    # Weeks window
    m = re.search(r"last\s+(\d+)\s*weeks?", ql)
    n_weeks = int(m.group(1)) if m else 12
    n_weeks = max(1, min(n_weeks, 52))

    # Specific week / year (optional)
    wk = None; yr = None
    m = re.search(r"(?:^|\b)w(?:eek)?\s*[-_ ]?(\d{1,2})(?:\D|$)", ql)
    if m: wk = int(m.group(1))
    m2 = re.search(r"\b(20\d{2})\b", ql)
    if m2: yr = int(m2.group(1))

    # Plural / fleet scopes
    if "all drivers" in ql or re.search(r"\bdrivers\b", ql):
        return "drivers", None, n_weeks, wk, yr
    if "all buses" in ql or re.search(r"\bbuses\b", ql):
        return "buses", None, n_weeks, wk, yr
    if "all services" in ql or re.search(r"\bservices\b", ql):
        return "services", None, n_weeks, wk, yr

    # Single entities
    for pat, etype in [
        (r"(?:driver|bc|captain)\s*#?\s*([a-z0-9._-]+)", "driver"),
        (r"(?:bus|vehicle)\s*#?\s*([a-z0-9._-]+)", "bus"),
        (r"(?:svc|service)\s*#?\s*([a-z0-9._-]+)", "svc")
    ]:
        m = re.search(pat, ql)
        if m:
            return etype, m.group(1).upper(), n_weeks, wk, yr

    # Generic fleet question
    return "generic", q.strip(), n_weeks, wk, yr


def vector_week_start(year_series, week_series):
    return pd.to_datetime(
        year_series.astype(int).astype(str) + week_series.astype(int).astype(str) + "1",
        format="%G%V%w", errors="coerce"
    )


def _limit_last_n_weeks(df: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """Filter df (with alarm_year/week) to last n weeks available in df."""
    if df.empty:
        return df
    df = df.copy()
    df["sow"] = vector_week_start(df["alarm_year"], df["alarm_week"])
    latest = df["sow"].max()
    if pd.isna(latest):
        return df
    cutoff = latest - pd.Timedelta(weeks=n_weeks - 1)
    return df[df["sow"] >= cutoff]


def answer_entity_question(q: str, alarm_choice: str, df_alarm: pd.DataFrame, llm=None):
    etype, eid, n_weeks, wk, yr = _parse_query(q)

    # --- Single entity (driver/bus/svc) ---
    if etype in {"driver","bus","svc"}:
        col_map = {"driver": "driver_id", "bus": "bus_no", "svc": "svc_no"}
        col = col_map[etype]
        subset = df_alarm[df_alarm[col].astype(str).str.upper() == str(eid).upper()].copy()
        if subset.empty:
            return f"No {ALARM_MAP[alarm_choice]['short']} events found for **{etype} {eid}** with current filters.", pd.DataFrame()

        subset["start_of_week"] = vector_week_start(subset["alarm_year"], subset["alarm_week"])
        if wk is not None:
            if yr is None:
                yr = int(pd.to_numeric(subset["alarm_year"], errors="coerce").dropna().max())
            sow = vector_week_start(pd.Series([yr]), pd.Series([wk]))[0]
            subset = subset[subset["start_of_week"] == sow]
            if subset.empty:
                return f"No events for **{etype} {eid}** in **W{wk} {yr}**.", pd.DataFrame()

        # window: last n weeks
        subset_win = _limit_last_n_weeks(subset, n_weeks)

        weekly = (subset_win.groupby(["alarm_year","alarm_week"], as_index=False)
                            .size().rename(columns={"size":"Events"}))
        weekly["start_of_week"] = vector_week_start(weekly["alarm_year"], weekly["alarm_week"])
        weekly = weekly.sort_values("start_of_week")
        weekly["Week"] = "W" + weekly["alarm_week"].astype(int).astype(str) + " · " + weekly["alarm_year"].astype(int).astype(str)
        weekly_table = weekly[["Week","Events"]]

        trips_unique = subset_win.drop_duplicates(subset=["trip_id_norm"])
        total_events = int(len(subset_win))
        total_trips = int(trips_unique["trip_id_norm"].nunique())
        total_hours = float(trips_unique["trip_duration_hr"].sum())
        per_trip = (total_events / total_trips) if total_trips > 0 else 0.0
        per_hour = (total_events / total_hours) if total_hours > 0 else 0.0

        peak_row = weekly.sort_values("Events", ascending=False).head(1)
        peak_txt = ""
        if not peak_row.empty:
            pw = int(peak_row["alarm_week"].iloc[0]); py = int(peak_row["alarm_year"].iloc[0]); pc = int(peak_row["Events"].iloc[0])
            peak_txt = f" Peak: **W{pw} {py}** with **{pc}** events."

        answer = (
            f"**{ALARM_MAP[alarm_choice]['short']}** for **{etype} {eid}** (current depots/filters):\n"
            f"- Window: last **{n_weeks}** weeks\n"
            f"- Total events: **{total_events}** across **{total_trips}** trips\n"
            f"- Alarms per Trip: **{per_trip:.2f}** | Alarms per Hour: **{per_hour:.2f}**{peak_txt}"
        )
        return answer, weekly_table

    # --- Fleet scopes (drivers/buses/services) ---
    if etype in {"drivers","buses","services"}:
        scope_col = {"drivers": "driver_id", "buses": "bus_no", "services": "svc_no"}[etype]
        subset = _limit_last_n_weeks(df_alarm, n_weeks)
        if subset.empty:
            return f"No {etype} found in the last {n_weeks} weeks with current filters.", pd.DataFrame()

        trips_unique = subset.drop_duplicates(subset=["trip_id_norm"])
        scope_df = calc_metrics(subset, trips_unique, scope_col)
        cols = [scope_col, "Alarm Count", "Alarm Trips", "Alarms per Trip", "Alarms per Hour", "Total Duration (hr)"]
        scope_top = scope_df.sort_values(["Alarms per Trip","Alarm Count"], ascending=[False, False])[cols].head(15).reset_index(drop=True)
        nice_name = {"driver_id":"Driver", "bus_no":"Bus", "svc_no":"Service"}[scope_col]
        scope_top = scope_top.rename(columns={scope_col: nice_name})
        return f"Top {etype} in the last **{n_weeks}** weeks (ranked by *Alarms per Trip*):", scope_top

    # --- Generic fleet question / summary ---
    if etype == "generic":
        subset = _limit_last_n_weeks(df_alarm, n_weeks)
        if subset.empty:
            return f"No events found in the last {n_weeks} weeks with current filters.", pd.DataFrame()

        total_events = len(subset)
        drivers = subset["driver_id"].nunique()
        buses = subset["bus_no"].nunique()
        svcs = subset["svc_no"].nunique()
        depots = subset["depot_id_norm"].nunique()

        by_depot = (subset.groupby("depot_id_norm")["trip_id_norm"]
                    .count().reset_index().rename(columns={"trip_id_norm":"Events"})
                    .sort_values("Events", ascending=False))
        depot_line = ""
        if not by_depot.empty:
            topd = by_depot.iloc[0]
            depot_line = f"\n- Top depot: **{topd['depot_id_norm']}** with **{int(topd['Events']):,}** events"

        ans = (
            f"**Fleet summary** (last **{n_weeks}** weeks, current filters):\n"
            f"- Events: **{total_events:,}** across **{drivers:,}** drivers, **{buses:,}** buses, **{svcs:,}** services, **{depots:,}** depots"
            f"{depot_line}"
        )

        if initialize_llm() is not None:
            llm = initialize_llm()
            try:
                stats_json = {
                    "n_weeks": n_weeks,
                    "totals": {"events": int(total_events), "drivers": int(drivers), "buses": int(buses), "services": int(svcs), "depots": int(depots)},
                    "by_depot": by_depot.to_dict(orient="records")[:10],
                }
                prompt = f"""You are an SMRT safety analyst. Summarize briefly and clearly:
                Data: {stats_json}
                Question: {q}
                Answer in markdown with a short headline and bullets."""
                ai_ans = llm.predict(prompt)
                return ai_ans, by_depot.rename(columns={"depot_id_norm":"Depot"})
            except Exception:
                pass

        return ans, by_depot.rename(columns={"depot_id_norm":"Depot"})

    # fallback
    return (
        "I couldn’t interpret your question. Try:\n"
        "- `driver 30450 last 4 weeks`\n"
        "- `all buses last 8 weeks`\n"
        "- `which depot has most HA alarms last 12 weeks?`"
    ), pd.DataFrame()


# ---------- App ----------
def main():
    llm = initialize_llm()

    # --- Sidebar chat at the top ---
    chat_box = st.sidebar.container()
    with chat_box:
        st.markdown("### 💬 Assistant — Driver / Bus / Service")
        st.caption(
            "Examples: `driver 30450 last 4 weeks`, `bus SMB123A`, `svc 973`, "
            "`all drivers last 8 weeks`, `all buses`, `which depot has most HA alarms?`"
        )
        if "qa" not in st.session_state:
            st.session_state.qa = []
        q_input = st.text_input("Question", key="qa_input", placeholder="e.g., all drivers last 6 weeks")
        colA, colB = st.columns(2)
        ask_clicked = colA.button("Ask", key="qa_btn")
        clear_clicked = colB.button("Clear", key="qa_clear")
        pending_q = q_input.strip() if (ask_clicked and q_input.strip()) else None
        if clear_clicked:
            st.session_state.qa = []
        st.markdown("---")

    # --- Other sidebar controls (below chat) ---
    with st.sidebar:
        st.markdown("### Alarm Type")
        alarm_choice = st.selectbox(
            "Select alarm", options=list(ALARM_MAP.keys()),
            format_func=lambda x: f"{x} – {ALARM_MAP[x]['long']}",
            index=0, key="alarm_select_fast"
        )
        st.markdown("### ⚙️ Filters")

        tele_file = _cfg("TELEMATICS_URL") or _cfg("TELEMATICS_PATH")
        excl_file = _cfg("EXCLUSIONS_URL") or _cfg("EXCLUSIONS_PATH")
        head_file = _cfg("HEADCOUNTS_URL") or _cfg("HEADCOUNTS_PATH")

        if not tele_file or not excl_file or not head_file:
            st.error("Data file paths are not configured. Please set TELEMATICS_URL/PATH, EXCLUSIONS_URL/PATH, and HEADCOUNTS_URL/PATH in your environment variables.")
            return

        file_mtime = max(_maybe_mtime(tele_file), _maybe_mtime(excl_file), _maybe_mtime(head_file))

    headcounts, df_raw, weekly_all = load_and_process_data(tele_file, excl_file, head_file, file_mtime)
    depot_choices = sorted(headcounts["depot_id"].unique()) if not headcounts.empty else []
    with st.sidebar.form("filters_form", clear_on_submit=False):
        default_depots = ["WDLAND","KRANJI"] if "WDLAND" in depot_choices else depot_choices[:2]
        depots = st.multiselect("Depot(s)", depot_choices, default=default_depots)
        only_completed = st.checkbox("Only completed weeks (is_week = 'Y')", value=True)
        exclude_null_driver = st.checkbox("Exclude rows with missing/zero driver_id", value=True)
        _ = st.form_submit_button("✅ Apply Filters")

    if not depots:
        _ = st.warning("Please select at least one Depot to view the analysis.")
        return

    depots_tuple = tuple(depots)

    _ = st.title(f"Weekly {ALARM_MAP[alarm_choice]['long']} Review")

    # Fast base slice (cached)
    df_alarm, week_map, weekly = slice_by_filters(
        df_raw, weekly_all, depots_tuple, alarm_choice, only_completed, exclude_null_driver
    )
    if week_map.empty:
        _ = st.info("No events found with the current filters.")
        return

    # --- Process chat AFTER data slice so answers use df_alarm ---
    with chat_box:
        if pending_q:
            ans, tbl = answer_entity_question(pending_q, alarm_choice, df_alarm, llm)
            st.session_state.qa.append({"q": pending_q, "a": ans, "df": tbl})

        for item in st.session_state.qa[-4:]:
            st.markdown(f"**You:** {item['q']}")
            st.markdown(item["a"])
            if isinstance(item.get("df"), pd.DataFrame) and not item["df"].empty:
                st.dataframe(item["df"], height=220, use_container_width=True)
            st.markdown("---")

    # Week picker
    sel_label = st.sidebar.selectbox("Week", options=week_map["label"].tolist(), index=len(week_map)-1, key="week_select_fast")
    sel_row = week_map.loc[week_map["label"] == sel_label].iloc[0]
    w1_year, w1_week = int(sel_row["alarm_year"]), int(sel_row["alarm_week"])

    # KPIs (cached)
    weekly_sum, total_headcount = per_week_kpis(weekly, headcounts, depots_tuple)
    if total_headcount <= 0 or weekly_sum.empty:
        _ = st.error("Selected depots have zero headcount or no weekly data.")
        return

    w1_row = weekly_sum[(weekly_sum["alarm_year"]==w1_year) & (weekly_sum["alarm_week"]==w1_week)]
    if w1_row.empty:
        _ = st.error(f"No data found for Week {w1_week}, {w1_year}.")
        return

    w1_metric = float(w1_row["per_bc"].iloc[0])
    w1_events_count = int(w1_row["alarm_sum"].iloc[0])
    w1_sow = pd.to_datetime(f"{w1_year}{w1_week}1", format="%G%V%w")
    w2_sow = w1_sow - pd.Timedelta(weeks=1)
    w2_year, w2_week = int(w2_sow.isocalendar().year), int(w2_sow.isocalendar().week)
    w2_row = weekly_sum[(weekly_sum["alarm_year"]==w2_year) & (weekly_sum["alarm_week"]==w2_week)]
    if not w2_row.empty:
        delta = w1_metric - float(w2_row["per_bc"].iloc[0]); delta_text = f"{delta:.2f} vs. W{w2_week}"
        w2_events_count = int(w2_row["alarm_sum"].iloc[0])
    else:
        delta = 0.0; delta_text = "No prior week data"; w2_events_count = 0

    _ = st.markdown(f"## 📈 Performance Snapshot: Week {w1_week}, {w1_year}")
    c1, c2 = st.columns(2)
    _ = c1.metric(f"Avg {ALARM_MAP[alarm_choice]['short']} per BC (W{w1_week})", f"{w1_metric:.2f}", delta=delta_text, delta_color="inverse")
    _ = c2.metric(f"Total {ALARM_MAP[alarm_choice]['short']} Incidents", f"{w1_events_count:,.0f}",
                  delta=f"{(w1_events_count - w2_events_count):,.0f}")

    weekly_sum["week_label"] = weekly_sum["start_of_week"].dt.strftime("%d %b") + "<br>W" + weekly_sum["alarm_week"].astype(int).astype(str)
    chart_data = weekly_sum[weekly_sum["start_of_week"] <= w1_sow].tail(12)
    _ = st.plotly_chart(
        trend_chart(
            chart_data, "per_bc",
            f"Avg {ALARM_MAP[alarm_choice]['short']} per Bus Captain",
            alarm_choice
        ),
        use_container_width=True
    )

    # ===== Current-week slices =====
    if df_alarm.empty:
        w1_events = pd.DataFrame(columns=["trip_id_norm","driver_id","bus_no","svc_no","trip_duration_hr"])
        trips_unique = pd.DataFrame(columns=["trip_id_norm","driver_id","bus_no","svc_no","trip_duration_hr"])
    else:
        w1_events = df_alarm.loc[
            (df_alarm["alarm_year"]==w1_year) & (df_alarm["alarm_week"]==w1_week),
            ["trip_id_norm","driver_id","bus_no","svc_no","trip_duration_hr"]
        ]
        trips_unique = w1_events.drop_duplicates(subset=["trip_id_norm"])

    # ----- AI Deep Dive -----
    _ = st.markdown("---")
    _ = st.subheader("🤖 AI-Powered Deep Dive")
    if llm is None:
        _ = st.info("AI model is not configured.")
    elif w1_events.empty:
        _ = st.info(f"No {ALARM_MAP[alarm_choice]['short']} rows for the selected week.")
    else:
        with st.expander("Context for AI (counts)", expanded=False):
            _ = st.write(f"Rows (events): **{len(w1_events):,}** | Unique trips: **{trips_unique['trip_id_norm'].nunique():,}**")
        if st.button("Generate AI Analysis for Selected Week", type="primary"):
            with st.spinner("Generating insights..."):
                driver_perf = calc_metrics(w1_events, trips_unique, "driver_id")
                bus_perf    = calc_metrics(w1_events, trips_unique, "bus_no")
                svc_perf    = calc_metrics(w1_events, trips_unique, "svc_no")

                total_alarm_trips = driver_perf["Alarm Trips"].sum() if not driver_perf.empty else 0
                fleet_avg         = (driver_perf["Alarm Count"].sum() / total_alarm_trips) if total_alarm_trips > 0 else 0.0

                ai_md = generate_ai_deep_dive(llm, alarm_choice, w1_metric, delta, driver_perf, bus_perf, svc_perf, fleet_avg)
                _ = st.markdown(ai_md or "")

    if w1_events.empty:
        _ = st.caption("Note: No matching event rows found. Tables will be empty.")

    # ----- Manual Tables (Streamlit-safe, no inline ternaries) -----
    _ = st.markdown("---")
    _ = st.subheader(f"🔬 Manual Analysis Tables: Week {w1_week}")
    tab1, tab2, tab3 = st.tabs(["**Drivers** 🧑‍✈️", "**Buses** 🚌", "**Services** 🗺️"])

    with tab1:
        df_dr = calc_metrics(w1_events, trips_unique, "driver_id")
        cols_show = ["driver_id", "Alarm Count","Alarm Trips","Alarms per Trip","Alarms per Hour","Total Duration (hr)"]
        if df_dr.empty:
            st.info("No driver events this week.")
        else:
            view = df_dr.sort_values(["Alarms per Trip","Alarm Count"], ascending=[False, False])[cols_show].copy()
            for c in ["Alarm Count","Alarm Trips","Alarms per Trip","Alarms per Hour","Total Duration (hr)"]:
                if c in view.columns:
                    view[c] = pd.to_numeric(view[c], errors="coerce")
            view = view.round(2)
            st.dataframe(view, use_container_width=True)

    with tab2:
        df_bs = calc_metrics(w1_events, trips_unique, "bus_no")
        cols_show = ["bus_no", "Alarm Count","Alarm Trips","Alarms per Trip","Alarms per Hour","Total Duration (hr)"]
        if df_bs.empty:
            st.info("No bus events this week.")
        else:
            view = df_bs.sort_values(["Alarms per Trip","Alarm Count"], ascending=[False, False])[cols_show].copy()
            for c in ["Alarm Count","Alarm Trips","Alarms per Trip","Alarms per Hour","Total Duration (hr)"]:
                if c in view.columns:
                    view[c] = pd.to_numeric(view[c], errors="coerce")
            view = view.round(2)
            st.dataframe(view, use_container_width=True)

    with tab3:
        df_sv = calc_metrics(w1_events, trips_unique, "svc_no")
        cols_show = ["svc_no", "Alarm Count","Alarm Trips","Alarms per Trip","Alarms per Hour","Total Duration (hr)"]
        if df_sv.empty:
            st.info("No service events this week.")
        else:
            view = df_sv.sort_values(["Alarms per Trip","Alarm Count"], ascending=[False, False])[cols_show].copy()
            for c in ["Alarm Count","Alarm Trips","Alarms per Trip","Alarms per Hour","Total Duration (hr)"]:
                if c in view.columns:
                    view[c] = pd.to_numeric(view[c], errors="coerce")
            view = view.round(2)
            st.dataframe(view, use_container_width=True)

    return None


if __name__ == "__main__":
    _ = main()
