# app.py — Fleet Intelligence Hub
# ─────────────────────────────────────────────────────────────────────────────
# CHANGELOG — v-local-agentic-light-2025-genai-v2
# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 · .env credentials always picked up correctly
#   • load_dotenv(override=True) called at MODULE LEVEL and inside
#     initialize_llm() so .env is re-read on every page load.
#   • Removed @st.cache_resource from the outer initialize_llm() —
#     it was caching None on first failure and never retrying.
#   • New _create_llm() is cached by credential VALUES, so cache busts
#     automatically when .env content changes — no restart needed.
#   • "🔄 Reconnect AI" sidebar button forces an immediate retry.
#   • Sidebar shows exact .env path, file-exists check, and which
#     of AZURE_ENDPOINT / OPENAI_API_KEY / AZURE_DEPLOYMENT is missing.
#
# FIX 2 · AI Insight panels always visible (old-format layout)
#   • Tab 1 Weekly Deep Dive  → blue  callout always rendered.
#   • Tab 2 4-Week Pattern    → green callout always rendered.
#   • Tab 3 Risk Forecast     → orange callout always rendered.
#   With AI connected  → full AI narrative paragraph.
#   Without AI         → key data stats + "configure credentials" prompt.
# ─────────────────────────────────────────────────────────────────────────────


import os
import re
import sys
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Suppress torch.classes Streamlit watcher error (harmless but noisy)
# Caused by PyTorch being installed in the same env — not related to this app
try:
    import torch
    # Prevent Streamlit's file watcher from inspecting torch internals
    if hasattr(torch, "_classes"):
        import streamlit.watcher.local_sources_watcher as _lsw
        _orig = _lsw.get_module_paths
        def _safe_get_module_paths(module):
            try: return _orig(module)
            except Exception: return set()
        _lsw.get_module_paths = _safe_get_module_paths
except Exception:
    pass

# Load .env from script directory so it works regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH   = os.path.join(_SCRIPT_DIR, ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

APP_VERSION = "v-azure-2025-genai-v3"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fleet Intelligence Hub",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #F4F6F9;
    color: #1e293b;
  }
  .stApp { background-color: #F4F6F9; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
    border-right: 1px solid #e2e8f0;
  }

  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #0f4c8a; letter-spacing: -0.5px; }
  h1 { font-size: 1.6rem; border-bottom: 2px solid #bfdbfe; padding-bottom: 8px; }

  div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  div[data-testid="metric-container"] label {
    color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f4c8a; font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
  }

  .chat-user {
    background: #eff6ff; border-left: 3px solid #3b82f6;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    margin: 6px 0; font-size: 0.88rem; color: #1e3a5f;
  }
  .chat-bot {
    background: #f0fdf4; border-left: 3px solid #22c55e;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    margin: 6px 0; font-size: 0.88rem; color: #14532d;
  }
  .agent-tool-call {
    background: #fffbeb; border: 1px solid #fcd34d; border-radius: 6px;
    padding: 6px 10px; font-size: 0.75rem; color: #92400e;
    font-family: 'IBM Plex Mono', monospace; margin: 3px 0;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: #e2e8f0; border-radius: 8px; padding: 4px; gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #64748b; font-size: 0.82rem; padding: 6px 16px; border-radius: 6px; font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #0f4c8a !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  }

  .stDataFrame { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }

  .stButton > button {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%);
    color: #ffffff; border: none; border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.5px;
    padding: 8px 20px; transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(30,64,175,0.3);
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
  }

  .alert-red   { background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; padding:12px 16px; margin:8px 0; color:#7f1d1d; }
  .alert-amber { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px; padding:12px 16px; margin:8px 0; color:#78350f; }
  .alert-green { background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:12px 16px; margin:8px 0; color:#14532d; }

  .badge-red   { display:inline-block; background:#dc2626; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; }
  .badge-amber { display:inline-block; background:#d97706; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; }
  .badge-green { display:inline-block; background:#16a34a; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; }
  .badge-blue  { display:inline-block; background:#2563eb; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.7rem; font-family:'IBM Plex Mono',monospace; }

  .section-title {
    font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
    letter-spacing:2px; text-transform:uppercase; color:#94a3b8; padding:4px 0; margin-top:8px;
  }
  .stSelectbox > div > div { background:#ffffff; border-color:#e2e8f0; color:#1e293b; }
  .stMultiSelect > div > div { background:#ffffff; border-color:#e2e8f0; }
  .stTextInput > div > div > input { background:#ffffff; border-color:#e2e8f0; color:#1e293b; }
  .stCheckbox label { color:#475569; font-size:0.82rem; }
  .streamlit-expanderHeader { background:#f8fafc; color:#0f4c8a; font-size:0.82rem; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ALARM_MAP = {
    "HA": {"long": "Harsh Acceleration", "short": "HA", "color": "#f59e0b"},
    "HB": {"long": "Harsh Braking",      "short": "HB", "color": "#ef4444"},
    "HC": {"long": "Harsh Cornering",    "short": "HC", "color": "#a855f7"},
    "HF": {"long": "Fatigue",            "short": "HF", "color": "#06b6d4"},
}

# ── Plotly theme: NO xaxis/yaxis here to avoid keyword conflict ──
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="IBM Plex Sans", color="#475569", size=12),
    margin=dict(l=20, r=20, t=45, b=20),
    height=380,
)
# Applied separately wherever needed
AXIS_STYLE = dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", zerolinecolor="#e2e8f0")

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# SAFE LANGCHAIN IMPORTS
# ─────────────────────────────────────────────
try:
    from langchain_openai import AzureChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_OK = True
except Exception:
    LANGCHAIN_OK       = False
    AzureChatOpenAI    = None
    AgentExecutor      = None
    create_react_agent = None
    PromptTemplate     = None

    class Tool:
        """Dummy so build_agent_tools never raises NameError."""
        def __init__(self, name="", func=None, description=""):
            self.name = name; self.func = func; self.description = description

# ─────────────────────────────────────────────
# DIRECT OPENAI SDK FALLBACK (works without LangChain)
# ─────────────────────────────────────────────
try:
    from openai import AzureOpenAI as _AzureOpenAI
    OPENAI_SDK_OK = True
except Exception:
    _AzureOpenAI  = None
    OPENAI_SDK_OK = False

# ─────────────────────────────────────────────
# CONFIG HELPERS
# ─────────────────────────────────────────────
def get_config(key, default=""):
    """Read from env → Streamlit secrets → default. Works on Azure App Service and locally."""
    val = os.environ.get(key)
    if val: return val.strip()
    try:
        val = st.secrets.get(key)
        if val: return str(val).strip()
    except Exception:
        pass
    return default

def _cfg(key: str, default: str = "") -> str:
    return get_config(key, default)

def _maybe_mtime(p: str) -> float:
    if not p or p.lower().startswith("http"): return 0.0
    try: return os.path.getmtime(p)
    except: return 0.0

def smart_read_csv(path_or_url: str, **kwargs) -> pd.DataFrame:
    """Read CSV from local path or HTTP/SAS URL (Azure Blob Storage)."""
    if not path_or_url: return pd.DataFrame()
    p = str(path_or_url).strip()
    if p.lower().startswith("http"):
        try: return pd.read_csv(p, **kwargs)
        except Exception as e:
            st.error(f"URL read error `{p[:60]}...`: {e}")
            return pd.DataFrame()
    try: return pd.read_csv(p, **kwargs)
    except Exception as e:
        st.error(f"Could not load `{p}`: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
# THIN LLM WRAPPER — uniform .predict() API
# Works with LangChain AzureChatOpenAI OR direct openai SDK
# ─────────────────────────────────────────────
class _DirectAzureLLM:
    """Lightweight wrapper around openai.AzureOpenAI that exposes .predict()
    so the rest of the app works identically whether LangChain is installed or not."""

    def __init__(self, client, deployment: str):
        self._client     = client
        self._deployment = deployment

    def predict(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    # LangChain-compatible alias
    def invoke(self, data: dict) -> dict:
        answer = self.predict(data.get("input", ""))
        return {"output": answer, "intermediate_steps": []}

# ─────────────────────────────────────────────
# LLM INITIALISATION — re-reads .env every call
# Tries LangChain first, falls back to direct openai SDK
# ─────────────────────────────────────────────
@st.cache_resource
def _create_llm_langchain(endpoint: str, api_key: str, deploy: str, api_ver: str):
    """LangChain-backed LLM — cached by credential values."""
    llm = AzureChatOpenAI(
        azure_endpoint=endpoint, openai_api_key=api_key,
        azure_deployment=deploy, api_version=api_ver,
        temperature=0.1, max_retries=2,
    )
    llm.predict("ping")   # verify connectivity
    return llm

@st.cache_resource
def _create_llm_direct(endpoint: str, api_key: str, deploy: str, api_ver: str):
    """Direct openai SDK LLM — cached by credential values."""
    client = _AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_ver,
    )
    wrapper = _DirectAzureLLM(client, deploy)
    wrapper.predict("ping")   # verify connectivity
    return wrapper

def _clear_llm_caches():
    try: _create_llm_langchain.clear()
    except Exception: pass
    try: _create_llm_direct.clear()
    except Exception: pass

def initialize_llm():
    """
    Returns (llm, status_str).
    Re-reads .env every call so credential updates take effect without restart.
    Tries LangChain first; falls back to direct openai SDK if LangChain missing.
    """
    # Re-read .env every call — key fix for credential pickup
    load_dotenv(dotenv_path=_ENV_PATH, override=True)

    endpoint = _cfg("AZURE_ENDPOINT").rstrip("/")
    api_key  = _cfg("OPENAI_API_KEY")
    deploy   = _cfg("AZURE_DEPLOYMENT")
    api_ver  = _cfg("AZURE_API_VERSION", "2024-02-15-preview")

    missing = [k for k, v in {
        "AZURE_ENDPOINT":   endpoint,
        "OPENAI_API_KEY":   api_key,
        "AZURE_DEPLOYMENT": deploy,
    }.items() if not v]
    if missing:
        return None, f"Missing in .env: {', '.join(missing)}"

    # ── Try LangChain first ──
    if LANGCHAIN_OK and AzureChatOpenAI is not None:
        try:
            llm = _create_llm_langchain(endpoint, api_key, deploy, api_ver)
            return llm, "connected (LangChain)"
        except Exception as e:
            _clear_llm_caches()
            # fall through to direct SDK

    # ── Fall back to direct openai SDK ──
    if OPENAI_SDK_OK and _AzureOpenAI is not None:
        try:
            llm = _create_llm_direct(endpoint, api_key, deploy, api_ver)
            return llm, "connected (openai SDK)"
        except Exception as e:
            _clear_llm_caches()
            return None, f"Connection failed: {e}"

    return None, "Neither langchain-openai nor openai package found. Run: pip install openai"

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def _file_accessible(p: str) -> bool:
    """True if path is a URL or an existing local file."""
    if not p: return False
    if str(p).lower().startswith("http"): return True
    return os.path.exists(str(p))

def load_and_process_data(tele_file, excl_file, head_file, model_file, _mtime):
    df_raw   = smart_read_csv(tele_file)
    df_excl  = smart_read_csv(excl_file)  if _file_accessible(excl_file)  else pd.DataFrame()
    df_head  = smart_read_csv(head_file)
    df_model = smart_read_csv(model_file) if _file_accessible(model_file) else pd.DataFrame()

    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    df_raw.columns = [c.lower().strip() for c in df_raw.columns]
    for c in ["bus_no","depot_id","svc_no","driver_id","alarm_type"]:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].astype(str).str.strip().str.upper()
    if "driver_id" in df_raw.columns:
        df_raw["driver_id"] = df_raw["driver_id"].str.replace(r"\.0$", "", regex=True)

    # Bus model merge
    if not df_model.empty:
        df_model.columns = [c.lower().strip() for c in df_model.columns]
        id_col = next((c for c in df_model.columns if c in ["bus_no","bus_number","vehicle_no"]), None)
        if id_col:
            df_model = df_model.rename(columns={id_col: "bus_no"})
            df_model["bus_no"] = df_model["bus_no"].astype(str).str.strip().str.upper()
            if "model" in df_model.columns:
                df_raw = df_raw.merge(df_model[["bus_no","model"]], on="bus_no", how="left")
                df_raw["model"] = df_raw["model"].fillna("Unknown")
    if "model" not in df_raw.columns:
        df_raw["model"] = "Unknown"

    # Exclusions
    if not df_excl.empty and "bus_no" in df_excl.columns:
        excl = set(df_excl["bus_no"].astype(str).str.strip().str.upper())
        df_raw = df_raw[~df_raw["bus_no"].isin(excl)]

    # Dates & durations
    df_raw["alarm_date"]        = pd.to_datetime(df_raw.get("alarm_calendar_date",""),    dayfirst=True, errors="coerce")
    df_raw["trip_departure_dt"] = pd.to_datetime(df_raw.get("trip_departure_datetime",""), dayfirst=True, errors="coerce")
    df_raw["trip_arrival_dt"]   = pd.to_datetime(df_raw.get("trip_arrival_datetime",""),   dayfirst=True, errors="coerce")
    df_raw["trip_duration_hr"]  = (
        (df_raw["trip_arrival_dt"] - df_raw["trip_departure_dt"]).dt.total_seconds() / 3600
    ).fillna(0.0)

    iso = df_raw["alarm_date"].dt.isocalendar()
    df_raw["alarm_year"]  = iso.year.astype("Int64")
    df_raw["alarm_week"]  = iso.week.astype("Int64")
    df_raw["day_of_week"] = df_raw["alarm_date"].dt.dayofweek

    if "trip_id" not in df_raw.columns:
        df_raw["trip_id"] = range(len(df_raw))
    df_raw["trip_id_norm"] = df_raw["trip_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    comp_col = next((c for c in ["is_week_completed","is_week"] if c in df_raw.columns), None)
    df_raw["is_week_completed_flag"] = (
        df_raw[comp_col].astype(str).str.strip().str.upper().isin(["Y","YES","TRUE","1"])
        if comp_col else False
    )
    df_raw["alarm_up"] = df_raw["alarm_type"].astype(str).str.strip().str.upper()

    if not df_head.empty and "depot_id" in df_head.columns:
        df_head["depot_id"] = df_head["depot_id"].astype(str).str.strip().str.upper()

    weekly_all = (
        df_raw[df_raw["alarm_date"].notna()]
        .groupby(["alarm_up","depot_id","alarm_year","alarm_week"], as_index=False)["trip_id_norm"]
        .count().rename(columns={"trip_id_norm":"alarm_sum"})
    )
    diagnostics = {
        "rows": len(df_raw),
        "depots": df_raw["depot_id"].nunique() if "depot_id" in df_raw.columns else 0,
        "date_range": (f"{df_raw['alarm_date'].min().date()} → {df_raw['alarm_date'].max().date()}"
                       if df_raw["alarm_date"].notna().any() else "N/A"),
        "model_matched": int((df_raw["model"] != "Unknown").sum()),
    }
    return df_raw, df_head, weekly_all, diagnostics

# ─────────────────────────────────────────────
# SLICING
# ─────────────────────────────────────────────
@st.cache_data
def slice_by_filters(df_raw, weekly_all, depots_tuple, alarm_choice, only_completed, exclude_null_driver):
    depots_set = set(depots_tuple)
    mask = (
        (df_raw["alarm_up"] == alarm_choice) &
        (df_raw["depot_id"].isin(depots_set)) &
        (df_raw["alarm_date"].notna())
    )
    df_alarm = df_raw.loc[mask].copy()
    if only_completed:
        df_alarm = df_alarm[df_alarm["is_week_completed_flag"]]
    if exclude_null_driver:
        df_alarm = df_alarm[
            df_alarm["driver_id"].notna() &
            (df_alarm["driver_id"].astype(str).str.strip() != "") &
            (df_alarm["driver_id"].astype(str) != "0")
        ]
    week_map = pd.DataFrame()
    if not df_alarm.empty:
        week_map = (df_alarm[["alarm_year","alarm_week"]].dropna().drop_duplicates()
                    .sort_values(["alarm_year","alarm_week"]))
        week_map["label"] = week_map.apply(
            lambda r: f"W{int(r['alarm_week'])} · {int(r['alarm_year'])}", axis=1)
    wk_mask = (weekly_all["alarm_up"] == alarm_choice) & (weekly_all["depot_id"].isin(depots_set))
    weekly = (weekly_all.loc[wk_mask, ["alarm_year","alarm_week","alarm_sum"]]
              .groupby(["alarm_year","alarm_week"], as_index=False)["alarm_sum"].sum())
    return df_alarm, week_map, weekly

@st.cache_data
def per_week_kpis(weekly, headcounts, depots_tuple):
    head_sel = headcounts[headcounts["depot_id"].isin(depots_tuple)] if not headcounts.empty else pd.DataFrame()
    total_hc = float(head_sel["headcount"].sum()) if not head_sel.empty else 1.0
    if total_hc <= 0: total_hc = 1.0
    wkly = weekly.copy()
    wkly["per_bc"] = wkly["alarm_sum"] / total_hc
    wkly["start_of_week"] = pd.to_datetime(
        wkly["alarm_year"].astype(int).astype(str) + wkly["alarm_week"].astype(int).astype(str) + "1",
        format="%G%V%w", errors="coerce")
    return wkly.sort_values("start_of_week"), total_hc

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def calc_metrics(ev_df, trips_df, category):
    if ev_df.empty: return pd.DataFrame()
    # Ensure category column is string (avoids numeric driver_id display)
    ev_df = ev_df.copy()
    if category in ev_df.columns:
        ev_df[category] = ev_df[category].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if not trips_df.empty and category in trips_df.columns:
        trips_df = trips_df.copy()
        trips_df[category] = trips_df[category].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    counts  = ev_df.groupby(category, sort=False).size().rename("Alarm Count")
    trips_u = (ev_df[["trip_id_norm", category]].drop_duplicates()
               .groupby(category, sort=False)["trip_id_norm"].nunique().rename("Alarm Trips"))
    if not trips_df.empty and category in trips_df.columns:
        dur = (trips_df.drop_duplicates("trip_id_norm")
               .groupby(category, sort=False)["trip_duration_hr"].sum()
               .rename("Total Duration (hr)"))
    else:
        dur = pd.Series(0.0, index=counts.index, name="Total Duration (hr)")
    df = pd.concat([counts, trips_u, dur], axis=1).fillna(0)
    df["Alarms per Trip"] = df["Alarm Count"] / df["Alarm Trips"].replace(0, np.nan)
    df["Alarms per Hour"] = df["Alarm Count"] / df["Total Duration (hr)"].where(df["Total Duration (hr)"] > 0)
    return df.fillna(0.0)

# ─────────────────────────────────────────────
# SMART FORECAST
# ─────────────────────────────────────────────
def calculate_smart_forecast(df_filtered, weekly_sum, current_week, total_hc):
    if df_filtered.empty or weekly_sum.empty:
        return 0.0, pd.DataFrame(), {}
    curr_data = df_filtered[df_filtered["alarm_week"] == current_week]
    if not curr_data.empty and "day_of_week" in curr_data.columns:
        max_day_idx = int(curr_data["day_of_week"].max())
        day_name    = curr_data["alarm_date"].max().strftime('%A') if curr_data["alarm_date"].notna().any() else "Friday"
    else:
        max_day_idx = 4; day_name = "Friday"
    past_weeks = weekly_sum[weekly_sum["alarm_week"] < current_week]["alarm_week"].unique()[-12:]
    ratios = []
    for w in past_weeks:
        wd = df_filtered[df_filtered["alarm_week"] == w]
        if len(wd) > 5:
            ratios.append(len(wd[wd["day_of_week"] <= max_day_idx]) / len(wd))
    comp_rate = float(np.mean(ratios)) if ratios else (max_day_idx + 1) / 7.0
    comp_rate = max(0.05, comp_rate)
    momentum = 1.0
    if len(weekly_sum) >= 5:
        rates = weekly_sum[weekly_sum["alarm_week"] < current_week].tail(4)["per_bc"].values
        if len(rates) > 1:
            s, _ = np.polyfit(np.arange(len(rates)), rates, 1)
            if s > 0: momentum = 1 + (s * 0.5)
    display_weeks = weekly_sum["alarm_week"].unique()[-5:]
    comp_rows = []
    for w in display_weeks:
        is_curr  = (w == current_week)
        wk_slice = df_filtered[df_filtered["alarm_week"] == w]
        hc       = max(1, total_hc)
        if is_curr:
            count_so_far = (len(wk_slice[wk_slice["day_of_week"] <= max_day_idx])
                            if "day_of_week" in wk_slice.columns else len(wk_slice))
            raw    = count_so_far / comp_rate
            adj    = raw * momentum
            avg_4  = weekly_sum[weekly_sum["alarm_week"] < w].tail(4)["per_bc"].mean()
            if pd.isna(avg_4): avg_4 = count_so_far / hc
            w_prof = min(1.0, comp_rate + 0.15)
            final  = ((adj / hc) * w_prof) + (avg_4 * (1 - w_prof))
            comp_rows.append({"week_label": f"W{int(w)} (Fcst)", "status": "In Progress", "display_rate": float(final)})
        else:
            comp_rows.append({"week_label": f"W{int(w)}", "status": "Completed", "display_rate": float(len(wk_slice) / hc)})
    proj_val = comp_rows[-1]["display_rate"] if comp_rows else 0.0
    return proj_val, pd.DataFrame(comp_rows), {"day": day_name, "completion_rate": comp_rate}

# ─────────────────────────────────────────────
# CHARTS  — xaxis/yaxis passed separately to avoid keyword conflict
# ─────────────────────────────────────────────
def _apply_layout(fig, title_text, title_color="#0f4c8a", extra=None):
    """Helper: apply PLOTLY_THEME + axis style without keyword conflicts."""
    layout = dict(
        title=dict(text=title_text, font=dict(size=13, color=title_color)),
        xaxis=dict(**AXIS_STYLE),
        yaxis=dict(**AXIS_STYLE),
        **PLOTLY_THEME,
    )
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)
    return fig

def trend_chart_12wk(df_agg, y_col, alarm_name):
    """12-week trend line chart — matches original screenshot style with date+week labels."""
    fig = go.Figure()
    if df_agg.empty: return fig
    max_y = max(6.5, float(df_agg[y_col].max()) * 1.15)

    # Build x-axis labels: "08 Dec\nW50" style to match original screenshot
    if "start_of_week" in df_agg.columns:
        x_labels = [
            f"{row['start_of_week'].strftime('%d %b')}<br>W{int(row['alarm_week'])}"
            for _, row in df_agg.iterrows()
        ]
    else:
        x_labels = df_agg.get("week_label", df_agg.get("alarm_week", range(len(df_agg)))).tolist()

    # Coloured threshold bands
    fig.add_hrect(y0=0,   y1=3,     fillcolor="rgba(34,197,94,0.10)",  line_width=0, layer="below")
    fig.add_hrect(y0=3,   y1=5,     fillcolor="rgba(245,158,11,0.09)", line_width=0, layer="below")
    fig.add_hrect(y0=5,   y1=max_y, fillcolor="rgba(239,68,68,0.10)",  line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=x_labels, y=df_agg[y_col].round(2),
        mode="lines+markers+text",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=7, color="#2563eb", line=dict(color="#93c5fd", width=1.5)),
        text=df_agg[y_col].round(2),
        textposition="top center",
        textfont=dict(color="#1e293b", size=11, family="IBM Plex Mono"),
        name=alarm_name,
    ))

    # Threshold lines
    fig.add_hline(y=3.0, line_dash="dot", line_color="rgba(34,197,94,0.8)",  line_width=1.5,
                  annotation_text="Normal 3.0", annotation_position="right",
                  annotation_font_color="rgba(34,197,94,0.9)", annotation_font_size=10)
    fig.add_hline(y=5.0, line_dash="dot", line_color="rgba(239,68,68,0.8)", line_width=1.5,
                  annotation_text="Critical 5.0", annotation_position="right",
                  annotation_font_color="rgba(239,68,68,0.9)", annotation_font_size=10)

    _apply_layout(fig, f"12-Week Trend: {alarm_name} per BC",
                  extra={
                      "yaxis": dict(title="Alarms per BC", range=[0, max_y], **AXIS_STYLE),
                      "xaxis": dict(title="Week", **AXIS_STYLE, tickangle=0),
                      "showlegend": False,
                      "height": 400,
                  })
    return fig

def forecast_bar_chart(comp_df):
    fig = go.Figure()
    if comp_df.empty: return fig
    colors = [
        ("#2563eb" if r["display_rate"] <= 5.0 else "#dc2626")
        if r["status"] == "In Progress" else "#cbd5e1"
        for _, r in comp_df.iterrows()
    ]
    fig.add_trace(go.Bar(
        x=comp_df["week_label"], y=comp_df["display_rate"],
        marker_color=colors, marker_line=dict(color="#e2e8f0", width=1),
        text=comp_df["display_rate"].round(2), textposition="outside",
        textfont=dict(color="#475569", size=11),
    ))
    fig.add_hline(y=3.0, line_dash="dot", line_color="rgba(34,197,94,0.7)", line_width=1.5)
    fig.add_hline(y=5.0, line_dash="dot", line_color="rgba(239,68,68,0.7)", line_width=1.5)
    _apply_layout(fig, "Risk Forecast (5-Week View)",
                  extra={"yaxis": dict(title="Rate per BC", **AXIS_STYLE),
                         "xaxis": dict(**AXIS_STYLE)})
    return fig

def depot_bar_chart(df_alarm, alarm_choice):
    if df_alarm.empty or "depot_id" not in df_alarm.columns: return go.Figure()
    counts = df_alarm["depot_id"].value_counts().reset_index()
    counts.columns = ["Depot","Count"]
    fig = px.bar(counts, x="Count", y="Depot", orientation="h",
                 color="Count", color_continuous_scale=["#bfdbfe","#2563eb"],
                 title=f"Depot Distribution — {alarm_choice}")
    fig.update_layout(xaxis=dict(**AXIS_STYLE), yaxis=dict(**AXIS_STYLE), **PLOTLY_THEME)
    return fig

def model_pie_chart(df_alarm):
    if df_alarm.empty or "model" not in df_alarm.columns: return go.Figure()
    counts = df_alarm["model"].value_counts().head(8).reset_index()
    counts.columns = ["Model","Count"]
    fig = px.pie(counts, values="Count", names="Model", hole=0.55,
                 color_discrete_sequence=px.colors.sequential.Blues,
                 title="Alarm Distribution by Bus Model")
    fig.update_layout(**PLOTLY_THEME)
    return fig

def driver_trend_bar(df_alarm, weeks_list):
    """Bar chart of top drivers over last 4 weeks — for 4-Week Pattern tab."""
    if df_alarm.empty: return go.Figure()
    d4 = df_alarm[df_alarm["alarm_week"].isin(weeks_list)]
    if d4.empty: return go.Figure()
    top_drv = d4.groupby("driver_id").size().sort_values(ascending=False).head(15).index
    d4_top  = d4[d4["driver_id"].isin(top_drv)]
    pivot   = d4_top.groupby(["driver_id","alarm_week"]).size().reset_index(name="count")
    fig = px.bar(pivot, x="driver_id", y="count", color=pivot["alarm_week"].astype(str),
                 barmode="group", title="Top 15 Drivers — 4-Week Alarm Count",
                 labels={"driver_id":"Driver","count":"Alarms","color":"Week"})
    fig.update_layout(xaxis=dict(tickangle=-45, **AXIS_STYLE),
                      yaxis=dict(**AXIS_STYLE), **PLOTLY_THEME,
                      legend=dict(title="Week"))
    return fig

# ─────────────────────────────────────────────
# AGENT TOOLS
# ─────────────────────────────────────────────
def build_agent_tools(df_alarm, weekly_sum, total_hc, alarm_choice, df_raw, depots_tuple):
    """
    Expert-grade analyst tools — covers driver/bus/service/depot/model dimensions,
    cross-entity correlation, anomaly detection, and cohort benchmarking.
    """

    # ── Fleet baseline stats (computed once, reused across tools) ────────
    _fleet_trips  = df_alarm["trip_id_norm"].nunique() if not df_alarm.empty else 1
    _fleet_alarms = len(df_alarm)
    _fleet_rate   = round(_fleet_alarms / _fleet_trips, 3) if _fleet_trips > 0 else 0
    _has_model    = "model" in df_alarm.columns
    _has_depot    = "depot_id" in df_alarm.columns
    _has_duration = "trip_duration_hr" in df_alarm.columns

    def _safe_rate(cnt, trips): return round(cnt/trips, 3) if trips > 0 else 0
    def _pct(part, total):     return round(100*part/total, 1) if total > 0 else 0
    def _status(rate):         return "🔴 CRITICAL" if rate > 5 else ("🟡 ELEVATED" if rate > 3 else "🟢 NORMAL")
    def _vs_fleet(rate):
        diff = rate - _fleet_rate
        return f"{diff:+.3f} vs fleet avg ({_fleet_rate:.3f})"

    def _week_breakdown(d, col="alarm_week"):
        rows = d.groupby(col).size().reset_index(name="n")
        return {f"W{int(r[col])}": int(r["n"]) for _, r in rows.iterrows()}

    # ────────────────────────────────────────────────────────────────────
    # TOOL 1 — Deep driver profile
    # ────────────────────────────────────────────────────────────────────
    def tool_driver_deep_profile(driver_id: str) -> str:
        """Full driver profile: rate, buses driven, services, depots, weekly trend, peer comparison."""
        did  = driver_id.strip().upper()
        d    = df_alarm[df_alarm["driver_id"].astype(str).str.upper() == did]
        if d.empty: return f"No {alarm_choice} data for driver {did}."

        total = len(d)
        trips = d["trip_id_norm"].nunique()
        rate  = _safe_rate(total, trips)
        hrs   = d.drop_duplicates("trip_id_norm")["trip_duration_hr"].sum() if _has_duration else 0
        depot = d["depot_id"].mode()[0] if _has_depot and not d.empty else "N/A"

        # Buses this driver operates — detect if problem follows driver or bus
        bus_stats = (d.groupby("bus_no").agg(
            alarms=("bus_no","count"),
            trips=("trip_id_norm","nunique")).reset_index())
        bus_stats["rate"] = bus_stats.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        bus_stats = bus_stats.sort_values("rate", ascending=False)

        # Services this driver runs
        svc_counts = d["svc_no"].value_counts().head(5).to_dict()

        # Weekly consistency — flags if issue is chronic vs acute
        wk_breakdown = _week_breakdown(d)
        wk_present   = len(wk_breakdown)
        all_weeks    = weekly_sum["alarm_week"].nunique() if not weekly_sum.empty else 1
        chronic      = wk_present >= max(2, all_weeks * 0.6)

        # Peer percentile within depot
        depot_drivers = df_alarm[df_alarm["depot_id"] == depot] if _has_depot else df_alarm
        peer_rates = (depot_drivers.groupby("driver_id")
                      .agg(a=("driver_id","count"), t=("trip_id_norm","nunique"))
                      .assign(rate=lambda x: x.a/x.t.clip(lower=1))["rate"])
        pct_rank = _pct((peer_rates < rate).sum(), len(peer_rates))

        # Model breakdown (if bus uses same model always, or varies)
        model_str = ""
        if _has_model:
            models = d["model"].value_counts()
            model_str = f"\nBus models operated: {models.head(3).to_dict()}"

        # Root cause signal
        n_buses = len(bus_stats)
        high_rate_buses = bus_stats[bus_stats["rate"] > _fleet_rate]
        if n_buses >= 3 and len(high_rate_buses) >= 2:
            signal = "⚠️ DRIVER-SPECIFIC — high rate across multiple buses → coaching/retraining indicated"
        elif n_buses == 1:
            signal = "🔍 POSSIBLE VEHICLE INTERACTION — single bus, check sensor calibration first"
        else:
            bus_top = bus_stats.iloc[0]
            signal = f"🔍 INVESTIGATE: highest rate on bus {bus_top['bus_no']} ({bus_top['rate']:.2f}/trip) — may be vehicle + driver combo"

        lines = [
            f"## Driver {did} — {alarm_choice} Deep Profile",
            f"**Depot:** {depot} | **Total alarms:** {total:,} | **Trips:** {trips:,} | **Rate:** {rate:.3f}/trip",
            f"**Fleet avg:** {_fleet_rate:.3f}/trip | **Relative:** {_vs_fleet(rate)}",
            f"**Status:** {_status(rate)} | **Peer rank in depot:** top {100-pct_rank:.0f}% (worse than {pct_rank:.0f}% of peers)",
            f"**Pattern:** {'CHRONIC (present ' + str(wk_present) + '/' + str(all_weeks) + ' weeks)' if chronic else 'ACUTE (recent spike)'}",
            "",
            "**Buses driven (rate per bus):**",
        ]
        for _, row in bus_stats.head(6).iterrows():
            flag = " ← HIGH" if row["rate"] > _fleet_rate * 1.5 else ""
            lines.append(f"  • {row['bus_no']}: {row['alarms']} alarms / {row['trips']} trips = **{row['rate']:.3f}/trip**{flag}")
        lines += [
            model_str,
            f"\n**Services operated:** {svc_counts}",
            f"\n**Weekly alarm counts:** {wk_breakdown}",
            f"\n**Root Cause Signal:** {signal}",
            f"\n**Recommended action:** {'Schedule driver coaching session with supervisor' if 'DRIVER-SPECIFIC' in signal else 'Inspect bus sensor + review driver+bus assignment history'}",
        ]
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 2 — Deep bus profile (with model context)
    # ────────────────────────────────────────────────────────────────────
    def tool_bus_deep_profile(bus_no: str) -> str:
        """Full bus profile: model, depot, drivers, services, weekly rate, model-cohort comparison."""
        bid = bus_no.strip().upper()
        d   = df_alarm[df_alarm["bus_no"].astype(str).str.upper() == bid]
        if d.empty: return f"No {alarm_choice} data for bus {bid}."

        total = len(d)
        trips = d["trip_id_norm"].nunique()
        rate  = _safe_rate(total, trips)
        model = d["model"].mode()[0] if _has_model and not d.empty else "Unknown"
        depot = d["depot_id"].mode()[0] if _has_depot and not d.empty else "N/A"
        hrs   = d.drop_duplicates("trip_id_norm")["trip_duration_hr"].sum() if _has_duration else 0

        # Model cohort comparison — is this bus worse than peers of same model?
        cohort_rate, cohort_size = _fleet_rate, 0
        if _has_model:
            cohort = df_alarm[df_alarm["model"] == model]
            cohort_trips  = cohort["trip_id_norm"].nunique()
            cohort_alarms = len(cohort)
            cohort_rate   = _safe_rate(cohort_alarms, cohort_trips)
            cohort_size   = cohort["bus_no"].nunique()

        # Drivers who operate this bus — detect if issue follows the bus
        drv_stats = (d.groupby("driver_id").agg(
            alarms=("driver_id","count"),
            trips=("trip_id_norm","nunique")).reset_index())
        drv_stats["rate"] = drv_stats.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        drv_stats = drv_stats.sort_values("rate", ascending=False)

        n_drivers     = len(drv_stats)
        high_rate_drv = drv_stats[drv_stats["rate"] > _fleet_rate]

        # Services
        svc_top = d["svc_no"].value_counts().head(5).to_dict()
        wk_breakdown = _week_breakdown(d)

        # Root cause signal
        if n_drivers >= 3 and len(high_rate_drv) >= 2:
            signal = "⚠️ VEHICLE-SPECIFIC — high rate across multiple drivers → sensor/mechanical inspection"
            action = "Raise SIS ticket for sensor calibration check; withdraw from service if rate >2× fleet avg"
        elif n_drivers == 1:
            drv_top = drv_stats.iloc[0]["driver_id"]
            signal  = f"🔍 SINGLE DRIVER — bus only operated by {drv_top}; may be driver behaviour not vehicle fault"
            action  = f"Review driver {drv_top} performance before raising vehicle inspection"
        else:
            signal = "🔍 MIXED SIGNAL — some drivers high, some normal → check driver+route assignment"
            action = "Compare driver rates on this bus; if top driver rate >3× bottom, issue is driver-specific"

        # vs model cohort
        vs_cohort = rate - cohort_rate
        cohort_str = (f"**Model cohort ({model}, {cohort_size} buses):** avg rate {cohort_rate:.3f}/trip | "
                      f"This bus is {vs_cohort:+.3f} {'above' if vs_cohort>0 else 'below'} model average")

        lines = [
            f"## Bus {bid} — {alarm_choice} Deep Profile",
            f"**Model:** {model} | **Depot:** {depot} | **Total alarms:** {total:,} | **Trips:** {trips:,}",
            f"**Rate:** {rate:.3f}/trip | {_vs_fleet(rate)} | **Status:** {_status(rate)}",
            cohort_str,
            "",
            "**Drivers operating this bus (rate per driver):**",
        ]
        for _, row in drv_stats.head(6).iterrows():
            flag = " ← HIGH" if row["rate"] > cohort_rate * 1.5 else ""
            lines.append(f"  • Driver {row['driver_id']}: {row['alarms']} alarms / {row['trips']} trips = **{row['rate']:.3f}/trip**{flag}")
        lines += [
            f"\n**Services:** {svc_top}",
            f"\n**Weekly counts:** {wk_breakdown}",
            f"\n**Root Cause Signal:** {signal}",
            f"\n**Recommended action:** {action}",
        ]
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 3 — Service/route deep profile
    # ────────────────────────────────────────────────────────────────────
    def tool_service_deep_profile(svc_no: str) -> str:
        """Service route analysis: alarm hotspots, driver breakdown, depot, time pattern."""
        sid = svc_no.strip().upper()
        d   = df_alarm[df_alarm["svc_no"].astype(str).str.upper() == sid]
        if d.empty: return f"No {alarm_choice} data for service {sid}."

        total = len(d)
        trips = d["trip_id_norm"].nunique()
        rate  = _safe_rate(total, trips)
        depots_on_svc = d["depot_id"].value_counts().to_dict() if _has_depot else {}

        drv_stats = (d.groupby("driver_id").agg(
            alarms=("driver_id","count"),
            trips=("trip_id_norm","nunique")).reset_index())
        drv_stats["rate"] = drv_stats.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        drv_stats = drv_stats.sort_values("rate", ascending=False)

        bus_top = d["bus_no"].value_counts().head(5).to_dict()
        wk_breakdown = _week_breakdown(d)

        # Classification: route vs driver issue
        n_drivers    = len(drv_stats)
        high_rate_d  = drv_stats[drv_stats["rate"] > _fleet_rate * 1.5]
        concentration = _pct(drv_stats.head(3)["alarms"].sum(), total)

        if concentration > 60 and len(high_rate_d) <= 2:
            signal = f"⚠️ DRIVER-CONCENTRATED — top 3 drivers account for {concentration:.0f}% of alarms on this route"
            action = "Target top 2-3 drivers on this service for coaching; route itself may be fine"
        elif n_drivers >= 5 and len(high_rate_d) >= 3:
            signal = "⚠️ ROUTE-SPECIFIC — multiple drivers trigger alarms → road conditions, stop layout, or speed profile"
            action = "Review route speed limits, bus stop approach zones; consider route safety audit"
        else:
            signal = "🔍 MIXED — moderate concentration; investigate both driver technique and route profile"
            action = "Monitor top 2 drivers + request route observation with supervisor"

        model_str = ""
        if _has_model:
            model_on_svc = d["model"].value_counts().head(3).to_dict()
            model_str = f"\n**Bus models on this service:** {model_on_svc}"

        lines = [
            f"## Service {sid} — {alarm_choice} Profile",
            f"**Total alarms:** {total:,} | **Trips:** {trips:,} | **Rate:** {rate:.3f}/trip",
            f"**Fleet avg:** {_fleet_rate:.3f} | {_vs_fleet(rate)} | **Status:** {_status(rate)}",
            f"**Operating depots:** {depots_on_svc}",
            model_str,
            "",
            "**Drivers on this service:**",
        ]
        for _, row in drv_stats.head(8).iterrows():
            flag = " ← FLAG" if row["rate"] > _fleet_rate * 1.5 else ""
            lines.append(f"  • Driver {row['driver_id']}: {row['alarms']} alarms / {row['trips']} trips = **{row['rate']:.3f}/trip**{flag}")
        lines += [
            f"\n**Top buses on route:** {bus_top}",
            f"\n**Weekly counts:** {wk_breakdown}",
            f"\n**Driver alarm concentration (top 3):** {concentration:.0f}%",
            f"\n**Root Cause Signal:** {signal}",
            f"\n**Recommended action:** {action}",
        ]
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 4 — Depot deep profile
    # ────────────────────────────────────────────────────────────────────
    def tool_depot_compare(input_str: str = "") -> str:
        """Compare all depots: rate, trend direction, top offenders per depot, model mix."""
        if df_alarm.empty or not _has_depot: return "No depot data available."
        results = []
        for depot in sorted(df_alarm["depot_id"].unique()):
            d     = df_alarm[df_alarm["depot_id"] == depot]
            total = len(d)
            trips = d["trip_id_norm"].nunique()
            rate  = _safe_rate(total, trips)
            n_drv = d["driver_id"].nunique()

            # Trend: compare first half vs second half of weeks available
            wks = sorted(d["alarm_week"].unique())
            if len(wks) >= 4:
                mid   = len(wks) // 2
                r_old = _safe_rate(len(d[d["alarm_week"].isin(wks[:mid])]),
                                   d[d["alarm_week"].isin(wks[:mid])]["trip_id_norm"].nunique())
                r_new = _safe_rate(len(d[d["alarm_week"].isin(wks[mid:])]),
                                   d[d["alarm_week"].isin(wks[mid:])]["trip_id_norm"].nunique())
                trend = f"{'📈 WORSENING' if r_new > r_old else '📉 IMPROVING'} ({r_old:.2f}→{r_new:.2f})"
            else:
                trend = "insufficient weeks"

            top_drv = d["driver_id"].value_counts().head(3).to_dict()
            model_str = d["model"].value_counts().head(2).to_dict() if _has_model else {}

            results.append({
                "depot": depot, "rate": rate, "total": total,
                "trips": trips, "drivers": n_drv, "trend": trend,
                "top_drivers": top_drv, "models": model_str,
            })

        results.sort(key=lambda x: x["rate"], reverse=True)
        lines = [f"## Depot Comparison — {alarm_choice}", ""]
        for r in results:
            lines += [
                f"### {r['depot']} — {_status(r['rate'])} | Rate: **{r['rate']:.3f}/trip**",
                f"Alarms: {r['total']:,} | Trips: {r['trips']:,} | Drivers: {r['drivers']} | Trend: {r['trend']}",
                f"Top drivers: {r['top_drivers']}",
                f"Bus models: {r['models']}",
                "",
            ]
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 5 — Model / vehicle-type analysis
    # ────────────────────────────────────────────────────────────────────
    def tool_model_analysis(input_str: str = "") -> str:
        """Compare alarm rates across bus models — identifies if specific model has systematic issue."""
        if not _has_model or df_alarm.empty: return "No model data available."
        model_stats = (df_alarm.groupby("model").agg(
            alarms=("model","count"),
            trips=("trip_id_norm","nunique"),
            buses=("bus_no","nunique"),
            drivers=("driver_id","nunique")).reset_index())
        model_stats["rate"] = model_stats.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        model_stats["alarm_pct"] = model_stats["alarms"].apply(lambda a: _pct(a, _fleet_alarms))
        model_stats = model_stats.sort_values("rate", ascending=False)

        lines = [f"## Bus Model Analysis — {alarm_choice}", f"Fleet average: {_fleet_rate:.3f}/trip", ""]
        for _, row in model_stats.iterrows():
            vs = row["rate"] - _fleet_rate
            flag = " ⚠️ ABOVE FLEET AVG" if vs > 0.5 else (" ✅ BELOW FLEET AVG" if vs < -0.3 else "")
            lines.append(
                f"**{row['model']}**: {row['rate']:.3f}/trip | {row['alarms']:,} alarms ({row['alarm_pct']:.1f}% of fleet) | "
                f"{row['buses']} buses | {row['drivers']} drivers | vs fleet: {vs:+.3f}{flag}"
            )

        # Flag if any model is >1.5× fleet rate
        outliers = model_stats[model_stats["rate"] > _fleet_rate * 1.5]
        if not outliers.empty:
            lines.append(f"\n⚠️ **Outlier models (>1.5× fleet rate):** {', '.join(outliers['model'].tolist())}")
            lines.append("→ Recommend: Check sensor calibration + maintenance schedule for these models")
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 6 — Cross-entity pattern / nexus finder
    # ────────────────────────────────────────────────────────────────────
    def tool_find_patterns(input_str: str = "") -> str:
        """
        Finds cross-dimension patterns: driver+bus combos, driver+service combos,
        model+depot combos. Identifies compounding risk factors.
        """
        if df_alarm.empty: return "No data."
        lines = [f"## Cross-Entity Pattern Analysis — {alarm_choice}", ""]

        # 1. Driver+Bus combos — same driver on same bus = compounding
        db = (df_alarm.groupby(["driver_id","bus_no"]).agg(
            alarms=("driver_id","count"), trips=("trip_id_norm","nunique")).reset_index())
        db["rate"] = db.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        db_hot = db[db["rate"] > _fleet_rate * 2].sort_values("rate", ascending=False).head(8)
        if not db_hot.empty:
            lines.append("### 🔴 High-Risk Driver+Bus Combinations (rate > 2× fleet avg)")
            for _, r in db_hot.iterrows():
                model = ""
                if _has_model:
                    m = df_alarm[(df_alarm["driver_id"]==r["driver_id"]) & (df_alarm["bus_no"]==r["bus_no"])]["model"].mode()
                    model = f" [{m.iloc[0]}]" if not m.empty else ""
                lines.append(f"  • Driver **{r['driver_id']}** on Bus **{r['bus_no']}**{model}: "
                             f"{r['alarms']} alarms / {r['trips']} trips = **{r['rate']:.3f}/trip**")
        else:
            lines.append("### Driver+Bus Combinations\nNo extreme driver+bus combos detected.")

        # 2. Driver+Service combos
        ds = (df_alarm.groupby(["driver_id","svc_no"]).agg(
            alarms=("driver_id","count"), trips=("trip_id_norm","nunique")).reset_index())
        ds["rate"] = ds.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        ds_hot = ds[ds["rate"] > _fleet_rate * 2].sort_values("rate", ascending=False).head(6)
        if not ds_hot.empty:
            lines.append("\n### 🟡 High-Risk Driver+Service Combinations")
            for _, r in ds_hot.iterrows():
                depot = ""
                if _has_depot:
                    dep = df_alarm[df_alarm["driver_id"]==r["driver_id"]]["depot_id"].mode()
                    depot = f" [{dep.iloc[0]}]" if not dep.empty else ""
                lines.append(f"  • Driver **{r['driver_id']}**{depot} on Service **{r['svc_no']}**: "
                             f"{r['alarms']} alarms / {r['trips']} trips = **{r['rate']:.3f}/trip**")

        # 3. Model+Depot combos — systematic model issues at specific depots
        if _has_model and _has_depot:
            md = (df_alarm.groupby(["model","depot_id"]).agg(
                alarms=("model","count"), trips=("trip_id_norm","nunique")).reset_index())
            md["rate"] = md.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
            md_hot = md[md["rate"] > _fleet_rate * 1.8].sort_values("rate", ascending=False).head(5)
            if not md_hot.empty:
                lines.append("\n### 🟠 Model+Depot Hotspots (possible fleet-wide maintenance issue)")
                for _, r in md_hot.iterrows():
                    lines.append(f"  • Model **{r['model']}** at **{r['depot_id']}**: "
                                 f"{r['alarms']} alarms / {r['trips']} trips = **{r['rate']:.3f}/trip**")

        # 4. Persistent offenders across weeks
        weeks = sorted(df_alarm["alarm_week"].unique())
        if len(weeks) >= 3:
            threshold = max(2, len(weeks) * 0.6)
            persist_d = (df_alarm.groupby(["driver_id","alarm_week"]).size()
                         .reset_index(name="n").groupby("driver_id")["alarm_week"].nunique())
            chronic_drivers = persist_d[persist_d >= threshold].sort_values(ascending=False).head(8)
            if not chronic_drivers.empty:
                lines.append(f"\n### 🔴 Chronically Persistent Drivers (present in {threshold:.0f}+ of {len(weeks)} weeks)")
                for drv, wk_count in chronic_drivers.items():
                    drv_rate = _safe_rate(
                        len(df_alarm[df_alarm["driver_id"]==drv]),
                        df_alarm[df_alarm["driver_id"]==drv]["trip_id_norm"].nunique())
                    lines.append(f"  • Driver **{drv}**: {wk_count}/{len(weeks)} weeks | rate {drv_rate:.3f}/trip → PRIORITY FOR ACTION")

        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 7 — Anomaly detector
    # ────────────────────────────────────────────────────────────────────
    def tool_detect_anomalies(input_str: str = "") -> str:
        """
        Statistical anomaly detection: entities with rates > mean + 2SD,
        new entrants to top-10, sudden week-on-week spikes, low-trip high-rate outliers.
        """
        if df_alarm.empty: return "No data."
        lines = [f"## Anomaly Detection — {alarm_choice}", f"Fleet rate: {_fleet_rate:.3f}/trip", ""]

        # Per-driver rates
        dr = (df_alarm.groupby("driver_id").agg(
            alarms=("driver_id","count"),
            trips=("trip_id_norm","nunique")).reset_index())
        dr["rate"] = dr.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        mean_r = dr["rate"].mean(); std_r = dr["rate"].std()
        threshold_2sd = mean_r + 2*std_r

        # 1. Statistical outliers (>mean + 2SD)
        outliers_2sd = dr[dr["rate"] > threshold_2sd].sort_values("rate", ascending=False).head(8)
        lines.append(f"### 📊 Statistical Outliers (rate > mean+2SD = {threshold_2sd:.3f}/trip)")
        if outliers_2sd.empty:
            lines.append("No statistical outliers detected.")
        else:
            for _, r in outliers_2sd.iterrows():
                depot = df_alarm[df_alarm["driver_id"]==r["driver_id"]]["depot_id"].mode()
                dep   = depot.iloc[0] if _has_depot and not depot.empty else "?"
                lines.append(f"  • Driver **{r['driver_id']}** [{dep}]: {r['rate']:.3f}/trip ({r['alarms']} alarms, {r['trips']} trips) — **{r['rate']/mean_r:.1f}× avg**")

        # 2. Low trip, high rate (unreliable but dangerous signal)
        low_trip = dr[(dr["trips"] <= 5) & (dr["rate"] > _fleet_rate * 1.5)]
        if not low_trip.empty:
            lines.append(f"\n### ⚠️ Low-Trip High-Rate (≤5 trips, rate >1.5× fleet) — validate data quality")
            for _, r in low_trip.head(6).iterrows():
                lines.append(f"  • Driver **{r['driver_id']}**: {r['rate']:.3f}/trip but only {r['trips']} trips → monitor next 2 weeks before action")

        # 3. Week-on-week spike for current week vs prior week
        weeks = sorted(df_alarm["alarm_week"].unique())
        if len(weeks) >= 2:
            w_curr, w_prev = weeks[-1], weeks[-2]
            d_curr = df_alarm[df_alarm["alarm_week"] == w_curr]
            d_prev = df_alarm[df_alarm["alarm_week"] == w_prev]
            curr_r = _safe_rate(len(d_curr), d_curr["trip_id_norm"].nunique())
            prev_r = _safe_rate(len(d_prev), d_prev["trip_id_norm"].nunique())
            pct_chg = _pct(curr_r - prev_r, prev_r)
            lines.append(f"\n### 📈 Week-on-Week: W{int(w_curr)} vs W{int(w_prev)}")
            lines.append(f"Rate change: {prev_r:.3f} → {curr_r:.3f} ({pct_chg:+.1f}%) {'🔴 SPIKE' if pct_chg > 20 else ('🟢 DROP' if pct_chg < -10 else '🟡 STABLE')}")

            # New entrants to top-10 this week vs last week
            top_curr = set(d_curr["driver_id"].value_counts().head(10).index)
            top_prev = set(d_prev["driver_id"].value_counts().head(10).index)
            new_entrants = top_curr - top_prev
            if new_entrants:
                lines.append(f"🆕 **New entrants to top-10 this week** (not in top-10 last week): {new_entrants}")

        # 4. Bus anomalies
        bus_r = (df_alarm.groupby("bus_no").agg(
            alarms=("bus_no","count"), trips=("trip_id_norm","nunique")).reset_index())
        bus_r["rate"] = bus_r.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        bus_outliers = bus_r[bus_r["rate"] > threshold_2sd].sort_values("rate", ascending=False).head(5)
        if not bus_outliers.empty:
            lines.append(f"\n### 🚌 Bus Anomalies (rate > {threshold_2sd:.3f}/trip)")
            for _, r in bus_outliers.iterrows():
                model = df_alarm[df_alarm["bus_no"]==r["bus_no"]]["model"].mode()
                mod   = model.iloc[0] if _has_model and not model.empty else "?"
                lines.append(f"  • Bus **{r['bus_no']}** [{mod}]: {r['rate']:.3f}/trip ({r['alarms']} alarms) — **{r['rate']/mean_r:.1f}× avg**")

        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 8 — Weekly trend (enhanced with per-depot breakdown)
    # ────────────────────────────────────────────────────────────────────
    def tool_weekly_trend(n_weeks: str = "8") -> str:
        """Fleet weekly trend with per-depot rates, momentum, and threshold status."""
        try: n = int(n_weeks)
        except: n = 8
        tail = weekly_sum.tail(n)
        if tail.empty: return "No weekly trend data."

        rates = tail["per_bc"].tolist()
        # Linear momentum: compare last 3 vs first 3 in window
        momentum = "IMPROVING 📉" if rates[-1] < rates[0] else "WORSENING 📈"
        avg4 = sum(rates[-4:])/4 if len(rates) >= 4 else sum(rates)/len(rates)

        lines = [f"## Fleet Weekly Trend — {alarm_choice} (last {n} weeks)",
                 f"Fleet avg last 4 wks: **{avg4:.3f}/BC** | Momentum: **{momentum}**", ""]

        for _, r in tail.iterrows():
            wk_num = int(r["alarm_week"]); yr = int(r.get("alarm_year", 0))
            rate   = r["per_bc"]
            total  = int(r.get("total", 0))
            bar    = "█" * min(20, int(rate * 2))
            status = _status(rate)
            lines.append(f"W{wk_num}/{yr}: **{rate:.3f}** {status}  {bar}  ({total:,} events)")

        # Per-depot trend for most recent week
        if _has_depot and not df_alarm.empty:
            last_wk = int(tail.iloc[-1]["alarm_week"])
            d_last  = df_alarm[df_alarm["alarm_week"] == last_wk]
            lines.append(f"\n**Depot breakdown W{last_wk}:**")
            for depot in sorted(d_last["depot_id"].unique()):
                dd = d_last[d_last["depot_id"] == depot]
                r  = _safe_rate(len(dd), dd["trip_id_norm"].nunique())
                lines.append(f"  • {depot}: {r:.3f}/trip ({len(dd):,} alarms)")

        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 9 — Top offenders with full context
    # ────────────────────────────────────────────────────────────────────
    def tool_top_offenders(category: str = "driver") -> str:
        """Ranked top offenders with depot, model, rate vs fleet, trend direction."""
        cat_map = {"driver":"driver_id","bus":"bus_no","svc":"svc_no","service":"svc_no","model":"model"}
        col = cat_map.get(category.lower().strip(), "driver_id")
        if col not in df_alarm.columns: return f"Column '{col}' not in data."

        stats = (df_alarm.groupby(col).agg(
            alarms=(col,"count"), trips=("trip_id_norm","nunique")).reset_index())
        stats["rate"] = stats.apply(lambda r: _safe_rate(r.alarms, r.trips), axis=1)
        stats = stats.sort_values("rate", ascending=False).head(15)

        lines = [f"## Top 15 {category.title()}s — {alarm_choice}", f"Fleet avg: {_fleet_rate:.3f}/trip", ""]
        for i, (_, r) in enumerate(stats.iterrows(), 1):
            eid = r[col]
            d_e = df_alarm[df_alarm[col] == eid]
            depot  = d_e["depot_id"].mode().iloc[0] if _has_depot and not d_e.empty else "?"
            model  = d_e["model"].mode().iloc[0]    if _has_model and not d_e.empty else ""
            weeks  = d_e["alarm_week"].nunique()
            vs     = r["rate"] - _fleet_rate
            flag   = " 🔴" if r["rate"] > 5 else (" 🟡" if r["rate"] > 3 else " 🟢")
            model_str = f" [{model}]" if model else ""
            lines.append(
                f"{i:2}. **{eid}**{model_str} [{depot}]{flag} "
                f"{r['rate']:.3f}/trip | {r['alarms']} alarms | {r['trips']} trips | "
                f"{weeks} weeks | vs fleet: {vs:+.3f}"
            )
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 10 — Week summary (enhanced)
    # ────────────────────────────────────────────────────────────────────
    def tool_week_summary(week_str: str) -> str:
        """Detailed single-week breakdown: depot, model, driver, bus, service, and vs prior week."""
        wk = None; yr = None
        m = re.search(r"w?(\d{1,2})", week_str.lower())
        if m: wk = int(m.group(1))
        m2 = re.search(r"\b(20\d{2})\b", week_str)
        if m2: yr = int(m2.group(1))
        if wk is None: return "Specify week e.g. W9 or W9 2026"
        if yr is None and not weekly_sum.empty:
            yr = int(weekly_sum["alarm_year"].dropna().max())

        d = df_alarm[df_alarm["alarm_week"] == wk]
        if yr: d = d[d["alarm_year"] == yr]
        if d.empty: return f"No data for W{wk} {yr or ''}."

        total = len(d); trips = d["trip_id_norm"].nunique()
        rate  = _safe_rate(total, trips)

        # vs prior week
        d_prev = df_alarm[df_alarm["alarm_week"] == wk-1]
        if yr: d_prev = d_prev[d_prev["alarm_year"] == yr]
        prev_rate = _safe_rate(len(d_prev), d_prev["trip_id_norm"].nunique()) if not d_prev.empty else None

        lines = [
            f"## W{wk} {yr or ''} — {alarm_choice} Summary",
            f"**Total alarms:** {total:,} | **Trips:** {trips:,} | **Rate:** {rate:.3f}/BC | **Status:** {_status(rate)}",
        ]
        if prev_rate is not None:
            chg = rate - prev_rate
            lines.append(f"**vs W{wk-1}:** {prev_rate:.3f} → {rate:.3f} ({chg:+.3f}, {'▲' if chg > 0 else '▼'})")

        if _has_depot:
            depot_b = d["depot_id"].value_counts()
            lines.append("\n**Depot breakdown:**")
            for dep, cnt in depot_b.items():
                dep_trips = d[d["depot_id"]==dep]["trip_id_norm"].nunique()
                dep_rate  = _safe_rate(cnt, dep_trips)
                lines.append(f"  • {dep}: {cnt:,} alarms | {dep_rate:.3f}/trip | {_status(dep_rate)}")

        if _has_model:
            lines.append(f"\n**Top models:** {d['model'].value_counts().head(4).to_dict()}")

        lines.append(f"\n**Top 8 drivers:** {d['driver_id'].value_counts().head(8).to_dict()}")
        lines.append(f"\n**Top 5 buses:** {d['bus_no'].value_counts().head(5).to_dict()}")
        lines.append(f"\n**Top 5 services:** {d['svc_no'].value_counts().head(5).to_dict()}")
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────
    # TOOL 11 — Root cause drill (enhanced)
    # ────────────────────────────────────────────────────────────────────
    def tool_root_cause_drill(entity: str) -> str:
        """
        Deep root cause: driver / bus / service / depot.
        Uses multi-dimension evidence to classify DRIVER / VEHICLE / ROUTE / FLEET issue.
        """
        parts = entity.strip().split(None, 1)
        if len(parts) < 2:
            return "Specify: 'driver 30450', 'bus SMB123D', 'svc 97', or 'depot KRANJI'"
        etype, eid = parts[0].lower(), parts[1].upper()
        col_map = {"driver":"driver_id","bus":"bus_no","svc":"svc_no","service":"svc_no","depot":"depot_id"}
        col = col_map.get(etype)
        if not col: return f"Unknown type '{etype}'. Use: driver/bus/svc/depot"

        d = df_alarm[df_alarm[col].astype(str).str.upper() == eid]
        if d.empty: return f"No {alarm_choice} data for {etype} {eid}."

        total = len(d); trips = d["trip_id_norm"].nunique(); rate = _safe_rate(total, trips)
        lines = [f"## Root Cause Analysis — {etype.title()} {eid}", f"**Rate:** {rate:.3f}/trip | **Status:** {_status(rate)} | **vs fleet:** {_vs_fleet(rate)}", ""]

        if etype == "driver":
            # Evidence dimension 1: consistent across buses?
            bus_rates = (d.groupby("bus_no").agg(a=("bus_no","count"),t=("trip_id_norm","nunique")).reset_index())
            bus_rates["rate"] = bus_rates.apply(lambda r: _safe_rate(r.a, r.t), axis=1)
            n_buses_high = len(bus_rates[bus_rates["rate"] > _fleet_rate])
            # Evidence dimension 2: consistent across services?
            svc_rates = (d.groupby("svc_no").agg(a=("svc_no","count"),t=("trip_id_norm","nunique")).reset_index())
            svc_rates["rate"] = svc_rates.apply(lambda r: _safe_rate(r.a, r.t), axis=1)
            n_svc_high = len(svc_rates[svc_rates["rate"] > _fleet_rate])
            # Evidence dimension 3: chronic vs acute?
            wks = d["alarm_week"].nunique(); all_wks = df_alarm["alarm_week"].nunique()
            chronic = wks >= max(2, all_wks * 0.6)
            # Verdict
            if n_buses_high >= 2 and n_svc_high >= 2:
                verdict = "🔴 DRIVER-SPECIFIC — high rate across multiple buses AND services → behaviour issue"
                action  = "Priority coaching session; if rate >5 consider temporary service reallocation"
            elif n_buses_high == 1:
                top_bus = bus_rates.sort_values("rate",ascending=False).iloc[0]
                verdict = f"🟡 DRIVER+VEHICLE INTERACTION — mainly on bus {top_bus['bus_no']} ({top_bus['rate']:.2f}/trip)"
                action  = f"Inspect bus {top_bus['bus_no']} sensor + route-check with driver together"
            else:
                verdict = "🟠 ROUTE/TIME SPECIFIC — spikes on certain services only"
                action  = "Review route assignment; check if specific services have road hazard or speed profile issue"
            lines += [
                f"**Buses driven:** {bus_rates[['bus_no','rate']].sort_values('rate',ascending=False).head(4).to_dict('records')}",
                f"**Services operated:** {svc_rates[['svc_no','rate']].sort_values('rate',ascending=False).head(4).to_dict('records')}",
                f"**Pattern:** {'CHRONIC' if chronic else 'ACUTE'} ({wks}/{all_wks} weeks active)",
                f"**Verdict:** {verdict}",
                f"**Recommended action:** {action}",
            ]

        elif etype == "bus":
            drv_rates = (d.groupby("driver_id").agg(a=("driver_id","count"),t=("trip_id_norm","nunique")).reset_index())
            drv_rates["rate"] = drv_rates.apply(lambda r: _safe_rate(r.a, r.t), axis=1)
            n_drv_high = len(drv_rates[drv_rates["rate"] > _fleet_rate])
            model = d["model"].mode().iloc[0] if _has_model and not d.empty else "Unknown"
            # Model cohort check
            cohort_rate = _fleet_rate
            if _has_model:
                cohort = df_alarm[df_alarm["model"] == model]
                cohort_rate = _safe_rate(len(cohort), cohort["trip_id_norm"].nunique())
            vs_cohort = rate - cohort_rate

            if n_drv_high >= 3:
                verdict = f"🔴 VEHICLE-SPECIFIC [{model}] — high rate regardless of driver → mechanical/sensor inspection"
                action  = "Raise SIS maintenance ticket immediately; check brake sensors, accelerometer calibration"
            elif len(drv_rates) == 1:
                only_drv = drv_rates.iloc[0]["driver_id"]
                verdict  = f"🟡 SINGLE DRIVER — only driver {only_drv} operates this bus; confirm driver, not vehicle"
                action   = f"Profile driver {only_drv} first; don't raise vehicle inspection until driver ruled out"
            else:
                verdict = "🟠 MIXED — some drivers trigger more alarms than others on this bus"
                action  = "Review which driver+bus assignments to reassign; highest-rate driver needs coaching"

            lines += [
                f"**Model:** {model} | **Model cohort avg:** {cohort_rate:.3f}/trip | **vs cohort:** {vs_cohort:+.3f}",
                f"**Drivers on this bus (top 5):** {drv_rates[['driver_id','rate']].sort_values('rate',ascending=False).head(5).to_dict('records')}",
                f"**Verdict:** {verdict}",
                f"**Recommended action:** {action}",
            ]

        elif etype in ("svc","service"):
            drv_rates = (d.groupby("driver_id").agg(a=("driver_id","count"),t=("trip_id_norm","nunique")).reset_index())
            drv_rates["rate"] = drv_rates.apply(lambda r: _safe_rate(r.a, r.t), axis=1)
            top3_pct  = _pct(drv_rates.sort_values("rate",ascending=False).head(3)["a"].sum(), total)
            n_drivers = len(drv_rates)
            if top3_pct > 70 and n_drivers > 5:
                verdict = f"🔴 DRIVER-CONCENTRATED — top 3 drivers = {top3_pct:.0f}% of alarms on this route"
                action  = "Coach top 3 drivers; route itself unlikely to be the issue"
            elif n_drivers >= 5:
                verdict = f"🟠 ROUTE-WIDE — {n_drivers} drivers all showing elevated alarms → route/road profile"
                action  = "Request route safety audit; check stop approach speeds and junctions"
            else:
                verdict = "🟡 INSUFFICIENT SAMPLE — few drivers; monitor for 2 more weeks"
                action  = "Hold action; monitor next 2 weeks before root cause conclusion"
            lines += [
                f"**Drivers on service ({n_drivers} total, top 5):** {drv_rates[['driver_id','rate']].sort_values('rate',ascending=False).head(5).to_dict('records')}",
                f"**Top-3 driver alarm concentration:** {top3_pct:.0f}%",
                f"**Verdict:** {verdict}",
                f"**Recommended action:** {action}",
            ]

        elif etype == "depot":
            drv_rates = (d.groupby("driver_id").agg(a=("driver_id","count"),t=("trip_id_norm","nunique")).reset_index())
            drv_rates["rate"] = drv_rates.apply(lambda r: _safe_rate(r.a, r.t), axis=1)
            bus_rates = (d.groupby("bus_no").agg(a=("bus_no","count"),t=("trip_id_norm","nunique")).reset_index())
            bus_rates["rate"] = bus_rates.apply(lambda r: _safe_rate(r.a, r.t), axis=1)
            top_model = d["model"].value_counts().head(2).to_dict() if _has_model else {}
            lines += [
                f"**Drivers (top 5):** {drv_rates[['driver_id','rate']].sort_values('rate',ascending=False).head(5).to_dict('records')}",
                f"**Buses (top 5):** {bus_rates[['bus_no','rate']].sort_values('rate',ascending=False).head(5).to_dict('records')}",
                f"**Models:** {top_model}",
                f"**Recommended action:** Review top 5 drivers first; if depot-wide issue, check training programme",
            ]

        return "\n".join(str(x) for x in lines)

    # ────────────────────────────────────────────────────────────────────
    # Return all tools
    # ────────────────────────────────────────────────────────────────────
    return [
        Tool(name="driver_deep_profile",  func=tool_driver_deep_profile,
             description="Full driver profile: rate, buses driven, services, depot, peer rank, root cause signal. Input: driver ID e.g. '30450'"),
        Tool(name="bus_deep_profile",     func=tool_bus_deep_profile,
             description="Full bus profile: model, drivers, rate vs model cohort, root cause. Input: bus number e.g. 'SMB323D'"),
        Tool(name="service_deep_profile", func=tool_service_deep_profile,
             description="Service/route profile: driver concentration, depot, route vs driver issue. Input: service number e.g. '97'"),
        Tool(name="depot_compare",        func=tool_depot_compare,
             description="Compare all depots: rate, trend, top offenders, model mix. Input: empty string"),
        Tool(name="model_analysis",       func=tool_model_analysis,
             description="Compare alarm rates by bus model — flags systematic vehicle-type issues. Input: empty string"),
        Tool(name="find_patterns",        func=tool_find_patterns,
             description="Cross-entity pattern finder: driver+bus combos, driver+service combos, model+depot hotspots, chronic offenders. Input: empty string"),
        Tool(name="detect_anomalies",     func=tool_detect_anomalies,
             description="Statistical anomaly detection: 2SD outliers, low-trip high-rate, week-on-week spikes, new entrants to top-10. Input: empty string"),
        Tool(name="weekly_trend",         func=tool_weekly_trend,
             description="Fleet weekly trend with depot breakdown and momentum. Input: number of weeks e.g. '8'"),
        Tool(name="top_offenders",        func=tool_top_offenders,
             description="Top 15 offenders with depot, model, rate vs fleet. Input: 'driver', 'bus', 'svc', or 'model'"),
        Tool(name="week_summary",         func=tool_week_summary,
             description="Detailed week summary: depot, model, driver, bus, service, vs prior week. Input: 'W9' or 'W9 2026'"),
        Tool(name="root_cause_drill",     func=tool_root_cause_drill,
             description="Multi-dimension root cause: classifies DRIVER/VEHICLE/ROUTE/FLEET. Input: 'driver 30450', 'bus SMB123D', 'svc 97', 'depot KRANJI'"),
    ]

# ─────────────────────────────────────────────
# REACT AGENT PROMPT + BUILDER
# ─────────────────────────────────────────────
REACT_PROMPT_TEMPLATE = """You are a world-class Fleet Operations Analyst specialising in bus telematics safety.
Alarm types: HA=Harsh Acceleration | HB=Harsh Braking | HC=Harsh Cornering
Thresholds: >5.0/BC = CRITICAL | 3.0-5.0 = ELEVATED | <3.0 = NORMAL
Current context — Alarm: {alarm_choice} | Depots: {depots}

ANALYST MINDSET:
- Always gather evidence across MULTIPLE dimensions (driver + bus + service + depot + model)
- Classify every root cause: DRIVER-SPECIFIC / VEHICLE-SPECIFIC / ROUTE-SPECIFIC / FLEET-WIDE
- Compare rates to fleet average — never report raw counts alone
- Flag chronic offenders (present 3+ weeks) separately from acute spikes
- Always end with a concrete, prioritised action list

AVAILABLE TOOLS:
{tools}

Tool names you can use: {tool_names}

CRITICAL FORMAT RULES — follow exactly or the parser will fail:
1. After EVERY Thought: you MUST write EITHER "Action:" OR "Final Answer:" — nothing else.
2. Never write a Thought: that ends without one of those two lines immediately following.
3. Use "Final Answer:" as soon as you have enough information — do not keep looping.
4. Action must be one of the tool names listed above — never invent a tool name.
5. Action Input must be on a single line immediately after Action:.

FORMAT:
Question: {input}
Thought: What tool do I need first?
Action: top_offenders
Action Input: driver
Observation: [result]
Thought: I have the data I need. I can now answer.
Final Answer:
## [Title]
[Rich markdown answer with specific IDs, counts, rates, root cause, and action table]

| Priority | Action | Target | Evidence |
|----------|--------|--------|---------|
| High | ... | ... | ... |

{agent_scratchpad}"""

def build_react_agent(llm, tools):
    if llm is None or not LANGCHAIN_OK or AgentExecutor is None: return None
    try:
        prompt   = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
        agent    = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent, tools=tools, verbose=False,
            handle_parsing_errors="Format error — please write 'Action: <tool>' or 'Final Answer: <answer>' after your Thought.",
            max_iterations=4,
            max_execution_time=25,
            early_stopping_method="generate",
            return_intermediate_steps=True,
        )
        return executor
    except Exception as e:
        # Agent build failed — fall through to Direct AI (Path 2)
        return None


def smart_rule_analyst(q: str, alarm_choice: str, df_alarm: pd.DataFrame,
                        weekly_sum: pd.DataFrame = None, w1_week: int = None,
                        depots: list = None) -> tuple:
    """
    Smart rule-based analyst that handles natural language questions
    without requiring an LLM. Returns (answer_markdown, optional_dataframe).
    Handles: top offenders, depot compare, week summary, spike analysis,
             trend, specific driver/bus/service lookup, and more.
    """
    if df_alarm is None or df_alarm.empty:
        return "No data loaded.", pd.DataFrame()

    ql = q.lower().strip()
    df = df_alarm.copy()

    # ── helpers ──────────────────────────────────────
    def _fmt_tbl(d, cols=None):
        if d.empty: return ""
        d2 = d[cols] if cols else d
        return "\n" + d2.to_markdown(index=False)

    def _rate(grp):
        cnt  = len(grp)
        trps = grp["trip_id_norm"].nunique() if "trip_id_norm" in grp.columns else 1
        return cnt, trps, round(cnt / trps, 3) if trps > 0 else 0.0

    # ── parse week/year filter from query ────────────
    wk_match   = re.search(r"w(\d{1,2})\s*[,\-]?\s*(\d{4})?", ql)
    yr_match   = re.search(r"20(\d{2})", ql)
    filter_wk  = int(wk_match.group(1)) if wk_match else None
    filter_yr  = int(wk_match.group(2)) if (wk_match and wk_match.group(2)) else (
                 int(yr_match.group(0)) if yr_match else None)
    n_match    = re.search(r"\b(\d+)\b", ql)
    top_n      = int(n_match.group(1)) if n_match and int(n_match.group(1)) <= 50 else 10

    # Apply week/year filter if requested
    df_q = df.copy()
    if filter_wk:
        df_q = df_q[df_q["alarm_week"] == filter_wk]
        if filter_yr and "alarm_year" in df_q.columns:
            df_q = df_q[df_q["alarm_year"] == filter_yr]
    week_label = f"W{filter_wk}" + (f" {filter_yr}" if filter_yr else "") if filter_wk else "current selection"

    # ── 1. SPECIFIC ENTITY LOOKUP ────────────────────
    for pat, col, label in [
        (r"(?:driver|bc|captain)\s*#?\s*([a-z0-9._-]+)", "driver_id", "Driver"),
        (r"(?:bus|vehicle)\s*#?\s*([a-z0-9._-]+)",        "bus_no",    "Bus"),
        (r"(?:svc|service)\s*#?\s*([a-z0-9._-]+)",        "svc_no",    "Service"),
    ]:
        m = re.search(pat, ql)
        if m:
            eid = m.group(1).upper()
            d   = df[df[col].astype(str).str.upper() == eid]
            if d.empty:
                return f"No {alarm_choice} data found for `{eid}`.", pd.DataFrame()
            cnt, trps, rate = _rate(d)
            wk_breakdown = (d.groupby(["alarm_week"]).size()
                             .reset_index(name="Alarms")
                             .rename(columns={"alarm_week": "Week"}))
            wk_breakdown["Week"] = "W" + wk_breakdown["Week"].astype(int).astype(str)
            depot = d["depot_id"].mode()[0] if "depot_id" in d.columns else "N/A"
            svc_top = d["svc_no"].value_counts().head(3).to_dict() if "svc_no" in d.columns else {}
            ans = (f"## {label} `{eid}` — {alarm_choice} Profile\n\n"
                   f"- **Total alarms:** {cnt:,}\n"
                   f"- **Unique trips:** {trps:,}\n"
                   f"- **Rate (alarms/trip):** **{rate:.3f}**\n"
                   f"- **Primary depot:** {depot}\n"
                   f"- **Top services:** {svc_top}\n\n"
                   f"**Weekly breakdown:**")
            return ans, wk_breakdown

    # ── 2. TOP OFFENDERS (drivers / buses / services / all) ──
    kw_top = any(w in ql for w in ["top", "offend", "worst", "high", "most alarm", "most incident"])
    kw_driver = any(w in ql for w in ["driver", "bc", "captain", "operator"])
    kw_bus    = any(w in ql for w in ["bus", "vehicle", "fleet"])
    kw_svc    = any(w in ql for w in ["service", "route", "svc"])

    if kw_top or "ranking" in ql or "list" in ql:
        results = []
        dset = df_q if not df_q.empty else df

        if kw_driver or not (kw_bus or kw_svc):
            dr = (dset.groupby("driver_id").size().reset_index(name="Alarms")
                  .sort_values("Alarms", ascending=False).head(top_n))
            if not dr.empty:
                dr["Depot"] = dr["driver_id"].map(
                    lambda x: dset[dset["driver_id"]==x]["depot_id"].mode()[0]
                    if "depot_id" in dset.columns and not dset[dset["driver_id"]==x].empty else "")
                results.append(("Drivers", dr.rename(columns={"driver_id": "Driver ID"})))

        if kw_bus:
            bs = (dset.groupby("bus_no").size().reset_index(name="Alarms")
                  .sort_values("Alarms", ascending=False).head(top_n))
            if not bs.empty:
                bs["Depot"] = bs["bus_no"].map(
                    lambda x: dset[dset["bus_no"]==x]["depot_id"].mode()[0]
                    if "depot_id" in dset.columns and not dset[dset["bus_no"]==x].empty else "")
                results.append(("Buses", bs.rename(columns={"bus_no": "Bus No"})))

        if kw_svc:
            sv = (dset.groupby("svc_no").size().reset_index(name="Alarms")
                  .sort_values("Alarms", ascending=False).head(top_n))
            if not sv.empty:
                results.append(("Services", sv.rename(columns={"svc_no": "Service"})))

        if results:
            name, tbl = results[0]
            fleet_total = len(dset)
            top_total   = tbl["Alarms"].sum()
            pct         = round(100 * top_total / fleet_total, 1) if fleet_total > 0 else 0
            ans = (f"## Top {top_n} {name} — {alarm_choice} ({week_label})\n\n"
                   f"**Fleet total:** {fleet_total:,} alarms | "
                   f"Top {top_n} account for **{top_total:,} alarms ({pct}%)**\n\n"
                   f"*Tip: Ask `driver <ID>` for a full profile of any driver below.*")
            return ans, tbl
        return f"No offender data available for {week_label}.", pd.DataFrame()

    # ── 3. DEPOT COMPARISON ──────────────────────────
    if any(w in ql for w in ["depot", "compare depot", "depot comparison", "by depot"]):
        dset = df_q if not df_q.empty else df
        if "depot_id" not in dset.columns:
            return "No depot data in dataset.", pd.DataFrame()
        dep_tbl = (dset.groupby("depot_id")
                   .agg(Alarms=("alarm_type","count"),
                        Drivers=("driver_id","nunique"),
                        Buses=("bus_no","nunique"))
                   .reset_index()
                   .rename(columns={"depot_id":"Depot"})
                   .sort_values("Alarms", ascending=False))
        best  = dep_tbl.iloc[-1]["Depot"]
        worst = dep_tbl.iloc[0]["Depot"]
        ans = (f"## Depot Comparison — {alarm_choice} ({week_label})\n\n"
               f"- **Highest alarms:** {worst} ({dep_tbl.iloc[0]['Alarms']:,} alarms)\n"
               f"- **Lowest alarms:** {best} ({dep_tbl.iloc[-1]['Alarms']:,} alarms)\n\n"
               f"**Full breakdown:**")
        return ans, dep_tbl

    # ── 4. WEEK SUMMARY ──────────────────────────────
    if any(w in ql for w in ["week summary", "summary", "weekly", "how was week", "w9", "w10"]) or filter_wk:
        target_wk = filter_wk or w1_week
        if target_wk is None:
            return "Please specify a week, e.g. `W9 summary` or `W10 2026`.", pd.DataFrame()
        dw = df[df["alarm_week"] == target_wk]
        if dw.empty:
            return f"No data for W{target_wk}.", pd.DataFrame()
        total = len(dw)
        rate_row = weekly_sum[weekly_sum["alarm_week"] == target_wk] if weekly_sum is not None else pd.DataFrame()
        rate = round(float(rate_row["per_bc"].values[0]), 3) if not rate_row.empty else "N/A"
        top_dr = dw["driver_id"].value_counts().head(5)
        top_bs = dw["bus_no"].value_counts().head(5)
        dep_split = dw["depot_id"].value_counts().to_dict() if "depot_id" in dw.columns else {}
        tbl = pd.DataFrame({
            "Metric": ["Total Alarms", "Rate (per BC)", "Active Drivers",
                       "Unique Buses", "Top Driver", "Top Bus", "Depot Spread"],
            "Value": [f"{total:,}", str(rate),
                      str(dw["driver_id"].nunique()),
                      str(dw["bus_no"].nunique()),
                      f"{top_dr.index[0]} ({top_dr.iloc[0]})" if len(top_dr) else "N/A",
                      f"{top_bs.index[0]} ({top_bs.iloc[0]})" if len(top_bs) else "N/A",
                      str(dep_split)]
        })
        ans = (f"## W{target_wk} Summary — {alarm_choice}\n\n"
               f"**{total:,} total alarms** | Rate: **{rate}**/BC\n"
               f"Depot breakdown: {dep_split}\n\n"
               f"**Top 5 Drivers:** {top_dr.to_dict()}\n"
               f"**Top 5 Buses:** {top_bs.to_dict()}\n\n"
               f"*Ask `driver <ID>` or `bus <No>` for a detailed profile.*")
        return ans, tbl

    # ── 5. SPIKE / TREND ANALYSIS ────────────────────
    if any(w in ql for w in ["spike", "jump", "surge", "increase", "why", "root cause", "cause", "trend"]):
        if weekly_sum is None or weekly_sum.empty:
            return "Weekly trend data not available.", pd.DataFrame()
        ws = weekly_sum.copy().sort_values("alarm_week")
        target_wk = filter_wk or (w1_week if w1_week else int(ws["alarm_week"].iloc[-1]))
        idx = ws[ws["alarm_week"] == target_wk].index
        if len(idx) == 0:
            return f"No weekly data for W{target_wk}.", pd.DataFrame()
        cur_rate = float(ws.loc[idx[0], "per_bc"])
        prev_idx = ws.index.get_loc(idx[0])
        prev_rate = float(ws.iloc[prev_idx - 1]["per_bc"]) if prev_idx > 0 else None
        delta = round(cur_rate - prev_rate, 3) if prev_rate else None

        # Find what drove the spike
        dw = df[df["alarm_week"] == target_wk]
        top_dep = dw["depot_id"].value_counts().head(3).to_dict() if "depot_id" in dw.columns else {}
        top_dr  = dw["driver_id"].value_counts().head(5).to_dict()
        top_bs  = dw["bus_no"].value_counts().head(5).to_dict()
        top_svc = dw["svc_no"].value_counts().head(5).to_dict() if "svc_no" in dw.columns else {}

        direction = f"↑ +{delta}" if delta and delta > 0 else (f"↓ {delta}" if delta else "N/A")
        ans = (f"## Root Cause Analysis — W{target_wk} {alarm_choice}\n\n"
               f"**Rate W{target_wk}:** {cur_rate:.3f}/BC | "
               + (f"**vs W{target_wk-1}:** {prev_rate:.3f} ({direction})\n\n" if prev_rate else "\n\n") +
               f"### Contributing Factors\n"
               f"- **Depot breakdown:** {top_dep}\n"
               f"- **Top drivers:** {top_dr}\n"
               f"- **Top buses:** {top_bs}\n"
               f"- **Top services:** {top_svc}\n\n"
               f"### Likely Root Cause Classification\n"
               + ("- 🔴 **FLEET-WIDE**: All depots elevated — check fleet-level factor (training, weather, route changes)\n"
                  if len(top_dep) >= 3 and min(top_dep.values()) > 50
                  else "- 🟠 **DEPOT-SPECIFIC**: One depot dominates — escalate to depot head\n"
                  if top_dep and list(top_dep.values())[0] > sum(list(top_dep.values())[1:])
                  else "- 🟡 **INDIVIDUAL OUTLIERS**: Concentrated in specific drivers/buses — target directly\n") +
               f"\n*Ask `driver <ID>` for a full profile of any driver above.*")
        trend_tbl = ws[["alarm_week","per_bc"]].tail(8).rename(
            columns={"alarm_week":"Week","per_bc":"Rate/BC"})
        trend_tbl["Week"] = "W" + trend_tbl["Week"].astype(int).astype(str)
        return ans, trend_tbl

    # ── 6. FLEET OVERVIEW / GENERAL ──────────────────
    if any(w in ql for w in ["overview", "fleet", "overall", "status", "all", "total", "how many"]):
        dset = df_q if not df_q.empty else df
        total = len(dset)
        drivers = dset["driver_id"].nunique()
        buses   = dset["bus_no"].nunique()
        dep_split = dset["depot_id"].value_counts().to_dict() if "depot_id" in dset.columns else {}
        top_dr = dset["driver_id"].value_counts().head(5)
        ans = (f"## Fleet Overview — {alarm_choice} ({week_label})\n\n"
               f"- **Total alarms:** {total:,}\n"
               f"- **Unique drivers:** {drivers:,}\n"
               f"- **Unique buses:** {buses:,}\n"
               f"- **Depot breakdown:** {dep_split}\n\n"
               f"**Top 5 drivers:** {top_dr.to_dict()}\n\n"
               f"*Try: `top 10 drivers` | `compare depots` | `W9 summary` | `spike W9`*")
        return ans, pd.DataFrame()

    # ── 7. HELP / FALLBACK ───────────────────────────
    # Show actual data examples so user knows real IDs
    sample_drivers = df["driver_id"].value_counts().head(3).index.tolist() if not df.empty else []
    sample_buses   = df["bus_no"].value_counts().head(3).index.tolist() if not df.empty else []
    sample_svcs    = df["svc_no"].value_counts().head(3).index.tolist() if "svc_no" in df.columns else []
    recent_wks     = sorted(df["alarm_week"].unique())[-3:] if not df.empty else []

    ans = (f"## 💬 Root Cause Analyst — Help\n\n"
           f"I can answer questions about **{alarm_choice}** data. Try:\n\n"
           f"| Question Type | Example |\n"
           f"|---------------|--------|\n"
           f"| Top offenders | `top 10 drivers` |\n"
           f"| Top buses | `top 5 buses` |\n"
           f"| Driver profile | `driver {sample_drivers[0] if sample_drivers else '30450'}` |\n"
           f"| Bus profile | `bus {sample_buses[0] if sample_buses else 'SBS123'}` |\n"
           f"| Week summary | `W{recent_wks[-1] if recent_wks else 9} summary` |\n"
           f"| Spike analysis | `why spike W{recent_wks[-1] if recent_wks else 9}` |\n"
           f"| Depot compare | `compare depots` |\n"
           f"| Fleet status | `fleet overview` |\n\n"
           f"*Real examples from your data:* "
           f"driver `{sample_drivers[0] if sample_drivers else 'N/A'}`, "
           f"bus `{sample_buses[0] if sample_buses else 'N/A'}`, "
           f"service `{sample_svcs[0] if sample_svcs else 'N/A'}`")
    return ans, pd.DataFrame()

# ─────────────────────────────────────────────
# AI DEEP DIVE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_ai_deep_dive(_llm, alarm_code, w1_metric, delta, driver_json, bus_json, svc_json, fleet_avg, _v=APP_VERSION):
    if _llm is None:
        return "⚠️ AI not configured. Set AZURE_ENDPOINT, OPENAI_API_KEY, AZURE_DEPLOYMENT in .env"
    payload = {
        "alarm_code": alarm_code,
        "overall_avg_per_bc": round(float(w1_metric), 3),
        "delta_vs_prev_week":  round(float(delta), 3) if delta else None,
        "fleet_avg_per_trip":  round(float(fleet_avg), 3) if fleet_avg else None,
        "top_drivers": driver_json,
        "top_buses":   bus_json,
        "top_services": svc_json,
    }
    try:
        return _llm.predict(f"""
You are a world-class Bus Operations Analyst producing a deep-dive intelligence report.
Alarm type: {alarm_code} | Data: {json.dumps(payload, default=float)}

Write a comprehensive markdown report with ALL of the following sections. Use real numbers from the data throughout.

## Executive Summary
Write 3-4 sentences: total alarms this week, fleet rate vs threshold, which depot/model led contributions, and overall direction vs prior week.

## Operational Patterns

### Key Concentrations
- **Depot-Level Concentration:** Which depot accounts for what % of total alarms? Flag as hotspot if >35%.
- **Model-Level Concentration:** Which bus model generated the most alarms and what % of total? Suggest maintenance/config review if >40%.
- **Driver-Level Contribution:** Name top 2 drivers with alarm counts. Note if they share the same bus model.

### Performance Context
- **Fleet Average Comparison:** State the fleet average alarms/trip. Identify top buses that significantly exceed this average.
- **Depot Performance:** Compare per-bus alarm rate across depots. Identify highest and lowest performing depot.

## Anomaly Detection

### Nexus of Risk
Identify any cross-category patterns: same driver appearing on same high-alarm bus, or same service + same driver combination. If found, flag as compounding risk factor.

### Low Workload High Rate Offenders
List any entities with fewer than 3 trips but above-average alarm rate. These are statistical outliers requiring validation.

## Recommended Actions
| Priority | Entity Type | Recommended Action | Data-Driven Rationale |
|----------|-------------|--------------------|-----------------------|
| High     | [Driver/Bus/Depot] | [Specific action] | [Specific numbers] |
| Medium   | [Driver/Bus/Depot] | [Specific action] | [Specific numbers] |
| Low      | [Driver/Bus/Depot] | [Specific action] | [Specific numbers] |

Use actions like: "Schedule coaching session", "Inspect sensor calibration", "Review route assignment", "Monitor closely next week", "Engage depot head".
""".strip())
    except Exception as e:
        return f"AI error: {e}"


@st.cache_data(show_spinner=False)
def generate_weekly_summary(_llm, alarm_code, alarm_name, week_num, year,
                             w1_metric, delta, w1_count, active_drivers,
                             trend_vals_json, depot_counts_json, _v=APP_VERSION):
    """Rich auto-generated weekly summary for Tab 1 callout panel."""
    if _llm is None:
        return None
    trend_list = json.loads(trend_vals_json) if trend_vals_json else []
    depot_dict = json.loads(depot_counts_json) if depot_counts_json else {}
    direction  = "worsening" if delta > 0 else ("improving" if delta < 0 else "stable")
    try:
        return _llm.predict(f"""
You are a Fleet Operations Analyst writing a weekly performance briefing for operations management.

WEEK DATA:
- Alarm: {alarm_code} ({alarm_name}) | Week {week_num}, {year}
- Rate: {w1_metric:.3f} per BC | Change vs prev week: {delta:+.3f} ({direction})
- Total incidents: {w1_count:,} | Drivers with alarms: {active_drivers}
- 12-week trend (oldest→newest): {trend_list}
- Depot breakdown this week: {depot_dict}

Write a structured briefing with these sections:

**Week {week_num} Performance Summary**
2-3 sentences: state the exact rate, whether it's above/below threshold (3.0=amber, 5.0=red), and the week-on-week direction with magnitude.

**Key Observations**
- Depot with highest contribution and its count
- Whether trend is improving or worsening vs 4-week average
- Any notable spike or dip vs recent weeks

**Recommended Focus**
One specific, actionable recommendation with the data rationale (e.g. which depot or pattern to prioritise).

Keep the entire response under 120 words. Be specific with numbers.
""".strip())
    except Exception as e:
        return f"AI summary error: {e}"


@st.cache_data(show_spinner=False)
def generate_4week_summary(_llm, alarm_code, alarm_name,
                            weeks_json, avg_4wk, trend_direction,
                            repeating_drivers_json, repeating_buses_json, _v=APP_VERSION):
    """Rich auto-generated 4-week pattern narrative for Tab 2."""
    if _llm is None:
        return None
    try:
        return _llm.predict(f"""
You are a Fleet Operations Analyst writing a 4-week systemic pattern report.

PATTERN DATA:
- Alarm: {alarm_code} ({alarm_name})
- 4-week average rate: {avg_4wk:.3f} per BC | Overall trend: {trend_direction}
- Weekly rates: {weeks_json}
- Drivers appearing in 3+ of last 4 weeks (persistent offenders): {repeating_drivers_json}
- Buses appearing in 3+ of last 4 weeks (persistent offenders): {repeating_buses_json}

Write a structured pattern analysis:

**4-Week Trend Summary**
2 sentences: Is the fleet improving, worsening, or cycling? Quote the range from lowest to highest week rate.

**Persistent Offenders**
- Top repeat driver(s) with total alarm count across the 4 weeks
- Top repeat bus(es) with total alarm count — note if same bus/driver keep appearing together

**Pattern Classification**
State whether this looks like: SYSTEMIC FLEET ISSUE / DEPOT-SPECIFIC PATTERN / INDIVIDUAL OUTLIER(S) / MIXED

**Recommended Action**
One targeted action for the most persistent offender, with specific data rationale.

Keep under 130 words. Use bold for entity IDs and numbers.
""".strip())
    except Exception as e:
        return f"AI summary error: {e}"


@st.cache_data(show_spinner=False)
def generate_forecast_summary(_llm, alarm_code, alarm_name,
                               current_rate, proj_rate, completion_pct,
                               trend_vals_json, depot_stats_json,
                               next_week_num, _v=APP_VERSION):
    """Rich auto-generated forecast narrative for Tab 3."""
    if _llm is None:
        return None
    risk = "CRITICAL (above 5.0 red threshold)" if proj_rate > 5 else ("ELEVATED (above 3.0 amber threshold)" if proj_rate > 3 else "NORMAL (within safe range below 3.0)")
    try:
        return _llm.predict(f"""
You are a Fleet Operations Analyst writing a risk forecast briefing for operations leadership.

FORECAST DATA:
- Alarm: {alarm_code} ({alarm_name})
- Current week rate (partial): {current_rate:.3f} | Projected full-week rate: {proj_rate:.3f}
- Risk classification: {risk}
- Week completion: {completion_pct:.0f}% of data available
- 12-week trend: {trend_vals_json}
- Depot alarm counts this week: {depot_stats_json}
- Next week: W{next_week_num}

Write a structured forecast briefing with these sections:

**Risk Assessment**
State the projected rate vs threshold clearly. Is it Critical/Elevated/Normal? Confidence level based on % week complete.

**Contributing Factors**
- Primary driver of the projected rate (which depot or trend)
- Whether the 12-week trend supports improvement or continued pressure

**W{next_week_num} Outlook**
Predict direction for next week based on trajectory. Mention any cyclical factors if visible in trend data.

**Pre-emptive Action**
One specific action to take before W{next_week_num} starts, with clear rationale from the data.

Keep under 130 words. Be direct and specific with numbers.
""".strip())
    except Exception as e:
        return f"AI forecast error: {e}"


@st.cache_data(show_spinner=False)
def generate_email_ai_analysis(_llm, summaries_json, alarm_details_json, report_date_str, _v=APP_VERSION):
    """AI-written analysis paragraph injected into the alert email body."""
    if _llm is None:
        return None
    try:
        return _llm.predict(f"""
You are Head of Fleet Safety writing an email analysis paragraph for management.

Report date: {report_date_str}
Alert summary: {summaries_json}
Alarm details (type, projected rate, top contributors): {alarm_details_json}

Write ONE professional email paragraph (5–7 sentences, no headers, no bullets):
- Open with the overall safety situation
- Identify the most critical alarm type and its projected rate
- Mention the depot or driver pattern most at risk
- Recommend immediate short-term action
- Close with confidence in the team to resolve this
Use formal, direct language suitable for senior management.
""".strip())
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# FEATURE 1 — AUTO MORNING BRIEFING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def generate_morning_briefing(_llm, df_alarm_json: str, weekly_json: str, alarm_choice: str) -> str:
    """Generate a plain-English top-3 briefing on load."""
    if _llm is None:
        return ""
    try:
        prompt = f"""You are a senior Fleet Operations Analyst. Based on the data snapshot below, 
write exactly 3 bullet points (no intro, no outro) that an ops manager needs to know RIGHT NOW.
Each bullet must be one sentence with specific numbers. Start each with a status emoji (🔴/🟡/🟢).

Alarm: {alarm_choice}
Weekly trend (last 8 weeks, rate/BC): {weekly_json}
Fleet snapshot: {df_alarm_json}

Rules:
- Flag the single highest-risk driver with their rate
- State whether fleet is improving or worsening with exact rate change
- Highlight any depot that is an outlier vs others

Output ONLY the 3 bullet points. No headers. No preamble."""
        if hasattr(_llm, "_client"):
            resp = _llm._client.chat.completions.create(
                model=_llm._deployment,
                messages=[{"role":"user","content":prompt}],
                temperature=0.1, max_tokens=300)
            return resp.choices[0].message.content.strip()
        return _llm.predict(prompt)
    except Exception:
        return ""

# ─────────────────────────────────────────────
# FEATURE 5 — EXPLAIN CHART
# ─────────────────────────────────────────────
def explain_chart(_llm, chart_title: str, chart_data_summary: str, alarm_choice: str) -> str:
    """Generate a 2-sentence plain-English chart explanation."""
    if _llm is None:
        return "Connect AI to enable chart explanations."
    try:
        prompt = f"""In exactly 2 sentences, explain what this chart shows and what action it implies.
Chart: {chart_title}
Alarm type: {alarm_choice}
Data summary: {chart_data_summary}
Be specific with numbers. End with one concrete recommended action."""
        if hasattr(_llm, "_client"):
            resp = _llm._client.chat.completions.create(
                model=_llm._deployment,
                messages=[{"role":"user","content":prompt}],
                temperature=0.1, max_tokens=150)
            return resp.choices[0].message.content.strip()
        return _llm.predict(prompt)
    except Exception as e:
        return f"Could not generate explanation: {e}"

def main():
    llm, llm_status = initialize_llm()

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-title">Fleet Intelligence Hub</div>', unsafe_allow_html=True)
        st.markdown("### 🚍 Control Panel")

        with st.expander("📂 Data Sources", expanded=False):
            tele_file  = st.text_input("Telematics CSV",  _cfg("TELEMATICS_URL",  "telematics_new_data_2207.csv"))
            excl_file  = st.text_input("Exclusions CSV",  _cfg("EXCLUSIONS_URL",  "vehicle_exclusions.csv"))
            head_file  = st.text_input("Headcounts CSV",  _cfg("HEADCOUNTS_URL",  "depot_headcounts.csv"))
            model_file = st.text_input("Bus Models CSV",  _cfg("MODEL_URL",        "model.csv"))

        _mtime = max(_maybe_mtime(tele_file), _maybe_mtime(excl_file), _maybe_mtime(head_file))
        if st.button("🔄 Reload Data"):
            st.cache_data.clear(); st.rerun()

        df_raw, headcounts, weekly_all, diag = load_and_process_data(
            tele_file, excl_file, head_file, model_file, _mtime)

        if df_raw.empty:
            st.error("⚠️ No data loaded. Check CSV paths above.")
            st.stop()

        with st.expander("📊 Data Info", expanded=False):
            st.caption(f"Rows: **{diag['rows']:,}** | Depots: **{diag['depots']}**")
            st.caption(f"Range: {diag['date_range']}")
            st.caption(f"Model matched: **{diag['model_matched']:,}**")
            if llm:
                st.caption(f"AI: ✅ {llm_status}")
            else:
                st.caption(f"AI: ❌ {llm_status}")
                st.caption(f"🔍 .env: `{_ENV_PATH}`")
                st.caption(f"   File: {'✅ found' if os.path.exists(_ENV_PATH) else '❌ NOT FOUND'}")
                ep = _cfg("AZURE_ENDPOINT"); ak = _cfg("OPENAI_API_KEY"); dp = _cfg("AZURE_DEPLOYMENT")
                st.caption(f"   ENDPOINT:   {'✅' if ep else '❌ empty'}")
                st.caption(f"   API_KEY:    {'✅' if ak else '❌ empty'}")
                st.caption(f"   DEPLOYMENT: {'✅' if dp else '❌ empty'}")
                # Show package status
                st.caption(f"   langchain:  {'✅' if LANGCHAIN_OK else '❌ not installed'}")
                st.caption(f"   openai SDK: {'✅' if OPENAI_SDK_OK else '❌ not installed'}")
                if not LANGCHAIN_OK and not OPENAI_SDK_OK:
                    st.caption("💡 Run: `pip install openai`")
                elif not LANGCHAIN_OK and OPENAI_SDK_OK:
                    st.caption("💡 openai SDK available — click Reconnect AI")
            st.caption(f"Chatbot: {'✅ ReAct Agent' if (llm and LANGCHAIN_OK) else ('✅ Direct AI' if llm else 'ℹ️ Rule-based fallback')}")

        if st.button("🔄 Reconnect AI", help="Re-reads .env and retries Azure OpenAI connection"):
            _clear_llm_caches()
            st.rerun()

        st.markdown("---")
        alarm_choice = st.selectbox("Alarm Type", options=list(ALARM_MAP.keys()),
                                    format_func=lambda x: f"{x} — {ALARM_MAP[x]['long']}")

        depot_opts     = sorted(headcounts["depot_id"].unique()) if not headcounts.empty else []
        default_depots = [d for d in ["WDLAND","KRANJI","JURONG"] if d in depot_opts] or depot_opts[:2]
        depots         = st.multiselect("Depot(s)", depot_opts, default=default_depots)
        only_completed = st.checkbox("Only completed weeks", value=True)
        exclude_null   = st.checkbox("Exclude missing drivers", value=True)

    if not depots:
        st.warning("Please select at least one depot.")
        st.stop()
    depots_tuple = tuple(sorted(depots))

    # ── Slice & KPIs ─────────────────────────
    df_alarm, week_map, weekly = slice_by_filters(
        df_raw, weekly_all, depots_tuple, alarm_choice, only_completed, exclude_null)
    weekly_sum, total_hc = per_week_kpis(weekly, headcounts, depots_tuple)

    # ── Agent ────────────────────────────────
    agent_tools = build_agent_tools(df_alarm, weekly_sum, total_hc, alarm_choice, df_raw, depots_tuple)
    react_agent = build_react_agent(llm, agent_tools) if llm else None

    # ── FEATURE 3: DRIVER WATCH LIST in sidebar ──────────────────────
    if not df_alarm.empty:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👁️ Driver Watch List")
            try:
                _wd = df_alarm.copy()
                _wd_rate = (_wd.groupby("driver_id").agg(a=("driver_id","count"), t=("trip_id_norm","nunique"))
                            .assign(rate=lambda x: (x.a/x.t.clip(lower=1)).round(2)))
                _fleet_avg_wd = _wd_rate["rate"].mean()
                # Chronic flag: present in 3+ weeks
                _wk_cnt = (_wd.groupby(["driver_id","alarm_week"]).size()
                           .reset_index().groupby("driver_id")["alarm_week"].nunique())
                _wd_rate["weeks"] = _wd_rate.index.map(_wk_cnt).fillna(0).astype(int)
                _wd_rate["score"] = _wd_rate["rate"] * (1 + 0.2*(_wd_rate["weeks"]-1).clip(lower=0))
                _top5 = _wd_rate.sort_values("score", ascending=False).head(5)
                for _drv, _row in _top5.iterrows():
                    _drv_str = str(_drv).replace(".0","")
                    _badge = "🔴" if _row["rate"] > 5 else ("🟡" if _row["rate"] > 3 else "🟢")
                    _chronic = " ⚡chronic" if _row["weeks"] >= 3 else ""
                    _wl_line = (
                        f"{_badge} **{_drv_str}**{_chronic} "
                        f"<span style='font-size:0.8rem;color:#64748b'>"
                        f"{_row['rate']:.2f}/trip | {int(_row['weeks'])}wk</span>"
                    )
                    st.markdown(_wl_line, unsafe_allow_html=True)
                st.caption(f"Fleet avg: {_fleet_avg_wd:.2f}/trip · {alarm_choice}")
            except Exception:
                st.caption("Watch list unavailable")

    # ─────────────────────────────────────────
    # MAIN CONTENT
    # ─────────────────────────────────────────
    st.markdown(f"## 📈 {ALARM_MAP[alarm_choice]['long']} Intelligence Hub")

    if week_map.empty:
        st.info("No data found with current filters.")
        return

    # Week picker
    sel_label = st.selectbox("Select Week", options=week_map["label"].tolist(), index=len(week_map)-1)
    sel_row   = week_map.loc[week_map["label"] == sel_label].iloc[0]
    w1_year, w1_week = int(sel_row["alarm_year"]), int(sel_row["alarm_week"])

    if weekly_sum.empty or total_hc <= 0:
        st.error("No weekly KPIs. Check headcounts CSV.")
        return

    w1_row = weekly_sum[(weekly_sum["alarm_year"]==w1_year) & (weekly_sum["alarm_week"]==w1_week)]
    if w1_row.empty:
        st.error(f"No data for W{w1_week} {w1_year}.")
        return

    w1_metric = float(w1_row["per_bc"].iloc[0])
    w1_count  = int(w1_row["alarm_sum"].iloc[0])
    w1_sow    = pd.to_datetime(f"{w1_year}{w1_week:02d}1", format="%G%V%w")
    w2_sow    = w1_sow - pd.Timedelta(weeks=1)
    w2_year, w2_week = int(w2_sow.isocalendar().year), int(w2_sow.isocalendar().week)
    w2_row    = weekly_sum[(weekly_sum["alarm_year"]==w2_year) & (weekly_sum["alarm_week"]==w2_week)]
    delta     = (w1_metric - float(w2_row["per_bc"].iloc[0])) if not w2_row.empty else 0.0
    w2_count  = int(w2_row["alarm_sum"].iloc[0]) if not w2_row.empty else 0

    # Status badge
    if w1_metric > 5:
        badge_html = '<span class="badge-red">🔴 CRITICAL</span>'
    elif w1_metric > 3:
        badge_html = '<span class="badge-amber">🟡 ELEVATED</span>'
    else:
        badge_html = '<span class="badge-green">🟢 NORMAL</span>'
    st.markdown(f"**W{w1_week} · {w1_year}** &nbsp; {badge_html}", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{alarm_choice} Rate per BC",  f"{w1_metric:.2f}", delta=f"{delta:+.2f} vs prev week", delta_color="inverse")
    k2.metric("Total Incidents",              f"{w1_count:,}",    delta=f"{w1_count-w2_count:+,}",    delta_color="inverse")
    k3.metric("Headcount (selected)",         f"{int(total_hc):,}")
    active_drv = df_alarm[df_alarm["alarm_week"]==w1_week]["driver_id"].nunique() if not df_alarm.empty else 0
    k4.metric("Drivers with Alarms",          f"{active_drv:,}")

    st.markdown("---")

    # ── FEATURE 2: ANOMALY ALERT BANNER ──────────────────────────────
    if not df_alarm.empty:
        try:
            _dr_rates = (df_alarm.groupby("driver_id").agg(a=("driver_id","count"), t=("trip_id_norm","nunique"))
                         .assign(rate=lambda x: (x.a/x.t.clip(lower=1)).round(3)))
            _mean_r = _dr_rates["rate"].mean(); _std_r = _dr_rates["rate"].std()
            _outliers = _dr_rates[_dr_rates["rate"] > _mean_r + 2*_std_r]
            _wks = sorted(df_alarm["alarm_week"].unique())
            _spike_msg = ""
            if len(_wks) >= 2:
                _d_curr = df_alarm[df_alarm["alarm_week"] == _wks[-1]]
                _d_prev = df_alarm[df_alarm["alarm_week"] == _wks[-2]]
                _r_curr = len(_d_curr)/_d_curr["trip_id_norm"].nunique() if _d_curr["trip_id_norm"].nunique()>0 else 0
                _r_prev = len(_d_prev)/_d_prev["trip_id_norm"].nunique() if _d_prev["trip_id_norm"].nunique()>0 else 0
                if _r_prev > 0 and (_r_curr - _r_prev)/_r_prev > 0.20:
                    _spike_msg = f"  🔺 W{int(_wks[-1])} rate spiked +{(_r_curr-_r_prev)/_r_prev*100:.0f}% vs prior week."
            if len(_outliers) > 0 or _spike_msg:
                _banner_parts = []
                if len(_outliers) > 0:
                    _top_o = _outliers.sort_values("rate", ascending=False).head(2)
                    _names = ", ".join([f"Driver {str(idx)} ({row['rate']:.2f}/trip)" for idx, row in _top_o.iterrows()])
                    _banner_parts.append(f"⚠️ **{len(_outliers)} statistical outlier(s)** detected: {_names}")
                if _spike_msg:
                    _banner_parts.append(_spike_msg.strip())
                st.warning("  |  ".join(_banner_parts) + f"  ·  *{alarm_choice} alerts*")
        except Exception:
            pass

    # ── FEATURE 1: AUTO MORNING BRIEFING ─────────────────────────────
    if "morning_briefing_key" not in st.session_state:
        st.session_state["morning_briefing_key"] = f"{alarm_choice}_{depots_tuple}_{w1_week}"
    _brief_key = f"{alarm_choice}_{depots_tuple}_{w1_week}"
    if llm and _brief_key != st.session_state.get("morning_briefing_key_done",""):
        try:
            _wt_data = weekly_sum.tail(8)[["alarm_week","per_bc"]].to_dict("records")
            _snap = {
                "total_alarms": len(df_alarm),
                "fleet_rate": round(len(df_alarm)/df_alarm["trip_id_norm"].nunique(),3) if df_alarm["trip_id_norm"].nunique()>0 else 0,
                "top_driver": df_alarm["driver_id"].value_counts().head(1).to_dict() if not df_alarm.empty else {},
                "depot_split": df_alarm["depot_id"].value_counts().to_dict() if not df_alarm.empty else {},
                "current_week_rate": w1_metric,
                "delta_vs_prior": round(delta, 3),
            }
            _briefing = generate_morning_briefing(llm, json.dumps(_snap), json.dumps(_wt_data), alarm_choice)
            if _briefing:
                st.session_state["morning_briefing_text"] = _briefing
                st.session_state["morning_briefing_key_done"] = _brief_key
        except Exception:
            pass

    if st.session_state.get("morning_briefing_text"):
        with st.expander("☀️ **Today's Fleet Briefing** — AI Morning Summary", expanded=True):
            st.markdown(st.session_state["morning_briefing_text"])
            if st.button("🔄 Refresh Briefing", key="refresh_briefing"):
                st.session_state.pop("morning_briefing_text", None)
                st.session_state.pop("morning_briefing_key_done", None)
                generate_morning_briefing.clear()
                st.rerun()

    # ── TABS — original layout ────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Weekly Deep Dive",
        "🧠 4-Week Pattern",
        "🔮 Risk Forecast",
        "🤖 AI Deep Dive",
        "📧 Consolidated Alert",
    ])

    # ── Week-level slices (shared across tabs) ──
    w1_events    = pd.DataFrame()
    trips_unique = pd.DataFrame()
    if not df_alarm.empty:
        w1_events    = df_alarm[(df_alarm["alarm_year"]==w1_year) & (df_alarm["alarm_week"]==w1_week)]
        trips_unique = w1_events.drop_duplicates("trip_id_norm")

    # ────────────────────────────────────────
    # TAB 1 — Weekly Deep Dive
    # ────────────────────────────────────────
    with t1:
        chart_data = weekly_sum[weekly_sum["start_of_week"] <= w1_sow].tail(12)

        # Pre-compute metrics once (shared across sub-tabs)
        cols_show = ["Alarm Count","Alarm Trips","Alarms per Trip","Alarms per Hour","Total Duration (hr)"]
        df_dr = calc_metrics(w1_events, trips_unique, "driver_id")
        df_bs = calc_metrics(w1_events, trips_unique, "bus_no")
        df_sv = calc_metrics(w1_events, trips_unique, "svc_no")

        st1a, st1b, st1c, st1d = st.tabs([
            "📈 Main Chart + Summary",
            "📊 Charts",
            "📋 Tables",
            "🤖 GenAI Analysis",
        ])

        # ── SUB-TAB A: Main Chart + Summary ─────────────────────────────
        with st1a:
            # Main 12-week trend chart
            st.plotly_chart(trend_chart_12wk(chart_data, "per_bc", ALARM_MAP[alarm_choice]["long"]),
                            use_container_width=True)

            # Explain chart button (Feature 5)
            if llm:
                with st.expander("💡 Explain this chart", expanded=False):
                    chart_summary = f"12-week rates: {chart_data['per_bc'].round(2).tolist()}, latest W{w1_week}: {w1_metric:.3f}/BC"
                    st.markdown(explain_chart(llm, f"12-Week Trend {alarm_choice}", chart_summary, alarm_choice))

            # AI Weekly Summary (always visible)
            ai_weekly = None
            if llm:
                with st.spinner("🤖 Generating weekly AI summary…"):
                    trend_vals_12 = chart_data["per_bc"].round(3).tolist()
                    depot_counts_w1 = w1_events["depot_id"].value_counts().to_dict() if not w1_events.empty and "depot_id" in w1_events.columns else {}
                    ai_weekly = generate_weekly_summary(
                        llm, alarm_choice, ALARM_MAP[alarm_choice]["long"],
                        w1_week, w1_year, w1_metric, delta, w1_count, active_drv,
                        json.dumps(trend_vals_12), json.dumps(depot_counts_w1),
                    )

            dir_word = "↑ up" if delta > 0 else "↓ down"
            if ai_weekly:
                insight_text = ai_weekly
                insight_style = "background:#eff6ff;border-left:4px solid #2563eb;"
                label_style = "color:#1e40af;"
            else:
                insight_text = (
                    f"Week {w1_week} · {w1_year}: {alarm_choice} rate is <b>{w1_metric:.3f}</b> per bus-change "
                    f"({dir_word} {abs(delta):.3f} vs prior week). "
                    f"{w1_count:,} events across {active_drv} active drivers. "
                    f"<i>Connect Azure OpenAI for full AI narrative.</i>"
                )
                insight_style = "background:#f8fafc;border-left:4px solid #94a3b8;"
                label_style = "color:#475569;"

            st.markdown(
                f'<div style="{insight_style}padding:14px 18px;border-radius:0 8px 8px 0;margin:12px 0;">'
                f'<span style="font-size:0.75rem;font-weight:700;{label_style}text-transform:uppercase;letter-spacing:1px;">'
                f'🤖 AI Weekly Insights — W{w1_week} · {w1_year}</span>'
                f'<p style="margin:6px 0 0;color:#1e293b;font-size:0.9rem;line-height:1.6;">{insight_text}</p>'
                f'</div>', unsafe_allow_html=True)

            if llm:
                if st.button("🔄 Regenerate AI Summary", key="t1_regen_a"):
                    generate_weekly_summary.clear(); st.rerun()

        # ── SUB-TAB B: Charts ────────────────────────────────────────────
        with st1b:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_depot = depot_bar_chart(w1_events, alarm_choice)
                st.plotly_chart(fig_depot, use_container_width=True)
                if llm:
                    with st.expander("💡 Explain", expanded=False):
                        depot_summary = w1_events["depot_id"].value_counts().to_dict() if not w1_events.empty else {}
                        st.markdown(explain_chart(llm, f"Depot Distribution {alarm_choice}", str(depot_summary), alarm_choice))
            with col_b:
                if "model" in df_alarm.columns:
                    fig_model = model_pie_chart(w1_events)
                    st.plotly_chart(fig_model, use_container_width=True)
                    if llm:
                        with st.expander("💡 Explain", expanded=False):
                            model_summary = w1_events["model"].value_counts().to_dict() if not w1_events.empty else {}
                            st.markdown(explain_chart(llm, f"Model Breakdown {alarm_choice}", str(model_summary), alarm_choice))

        # ── SUB-TAB C: Tables ────────────────────────────────────────────
        with st1c:
            st.markdown(f"**W{w1_week} · {w1_year} — {len(w1_events):,} events**")
            with st.expander("🧑‍✈️ Drivers", expanded=True):
                if not df_dr.empty:
                    flt = df_dr["Alarm Count"].sum()/df_dr["Alarm Trips"].sum() if df_dr["Alarm Trips"].sum()>0 else 0
                    st.caption(f"Fleet avg: **{flt:.2f}** alarms/trip")
                    st.dataframe(df_dr.sort_values(["Alarms per Trip","Alarm Count"], ascending=False)[cols_show].round(2),
                                 use_container_width=True)
                else:
                    st.info("No driver data for this week.")
            with st.expander("🚌 Buses", expanded=False):
                if not df_bs.empty:
                    st.dataframe(df_bs.sort_values(["Alarms per Trip","Alarm Count"], ascending=False)[cols_show].round(2),
                                 use_container_width=True)
                else:
                    st.info("No bus data.")
            with st.expander("🗺️ Services", expanded=False):
                if not df_sv.empty:
                    st.dataframe(df_sv.sort_values(["Alarms per Trip","Alarm Count"], ascending=False)[cols_show].round(2),
                                 use_container_width=True)
                else:
                    st.info("No service data.")

        # ── SUB-TAB D: GenAI Analysis ────────────────────────────────────
        with st1d:
            if llm:
                col_ai1, col_ai2 = st.columns([1,3])
                if col_ai2.button("🤖 Generate Full Deep Dive Analysis", type="primary", key="t1_ai"):
                    with st.spinner("Generating detailed analysis..."):
                        dr_p = df_dr.reset_index() if not df_dr.empty else pd.DataFrame()
                        bs_p = df_bs.reset_index() if not df_bs.empty else pd.DataFrame()
                        sv_p = df_sv.reset_index() if not df_sv.empty else pd.DataFrame()
                        def slim(d, key, n=10):
                            if d.empty: return []
                            cols_s = [key,"Alarm Count","Alarm Trips","Alarms per Trip","Total Duration (hr)"]
                            cols_s = [c for c in cols_s if c in d.columns]
                            return d.sort_values("Alarms per Trip", ascending=False).head(n)[cols_s].round(3).to_dict("records")
                        flt_avg = (dr_p["Alarm Count"].sum()/dr_p["Alarm Trips"].sum()
                                   if not dr_p.empty and dr_p["Alarm Trips"].sum()>0 else 0)
                        result = generate_ai_deep_dive(
                            llm, alarm_choice, w1_metric, delta,
                            json.dumps(slim(dr_p,"driver_id")),
                            json.dumps(slim(bs_p,"bus_no")),
                            json.dumps(slim(sv_p,"svc_no")), flt_avg)
                        st.session_state["t1_deep_dive"] = result
                if "t1_deep_dive" in st.session_state:
                    with st.expander("📄 Full Deep Dive Report", expanded=True):
                        st.markdown(st.session_state["t1_deep_dive"])
            else:
                st.info("Configure AI credentials in .env to enable AI insights.")

    # ────────────────────────────────────────
    # TAB 2 — 4-Week Pattern
    # ────────────────────────────────────────
    with t2:
        last4_weeks = list(weekly_sum["alarm_week"].unique()[-4:])
        avg_4wk     = weekly_sum.tail(4)["per_bc"].mean()
        weeks_data  = {}
        for wk in last4_weeks:
            row = weekly_sum[weekly_sum["alarm_week"] == wk]
            weeks_data[f"W{int(wk)}"] = round(float(row["per_bc"].values[0]), 2) if len(row) > 0 else 0.0
        _rates    = list(weeks_data.values())
        trend_dir = "improving ↓" if _rates[-1] < _rates[0] else ("worsening ↑" if _rates[-1] > _rates[0] else "stable →")

        # Build 4-week chart data
        chart_4wk = weekly_sum.tail(8)

        # Pre-compute tables
        drv_4 = pd.DataFrame()
        bus_4 = pd.DataFrame()
        if not df_alarm.empty:
            d4 = df_alarm[df_alarm["alarm_week"].isin(last4_weeks)]
            drv_4 = d4.groupby(["driver_id","alarm_week"]).size().unstack(fill_value=0)
            if not drv_4.empty:
                drv_4["Total"] = drv_4.sum(axis=1)
                drv_4 = drv_4.sort_values("Total", ascending=False).head(20)
                drv_4 = drv_4.rename(columns={w: f"W{int(w)}" for w in drv_4.columns if w != "Total"})
                # Ensure driver_id index is string
                drv_4.index = drv_4.index.astype(str).str.replace(r"\.0$", "", regex=True)
            bus_4 = d4.groupby(["bus_no","alarm_week"]).size().unstack(fill_value=0)
            if not bus_4.empty:
                bus_4["Total"] = bus_4.sum(axis=1)
                bus_4 = bus_4.sort_values("Total", ascending=False).head(20)
                bus_4 = bus_4.rename(columns={w: f"W{int(w)}" for w in bus_4.columns if w != "Total"})

        st2a, st2b, st2c, st2d = st.tabs([
            "📈 Main Chart + Summary",
            "📊 Charts",
            "📋 Tables",
            "🤖 GenAI Analysis",
        ])

        # ── SUB-TAB A: Main Chart + Summary ─────────────────────────────
        with st2a:
            st.plotly_chart(trend_chart_12wk(chart_4wk, "per_bc", ALARM_MAP[alarm_choice]["long"]),
                            use_container_width=True)
            if llm:
                with st.expander("💡 Explain this chart", expanded=False):
                    chart_sum = f"4-week rates: {weeks_data}, trend: {trend_dir}, avg: {avg_4wk:.3f}/BC"
                    st.markdown(explain_chart(llm, "4-Week Pattern Trend", chart_sum, alarm_choice))

            # AI 4-week summary
            ai_4wk = None
            if llm:
                with st.spinner("🤖 Generating 4-week pattern analysis…"):
                    _rep_drivers, _rep_buses = {}, {}
                    if not df_alarm.empty:
                        for eid, grp in df_alarm[df_alarm["alarm_week"].isin(last4_weeks)].groupby("driver_id"):
                            if grp["alarm_week"].nunique() >= 3:
                                _rep_drivers[str(eid)] = int(len(grp))
                        for eid, grp in df_alarm[df_alarm["alarm_week"].isin(last4_weeks)].groupby("bus_no"):
                            if grp["alarm_week"].nunique() >= 3:
                                _rep_buses[str(eid)] = int(len(grp))
                    ai_4wk = generate_4week_summary(
                        llm, alarm_choice, ALARM_MAP[alarm_choice]["long"],
                        json.dumps(weeks_data), avg_4wk, trend_dir,
                        json.dumps(dict(sorted(_rep_drivers.items(), key=lambda x: -x[1])[:5])),
                        json.dumps(dict(sorted(_rep_buses.items(),   key=lambda x: -x[1])[:5])),
                    )

            if ai_4wk:
                box_style4 = "background:#f0fdf4;border-left:4px solid #16a34a;"
                lbl_style4 = "color:#15803d;"
                insight_text4 = ai_4wk
            else:
                weeks_summary = " → ".join([f"{k}: {v:.2f}" for k, v in weeks_data.items()])
                insight_text4 = (
                    f"4-week baseline: <b>{avg_4wk:.3f}</b> alarms per bus-change. Trend is <b>{trend_dir}</b>. "
                    f"Weekly rates — {weeks_summary}. "
                    f"<i>Connect Azure OpenAI for repeating-offender analysis.</i>"
                )
                box_style4 = "background:#f8fafc;border-left:4px solid #94a3b8;"
                lbl_style4 = "color:#475569;"

            st.markdown(
                f'<div style="{box_style4}padding:14px 18px;border-radius:0 8px 8px 0;margin:12px 0;">'
                f'<span style="font-size:0.75rem;font-weight:700;{lbl_style4}text-transform:uppercase;letter-spacing:1px;">'
                f'🧠 AI 4-Week Pattern Analysis</span>'
                f'<p style="margin:6px 0 0;color:#1e293b;font-size:0.9rem;line-height:1.6;">{insight_text4}</p>'
                f'</div>', unsafe_allow_html=True)
            st.metric("4-Week Baseline (avg per BC)", f"{avg_4wk:.2f}")
            if llm:
                if st.button("🔄 Regenerate Pattern Summary", key="t2_regen_a"):
                    generate_4week_summary.clear(); st.rerun()

        # ── SUB-TAB B: Charts ────────────────────────────────────────────
        with st2b:
            fig_drv_bar = driver_trend_bar(df_alarm, last4_weeks)
            st.plotly_chart(fig_drv_bar, use_container_width=True)
            if llm:
                with st.expander("💡 Explain", expanded=False):
                    top5_drv = df_alarm[df_alarm["alarm_week"].isin(last4_weeks)]["driver_id"].value_counts().head(5).to_dict() if not df_alarm.empty else {}
                    st.markdown(explain_chart(llm, "Top Drivers 4-Week Bar Chart", str(top5_drv), alarm_choice))

        # ── SUB-TAB C: Tables ────────────────────────────────────────────
        with st2c:
            with st.expander("🧑‍✈️ 4-Week Driver Alarm Table", expanded=True):
                if not drv_4.empty:
                    st.dataframe(drv_4, use_container_width=True)
                else:
                    st.info("No driver data.")
            with st.expander("🚌 4-Week Bus Alarm Table", expanded=False):
                if not bus_4.empty:
                    st.dataframe(bus_4, use_container_width=True)
                else:
                    st.info("No bus data.")

        # ── SUB-TAB D: GenAI Analysis ────────────────────────────────────
        with st2d:
            if llm:
                if st.button("🧠 Run Full Systemic Scan (AI)", type="primary", key="t2_ai"):
                    with st.spinner("Scanning patterns..."):
                        context_data = {}
                        for wk in last4_weeks:
                            d = df_alarm[df_alarm["alarm_week"] == wk]
                            if not d.empty:
                                context_data[f"W{int(wk)}"] = {
                                    "total": int(len(d)),
                                    "per_bc": round(float(weekly_sum[weekly_sum["alarm_week"]==wk]["per_bc"].values[0])
                                                   if len(weekly_sum[weekly_sum["alarm_week"]==wk])>0 else 0, 3),
                                    "top_drivers": {str(k): v for k,v in d["driver_id"].value_counts().head(8).to_dict().items()},
                                    "top_buses":   d["bus_no"].value_counts().head(8).to_dict(),
                                    "top_services":d["svc_no"].value_counts().head(8).to_dict(),
                                    "depot_split": d["depot_id"].value_counts().to_dict(),
                                }
                        try:
                            result = llm.predict(f"""You are a world-class Bus Operations Analyst for {alarm_choice}.
DATA (last 4 weeks): {json.dumps(context_data, default=float)}

## 📈 Trend Overview
## 🔍 Persistent Offenders (Drivers, Buses, Services)
## 🔗 Operational Nexus
## 📊 Depot Analysis
## ✅ Priority Action Plan (table with Priority, Target, Action, Evidence)

Use specific IDs and counts. Be direct and operational.""".strip())
                            st.session_state["t2_systemic"] = result
                        except Exception as e:
                            st.error(f"AI error: {e}")
                if "t2_systemic" in st.session_state:
                    with st.expander("📄 Systemic Scan Report", expanded=True):
                        st.markdown(st.session_state["t2_systemic"])
            else:
                st.info("Configure AI credentials in .env to enable systemic scan.")

    # ────────────────────────────────────────
    # TAB 3 — Risk Forecast
    # ────────────────────────────────────────
    with t3:
        st3a, st3b, st3c = st.tabs([
            "📈 Main Chart + Summary",
            "📊 Forecast Chart",
            "🤖 GenAI Analysis",
        ])

        with st3a:
            if not df_alarm.empty and not weekly_sum.empty:
                proj_val, comp_df, explainer = calculate_smart_forecast(df_alarm, weekly_sum, w1_week, total_hc)
                f1, f2, f3 = st.columns(3)
                f1.metric("Projected End-of-Week Rate", f"{proj_val:.2f}", delta_color="inverse")
                f2.metric("Data Through",               explainer.get("day","N/A"))
                f3.metric("Completion Estimate",         f"{explainer.get('completion_rate',0)*100:.0f}%")
                if proj_val > 5:
                    st.markdown('<div class="alert-red">🔴 <strong>CRITICAL</strong> — Projected to exceed 5.0 threshold</div>', unsafe_allow_html=True)
                elif proj_val > 3:
                    st.markdown('<div class="alert-amber">🟡 <strong>ELEVATED</strong> — Projected to exceed 3.0 threshold</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-green">🟢 <strong>ON TRACK</strong> — Within normal range</div>', unsafe_allow_html=True)

                # AI forecast panel
                next_wk = w1_week + 1 if w1_week < 52 else 1
                ai_forecast = None
                if llm:
                    with st.spinner("🤖 Generating forecast analysis…"):
                        trend_vals_f = weekly_sum.tail(12)["per_bc"].round(2).tolist()
                        depot_stats_f = {}
                        for dep in depots_tuple:
                            d = df_alarm[df_alarm["depot_id"] == dep]
                            if not d.empty:
                                depot_stats_f[dep] = int(len(d[d["alarm_week"] == w1_week]))
                        ai_forecast = generate_forecast_summary(
                            llm, alarm_choice, ALARM_MAP[alarm_choice]["long"],
                            w1_metric, proj_val,
                            explainer.get("completion_rate", 0) * 100,
                            json.dumps(trend_vals_f), json.dumps(depot_stats_f), next_wk,
                        )
                if ai_forecast:
                    fc_text  = ai_forecast; fc_style = "background:#fff7ed;border-left:4px solid #ea580c;"; fc_lstyle = "color:#c2410c;"
                else:
                    status_word = "CRITICAL" if proj_val > 5 else ("ELEVATED" if proj_val > 3 else "ON TRACK")
                    fc_text = (f"Projected W{next_wk} rate: <b>{proj_val:.2f}</b> ({status_word}). "
                               f"Current W{w1_week} rate is {w1_metric:.3f}. "
                               f"Week is {explainer.get('completion_rate',0)*100:.0f}% complete. "
                               f"<i>Connect Azure OpenAI for full narrative outlook.</i>")
                    fc_style  = "background:#f8fafc;border-left:4px solid #94a3b8;"
                    fc_lstyle = "color:#475569;"
                st.markdown(
                    f'<div style="{fc_style}padding:14px 18px;border-radius:0 8px 8px 0;margin:12px 0;">'
                    f'<span style="font-size:0.75rem;font-weight:700;{fc_lstyle}text-transform:uppercase;letter-spacing:1px;">'
                    f'🔮 AI Forecast & Next-Week Outlook — W{next_wk}</span>'
                    f'<p style="margin:6px 0 0;color:#1e293b;font-size:0.9rem;line-height:1.6;">{fc_text}</p>'
                    f'</div>', unsafe_allow_html=True)
                if llm:
                    if st.button("🔄 Regenerate Forecast", key="t3_regen_a"):
                        generate_forecast_summary.clear(); st.rerun()
            else:
                st.info("Insufficient data for forecasting.")

        with st3b:
            if not df_alarm.empty and not weekly_sum.empty:
                proj_val2, comp_df2, _ = calculate_smart_forecast(df_alarm, weekly_sum, w1_week, total_hc)
                fig_fc = forecast_bar_chart(comp_df2)
                st.plotly_chart(fig_fc, use_container_width=True)
                if llm:
                    with st.expander("💡 Explain", expanded=False):
                        fc_summary = f"projected: {proj_val2:.2f}/BC, current: {w1_metric:.3f}/BC, week {w1_week}"
                        st.markdown(explain_chart(llm, "Risk Forecast Chart", fc_summary, alarm_choice))
            else:
                st.info("Insufficient data.")

        with st3c:
            if llm:
                if st.button("📋 Generate Full Executive Briefing", type="primary", key="t3_exec"):
                    with st.spinner("Drafting briefing..."):
                        trend_vals = weekly_sum.tail(12)["per_bc"].round(2).tolist()
                        depot_stats = {}
                        for dep in depots_tuple:
                            d = df_alarm[df_alarm["depot_id"]==dep]
                            if not d.empty:
                                depot_stats[dep] = int(len(d[d["alarm_week"]==w1_week]))
                        proj_v, _, _ = calculate_smart_forecast(df_alarm, weekly_sum, w1_week, total_hc)
                        try:
                            brief = llm.predict(f"""You are Head of Fleet Operations. Write a high-level Executive Briefing.
Alarm: {alarm_choice} | 12-Week Trend: {trend_vals} | Latest Rate: {w1_metric:.2f} | Projected: {proj_v:.2f}
Depot Breakdown: {depot_stats}
Rules: state CRITICAL/ELEVATED/NORMAL, mention trend direction, identify primary depot, recommend 2 actions.""".strip())
                            st.session_state["t3_brief"] = brief
                        except Exception as e:
                            st.error(f"AI error: {e}")
                if "t3_brief" in st.session_state:
                    with st.expander("📄 Executive Briefing", expanded=True):
                        st.markdown(st.session_state["t3_brief"])
            else:
                st.info("Configure AI credentials in .env to enable executive briefing.")

    # ────────────────────────────────────────
    # TAB 4 — AI Deep Dive
    # ────────────────────────────────────────
    with t4:
        st4a, st4b = st.tabs(["📊 Summary + Tables", "🤖 GenAI Full Analysis"])

        with st4a:
            st.markdown(f"**W{w1_week} · {w1_year} — {len(w1_events):,} events | {alarm_choice}**")
            with st.expander("🔍 Data Context", expanded=False):
                if not w1_events.empty:
                    st.caption(f"Events: {len(w1_events):,} | Trips: {trips_unique['trip_id_norm'].nunique():,}")
                    if "depot_id" in w1_events.columns:
                        st.caption(f"Depot breakdown: {w1_events['depot_id'].value_counts().to_dict()}")
                    if "model" in w1_events.columns:
                        st.caption(f"Top models: {w1_events['model'].value_counts().head(5).to_dict()}")
            cols_show4 = ["Alarm Count","Alarm Trips","Alarms per Trip","Alarms per Hour","Total Duration (hr)"]
            df_dr4 = calc_metrics(w1_events, trips_unique, "driver_id")
            df_bs4 = calc_metrics(w1_events, trips_unique, "bus_no")
            df_sv4 = calc_metrics(w1_events, trips_unique, "svc_no")
            with st.expander("🧑‍✈️ Driver Table", expanded=True):
                if not df_dr4.empty:
                    st.dataframe(df_dr4.sort_values(["Alarms per Trip","Alarm Count"], ascending=False)[cols_show4].round(2), use_container_width=True)
            with st.expander("🚌 Bus Table", expanded=False):
                if not df_bs4.empty:
                    st.dataframe(df_bs4.sort_values(["Alarms per Trip","Alarm Count"], ascending=False)[cols_show4].round(2), use_container_width=True)
            with st.expander("🗺️ Service Table", expanded=False):
                if not df_sv4.empty:
                    st.dataframe(df_sv4.sort_values(["Alarms per Trip","Alarm Count"], ascending=False)[cols_show4].round(2), use_container_width=True)

        with st4b:
            if llm is None:
                st.markdown('<div class="alert-amber">⚠️ <strong>AI Not Configured</strong> — Add credentials to .env</div>', unsafe_allow_html=True)
            elif w1_events.empty:
                st.info("No events for selected week.")
            else:
                if st.button("🤖 Generate Full Analysis", type="primary", key="t4_ai"):
                    with st.spinner("Generating detailed analysis..."):
                        dr_p4 = calc_metrics(w1_events, trips_unique, "driver_id").reset_index()
                        bs_p4 = calc_metrics(w1_events, trips_unique, "bus_no").reset_index()
                        sv_p4 = calc_metrics(w1_events, trips_unique, "svc_no").reset_index()
                        def slim4(d, key, n=10):
                            if d.empty: return []
                            c = [key,"Alarm Count","Alarm Trips","Alarms per Trip","Total Duration (hr)"]
                            c = [x for x in c if x in d.columns]
                            return d.sort_values("Alarms per Trip", ascending=False).head(n)[c].round(3).to_dict("records")
                        flt_avg4 = (dr_p4["Alarm Count"].sum()/dr_p4["Alarm Trips"].sum()
                                   if not dr_p4.empty and dr_p4["Alarm Trips"].sum()>0 else 0)
                        result4 = generate_ai_deep_dive(
                            llm, alarm_choice, w1_metric, delta,
                            json.dumps(slim4(dr_p4,"driver_id")),
                            json.dumps(slim4(bs_p4,"bus_no")),
                            json.dumps(slim4(sv_p4,"svc_no")), flt_avg4)
                        st.session_state["t4_analysis"] = result4
                if "t4_analysis" in st.session_state:
                    with st.expander("📄 Full Analysis Report", expanded=True):
                        st.markdown(st.session_state["t4_analysis"])

    # ────────────────────────────────────────
    # TAB 5 — Consolidated Alert
    # ────────────────────────────────────────
    # Power Automate webhook (hardcoded)
    _PA_URL = (
        "https://defaultc88daf55f4aa4f79a143526402ab9a.23.environment.api.powerplatform.com:443"
        "/powerautomate/automations/direct/workflows/9c97356356884121962f780d0b6e5e89"
        "/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0"
        "&sig=SbOZpTqO5gOax1mz06R0dJBVvLt3mN2t2tp3lDGfiv4"
    )

    with t5:
        st.markdown("#### 📧 Consolidated Alert Email Builder")
        st.caption("Scans HA, HB, HC — generates email for all metrics trending > 3.0")
        report_date = st.columns([1,2])[0].date_input("Report date", value=datetime.now())

        if st.button("🔍 Scan All Metrics & Build Email", type="primary"):
            active_alerts = []; summaries = []
            max_proj = 0.0

            # Resolve headcount once for selected depots
            _t5_hc_sel = headcounts[headcounts["depot_id"].isin(depots_tuple)] if not headcounts.empty else pd.DataFrame()
            _t5_hc = float(_t5_hc_sel["headcount"].sum()) if not _t5_hc_sel.empty else 1.0
            if _t5_hc <= 0: _t5_hc = 1.0

            for a_type in ["HA","HB","HC"]:
                # Use only_completed=False so current incomplete week is included in forecast
                _df_f, _, _ = slice_by_filters(df_raw, weekly_all, depots_tuple, a_type, False, exclude_null)
                _df_snap = _df_f[_df_f["alarm_date"] <= pd.to_datetime(report_date)] if not _df_f.empty else _df_f
                if _df_snap.empty: continue

                # Rebuild per-week summary directly from the date-snapped events
                _wk_s = (
                    _df_snap.groupby(["alarm_year","alarm_week"], as_index=False)
                    .size().rename(columns={"size":"alarm_sum"})
                )
                _wk_s["per_bc"] = _wk_s["alarm_sum"] / _t5_hc
                _wk_s["start_of_week"] = pd.to_datetime(
                    _wk_s["alarm_year"].astype(int).astype(str) +
                    _wk_s["alarm_week"].astype(int).astype(str) + "1",
                    format="%G%V%w", errors="coerce")
                _wk_s = _wk_s.sort_values("start_of_week")
                if _wk_s.empty: continue

                _lwk  = int(_wk_s["alarm_week"].iloc[-1])
                _proj, _, _ = calculate_smart_forecast(_df_snap, _wk_s, _lwk, _t5_hc)

                # Also compute the raw current-week rate (alarms so far / HC)
                # Use whichever is higher — prevents blending logic from masking
                # a progressing week that is already above threshold
                _curr_wk_events = _df_snap[_df_snap["alarm_week"] == _lwk]
                _raw_curr_rate  = len(_curr_wk_events) / _t5_hc if _t5_hc > 0 else 0.0
                _effective_rate = max(_proj, _raw_curr_rate)

                if _effective_rate > 3.0:
                    max_proj = max(max_proj, _effective_rate)
                    curr  = _df_snap[_df_snap["alarm_week"] == _lwk]
                    top10 = (curr.groupby(["depot_id","svc_no","bus_no","model","driver_id"])
                             .size().sort_values(ascending=False).head(10).reset_index(name="count"))
                    rows_html = "".join(
                        f"<tr><td>{r.depot_id}</td><td>{r.svc_no}</td><td>{r.bus_no}</td>"
                        f"<td>{r.model}</td><td>{r.driver_id}</td><td>{r['count']}</td></tr>"
                        for _, r in top10.iterrows())
                    risk_label = "Medium Risk (>3.0)" if _effective_rate <= 5.0 else "Critical Risk (>5.0)"
                    # Show projected rate if forecast > raw, else show raw with note
                    _display_rate = _proj if _proj >= _raw_curr_rate else _raw_curr_rate
                    _rate_note = " (projected)" if _proj >= _raw_curr_rate else " (current week rate — week in progress)"
                    summaries.append(
                        f"Projected value for end of the week ({a_type}): {_display_rate:.2f}{_rate_note} [{risk_label}]"
                    )
                    active_alerts.append(f"""
<h3>Top Contributing Factors for {a_type}</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;width:100%;">
  <tr style="background-color:#f2f2f2;"><th>Depot</th><th>Service</th><th>Bus</th><th>Model</th><th>Driver</th><th>Total {a_type} Count</th></tr>
  {rows_html}
</table>""")

            if active_alerts:
                is_crit  = max_proj > 5.0
                h_color  = "#b91c1c" if is_crit else "#d97706"
                zone     = "Red" if is_crit else "Yellow"
                proj_line = "&nbsp;&nbsp;&nbsp;".join(summaries)
                tables_html = "".join(active_alerts)

                email_html = f"""<h2 style="color:{h_color};">Operational Alert: HA (Harsh Acceleration) &amp; HC (Harsh Cornering) Trending to {zone} Zone</h2>
<p><strong>{proj_line}</strong></p>

<h3 style="color:{h_color};">Analysis</h3>
<p>Below are the top contributing factors for the flagged alarms this week.</p>

{tables_html}

<h3 style="color:{h_color};">Action Required</h3>
<ul>
  <li>Depot Head to engage with driver and monitor for the week.</li>
  <li>Talk with SIS to check on data / sensor issues for the list flagged out.</li>
</ul>
<p>Best regards,<br/>Data Analytics Team</p>"""

                st.session_state.alert_email = email_html
                st.success(f"✅ Alert generated for {len(active_alerts)} metric(s).")
            else:
                st.success("✅ All systems Green. No alarms trending > 3.0.")
                st.session_state.pop("alert_email", None)

        if "alert_email" in st.session_state:
            recipients = st.text_input(
                "Recipients", "devi02@smrt.com.sg; fleet_ops@smrt.com.sg", key="t5_recipients"
            )
            col_send, col_dl = st.columns([1, 1])
            with col_send:
                if st.button("📨 Send Alert via Power Automate", type="primary"):
                    try:
                        import requests as _req
                        res = _req.post(
                            _PA_URL,
                            json={
                                "subject": "Consolidated Operational Alert",
                                "body":    st.session_state.alert_email,
                                "recipient": recipients,
                            },
                            timeout=10,
                        )
                        if res.status_code in [200, 202]:
                            st.success("✅ Sent successfully via Power Automate!")
                        else:
                            st.error(f"❌ Failed — HTTP {res.status_code}: {res.text[:200]}")
                    except Exception as _e:
                        st.error(f"❌ Error sending: {_e}")
            with col_dl:
                st.download_button(
                    "⬇️ Download HTML",
                    st.session_state.alert_email,
                    "alert_email.html",
                    "text/html",
                )
            st.markdown("**Preview:**")
            st.components.v1.html(st.session_state.alert_email, height=600, scrolling=True)

    # ─────────────────────────────────────────────────────────────────────
    # CHATBOT — AI Root Cause Analyst with full data context + memory
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Root Cause Analyst")

    # ── Session state init ──────────────────────────────────────────────
    if "chat_log"        not in st.session_state: st.session_state.chat_log        = []
    if "chat_steps"      not in st.session_state: st.session_state.chat_steps      = {}
    if "chat_context"    not in st.session_state: st.session_state.chat_context    = None
    if "context_alarm"   not in st.session_state: st.session_state.context_alarm   = None
    if "context_week"    not in st.session_state: st.session_state.context_week    = None
    for _k in ["morning_briefing_text","morning_briefing_key_done","t1_deep_dive","t2_systemic","t3_brief","t4_analysis"]:
        if _k not in st.session_state: st.session_state[_k] = None

    # ── Build rich data snapshot for AI context ─────────────────────────
    def _build_fleet_context(df_alarm, weekly_sum, alarm_choice, depots, w1_week):
        """Builds a compact JSON-serialisable dict with ~4 weeks of fleet data."""
        ctx = {"alarm_type": alarm_choice, "depots": list(depots), "current_week": int(w1_week)}

        # Weekly trend — last 8 weeks
        wk_rows = []
        for _, r in weekly_sum.tail(8).iterrows():
            wk_rows.append({
                "week": f"W{int(r['alarm_week'])}/{int(r.get('alarm_year', 0))}",
                "rate_per_bc": round(float(r["per_bc"]), 3),
                "total": int(r["total"]) if "total" in r else None,
                "status": "CRITICAL" if r["per_bc"] > 5 else ("ELEVATED" if r["per_bc"] > 3 else "NORMAL"),
            })
        ctx["weekly_trend"] = wk_rows

        # Current week detailed breakdown
        cw = df_alarm[df_alarm["alarm_week"] == w1_week]
        if not cw.empty:
            ctx["current_week_detail"] = {
                "total_alarms": int(len(cw)),
                "depot_breakdown": cw["depot_id"].value_counts().head(6).to_dict(),
                "top_drivers": cw["driver_id"].value_counts().head(10).to_dict(),
                "top_buses":   cw["bus_no"].value_counts().head(10).to_dict(),
                "top_services":cw["svc_no"].value_counts().head(8).to_dict(),
            }
            if "model" in cw.columns:
                ctx["current_week_detail"]["model_breakdown"] = cw["model"].value_counts().head(6).to_dict()

        # Per-week detail for last 4 weeks (for cross-week pattern analysis)
        per_week = {}
        for wk in weekly_sum["alarm_week"].tail(4).values:
            d = df_alarm[df_alarm["alarm_week"] == wk]
            if not d.empty:
                per_week[f"W{int(wk)}"] = {
                    "total": int(len(d)),
                    "top_drivers": d["driver_id"].value_counts().head(8).to_dict(),
                    "top_buses":   d["bus_no"].value_counts().head(8).to_dict(),
                    "depot_split": d["depot_id"].value_counts().to_dict(),
                }
        ctx["per_week_detail"] = per_week

        # Overall top offenders (all data loaded)
        ctx["all_time_top_drivers"]  = df_alarm["driver_id"].value_counts().head(15).to_dict()
        ctx["all_time_top_buses"]    = df_alarm["bus_no"].value_counts().head(15).to_dict()
        ctx["all_time_top_services"] = df_alarm["svc_no"].value_counts().head(10).to_dict()
        ctx["all_time_depot_split"]  = df_alarm["depot_id"].value_counts().to_dict()
        ctx["total_records_loaded"]  = int(len(df_alarm))

        # Fleet-wide rates (alarms per trip) for top entities
        trips_all = df_alarm.drop_duplicates("trip_id_norm")
        total_trips = df_alarm["trip_id_norm"].nunique()
        total_alarms = len(df_alarm)
        ctx["fleet_rate_per_trip"] = round(total_alarms / total_trips, 3) if total_trips > 0 else 0

        # Top driver rates (for instant reference)
        try:
            dr = df_alarm.groupby("driver_id").agg(a=("driver_id","count"), t=("trip_id_norm","nunique")).reset_index()
            dr["rate"] = (dr["a"] / dr["t"].clip(lower=1)).round(3)
            ctx["top_driver_rates"] = dr.sort_values("rate", ascending=False).head(10).set_index("driver_id")[["a","t","rate"]].rename(columns={"a":"alarms","t":"trips"}).to_dict("index")
        except Exception: pass

        # Top bus rates
        try:
            br = df_alarm.groupby("bus_no").agg(a=("bus_no","count"), t=("trip_id_norm","nunique")).reset_index()
            br["rate"] = (br["a"] / br["t"].clip(lower=1)).round(3)
            if "model" in df_alarm.columns:
                br["model"] = br["bus_no"].map(df_alarm.groupby("bus_no")["model"].agg(lambda x: x.mode()[0] if not x.empty else "?"))
            ctx["top_bus_rates"] = br.sort_values("rate", ascending=False).head(10).set_index("bus_no")[["a","t","rate"] + (["model"] if "model" in br.columns else [])].rename(columns={"a":"alarms","t":"trips"}).to_dict("index")
        except Exception: pass

        # Model breakdown
        if "model" in df_alarm.columns:
            try:
                mr = df_alarm.groupby("model").agg(a=("model","count"), t=("trip_id_norm","nunique")).reset_index()
                mr["rate"] = (mr["a"] / mr["t"].clip(lower=1)).round(3)
                ctx["model_rates"] = mr.sort_values("rate", ascending=False).set_index("model")[["a","t","rate"]].rename(columns={"a":"alarms","t":"trips"}).to_dict("index")
            except Exception: pass

        # Chronic offenders (present 3+ weeks)
        try:
            weeks_available = df_alarm["alarm_week"].nunique()
            threshold = max(2, weeks_available * 0.5)
            persist = (df_alarm.groupby(["driver_id","alarm_week"]).size()
                       .reset_index().groupby("driver_id")["alarm_week"].nunique())
            ctx["chronic_drivers"] = persist[persist >= threshold].sort_values(ascending=False).head(8).to_dict()
        except Exception: pass

        return ctx

    # ── Rebuild context if alarm/week changed ───────────────────────────
    ctx_stale = (
        st.session_state.context_alarm != alarm_choice or
        st.session_state.context_week  != w1_week or
        st.session_state.chat_context is None
    )
    if ctx_stale and not df_alarm.empty:
        st.session_state.chat_context  = _build_fleet_context(df_alarm, weekly_sum, alarm_choice, depots, w1_week)
        st.session_state.context_alarm = alarm_choice
        st.session_state.context_week  = w1_week

    ctx_data = st.session_state.chat_context or {}
    fleet_rate = 0
    try:
        wt = ctx_data.get("weekly_trend", [])
        if wt: fleet_rate = round(sum(w["rate_per_bc"] for w in wt) / len(wt), 3)
    except Exception: pass

    SYSTEM_PROMPT = f"""You are a world-class Fleet Operations Analyst embedded in a live bus telematics dashboard.
You think like a domain expert with 15 years of bus fleet safety experience.

=== DOMAIN KNOWLEDGE ===
Alarm types: HA=Harsh Acceleration | HB=Harsh Braking | HC=Harsh Cornering
Thresholds: >5.0 alarms/BC = CRITICAL (red) | 3.0–5.0 = ELEVATED (amber) | <3.0 = NORMAL (green)
BC = Bus Change (shift/trip unit). Fleet average this period: {fleet_rate:.3f}/BC

Root cause decision tree:
- High rate across 3+ buses → DRIVER-SPECIFIC → coaching
- High rate across 3+ drivers → VEHICLE-SPECIFIC → sensor/maintenance inspection  
- High rate on 1 service with many drivers → ROUTE-SPECIFIC → route audit
- High rate fleet-wide with no single driver/bus pattern → FLEET-WIDE → training programme
- High rate on <5 trips → INSUFFICIENT DATA → monitor 2 more weeks before action

Anomaly flags:
- Chronic offender: driver/bus present in top-10 for 3+ consecutive weeks
- New entrant spike: entity not in top-10 last week, suddenly top-5 this week
- Low-trip outlier: ≤5 trips but rate >2× fleet avg (statistically unreliable)
- Model systematic: entire bus model above fleet avg → fleet-wide maintenance issue

=== LIVE DATA SNAPSHOT ({alarm_choice} | {", ".join(depots)} | current week: W{w1_week}) ===
{json.dumps(ctx_data, default=float, indent=2)}

=== CONVERSATION RULES ===
1. ALWAYS cite specific numbers: driver IDs, bus numbers, exact alarm counts, exact rates.
2. ALWAYS compare to fleet average — raw counts mean nothing without context.
3. Multi-dimension analysis: combine driver + bus + service + depot + model evidence together.
4. Root cause classification is MANDATORY for any entity question — state verdict clearly.
5. Follow-up questions refer to previous answers — maintain full context.
6. When asked about a spike: analyse all dimensions (depot_breakdown, top_drivers, model_breakdown, week-on-week).
7. For "what should we do" questions: give a PRIORITY ACTION TABLE (High/Medium/Low).
8. For trend questions: state direction, magnitude, whether improving/worsening, and predicted next week.
9. Flag both chronic issues AND new acute spikes — they need different responses.
10. If a question needs data not in snapshot, say exactly what entity to look up and offer to explain.

=== OUTPUT FORMAT ===
Use markdown with headers. Bold all entity IDs and key numbers.
End operational answers with a priority action table.
Be direct, specific, and operational — no generic observations.
"""

    # ── AI answer function using direct openai SDK ───────────────────────
    def _ai_chat_answer(user_question: str, chat_log: list, llm) -> str:
        """Send full conversation + data context to AI and get a rich answer."""
        if llm is None:
            return None

        # Build messages: system + conversation history + new question
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include last 8 turns of conversation for memory
        history_turns = chat_log[-16:]  # 8 user + 8 assistant
        for msg in history_turns:
            role    = msg["role"]
            content = msg["content"]
            # Trim very long previous answers to save tokens
            if role == "assistant" and len(content) > 800:
                content = content[:800] + "\n...[truncated for context]"
            messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_question})

        # Use LangChain if available, else direct SDK wrapper
        try:
            if LANGCHAIN_OK and hasattr(llm, "invoke"):
                # LangChain path — use predict on joined message
                full_prompt = "\n\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages[1:]
                )
                return llm.predict(SYSTEM_PROMPT + "\n\nCONVERSATION:\n" + full_prompt)
            elif hasattr(llm, "_client"):
                # Direct SDK path — supports proper multi-turn
                response = llm._client.chat.completions.create(
                    model=llm._deployment,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1800,
                )
                return response.choices[0].message.content
            else:
                return llm.predict(user_question)
        except Exception as e:
            return f"AI error: {e}"

    # ── Suggested questions ─────────────────────────────────────────────
    SUGGESTED_QS = [
        f"Why did {alarm_choice} spike in W{w1_week}?",
        "Who are the top 10 drivers this week and which depot are they from?",
        "Which buses are repeat offenders across the last 4 weeks?",
        "Compare depot performance — which has the worst rate?",
        f"What is the root cause for the highest offending driver?",
        "Is the fleet trend improving or worsening? Predict next week.",
        "Which service routes have the most alarms and why?",
    ]

    # ── Badge & intro ────────────────────────────────────────────────────
    if llm:
        mode_label = "ReAct Agent" if (LANGCHAIN_OK and react_agent) else "AI Analyst (Direct)"
        st.markdown(
            f'<span class="badge-blue">✅ {mode_label} — Full data context loaded '
            f'({st.session_state.chat_context.get("total_records_loaded", 0):,} records, '
            f'last {len(st.session_state.chat_context.get("weekly_trend", []))} weeks)</span>',
            unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-amber">ℹ️ Rule-based mode — configure .env for full AI analyst</span>',
                    unsafe_allow_html=True)
    st.caption("Ask anything: driver root cause, spike analysis, depot comparison, trend forecast, cross-week patterns.")

    # ── Suggested question buttons ───────────────────────────────────────
    with st.expander("💡 Suggested questions — click to ask instantly", expanded=False):
        cols = st.columns(2)
        for i, sq in enumerate(SUGGESTED_QS):
            if cols[i % 2].button(sq, key=f"sugq_{i}"):
                # Auto-submit immediately — don't just pre-fill, process now
                st.session_state.chat_log.append({"role": "user", "content": sq})
                with st.spinner("Analyst thinking…"):
                    answer = None
                    if react_agent and LANGCHAIN_OK:
                        try:
                            result = react_agent.invoke({
                                "input": sq,
                                "alarm_choice": alarm_choice,
                                "depots": ", ".join(depots),
                            })
                            raw_sq = result.get("output", "")
                            steps  = result.get("intermediate_steps", [])
                            _bad = ("agent stopped", "iteration limit", "time limit",
                                    "format error", "invalid format", "parsing error")
                            if raw_sq and not any(b in raw_sq.lower() for b in _bad):
                                answer = raw_sq
                                st.session_state.chat_steps[len(st.session_state.chat_log)] = steps
                        except Exception:
                            answer = None
                    if not answer and llm:
                        answer = _ai_chat_answer(sq, st.session_state.chat_log[:-1], llm)
                    if not answer:
                        rb_ans, rb_tbl = smart_rule_analyst(sq, alarm_choice, df_alarm, weekly_sum, w1_week, depots)
                        answer = rb_ans
                        if isinstance(rb_tbl, pd.DataFrame) and not rb_tbl.empty:
                            answer += "\n\n" + rb_tbl.to_markdown(index=False)
                st.session_state.chat_log.append({"role": "assistant", "content": answer})
                st.rerun()

    # ── Render conversation history ──────────────────────────────────────
    for i, msg in enumerate(st.session_state.chat_log):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">🧑 <strong>You:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="chat-bot">🤖 <strong>Fleet Analyst:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True)
            steps = st.session_state.chat_steps.get(i, [])
            if steps:
                with st.expander(f"🔍 Agent reasoning ({len(steps)} steps)", expanded=False):
                    for action, obs in steps:
                        st.markdown(
                            f'<div class="agent-tool-call">→ Tool: {action.tool} | '
                            f'Input: {str(action.tool_input)[:80]}</div>',
                            unsafe_allow_html=True)
                        st.caption(str(obs)[:400] + "…" if len(str(obs)) > 400 else str(obs))

    # ── Input area ───────────────────────────────────────────────────────
    user_q = st.text_area(
        "Ask the analyst:",
        height=80,
        placeholder="e.g.  Why spike in W9?  |  Top 10 drivers this week  |  Root cause for driver 21665  |  Compare depots W8 vs W9",
        key="main_chat_input",
    )
    col_ask, col_clr, col_ctx = st.columns([1, 1, 4])
    ask_btn   = col_ask.button("▶ Ask",    type="primary", key="main_ask")
    clear_btn = col_clr.button("🗑 Clear",               key="main_clear")
    with col_ctx:
        if st.session_state.chat_context:
            wks = len(st.session_state.chat_context.get("weekly_trend", []))
            recs = st.session_state.chat_context.get("total_records_loaded", 0)
            st.caption(f"🧠 Context: {recs:,} records | {wks} weeks | {alarm_choice} | {', '.join(depots)}")

    if clear_btn:
        st.session_state.chat_log   = []
        st.session_state.chat_steps = {}
        st.rerun()

    if ask_btn and user_q.strip():
        q = user_q.strip()
        st.session_state.chat_log.append({"role": "user", "content": q})

        with st.spinner("Analyst thinking…"):
            answer = None

            # ── Path 1: LangChain ReAct Agent (best) ──────────────────
            if react_agent and LANGCHAIN_OK:
                try:
                    result = react_agent.invoke({
                        "input": q,
                        "alarm_choice": alarm_choice,
                        "depots": ", ".join(depots),
                    })
                    raw_answer = result.get("output", "")
                    steps  = result.get("intermediate_steps", [])
                    # Discard agent output if it's a failure message or empty
                    _bad = ("agent stopped", "iteration limit", "time limit",
                            "format error", "invalid format", "parsing error")
                    if raw_answer and not any(b in raw_answer.lower() for b in _bad):
                        answer = raw_answer
                        msg_idx = len(st.session_state.chat_log)
                        st.session_state.chat_steps[msg_idx] = steps
                    # else: fall through to Path 2
                except Exception:
                    answer = None  # fall through

            # ── Path 2: Direct AI with full context + conversation memory ──
            if not answer and llm:
                answer = _ai_chat_answer(q, st.session_state.chat_log[:-1], llm)

            # ── Path 3: Rule-based fallback ────────────────────────────
            if not answer:
                rb_ans, rb_tbl = smart_rule_analyst(q, alarm_choice, df_alarm, weekly_sum, w1_week, depots)
                answer = rb_ans
                if isinstance(rb_tbl, pd.DataFrame) and not rb_tbl.empty:
                    answer += "\n\n" + rb_tbl.to_markdown(index=False)

        st.session_state.chat_log.append({"role": "assistant", "content": answer})
        st.rerun()


if __name__ == "__main__":
    main()
