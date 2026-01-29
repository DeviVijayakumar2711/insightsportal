import os
import json
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Try to import Azure OpenAI, handle failure gracefully
try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    AzureChatOpenAI = None

# Load local .env file (for local development only)
load_dotenv()

# ==========================================
# 🔧 CONFIGURATION (POWER AUTOMATE LINK)
# ==========================================
# I have embedded your specific link here so it works instantly on Azure
HARDCODED_PA_URL = "https://defaultc88daf55f4aa4f79a143526402ab9a.23.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/9c97356356884121962f780d0b6e5e89/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=SbOZpTqO5gOax1mz06R0dJBVvLt3mN2t2tp3lDGfiv4"

# --- 1. CONFIGURATION HELPER (Azure + Local) ---
def get_config(key, default=""):
    """
    Retrieves configuration from Azure Environment Variables first.
    """
    if key in os.environ:
        return os.environ[key]
    
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        try:
            return st.secrets.get(key, default)
        except Exception:
            pass
            
    return default

# --- 2. PAGE CONFIG & CSS ---
st.set_page_config(page_title="Fleet Operations Center", page_icon="🚍", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F4F6F9; }
    
    .metric-card {
        background-color: white; padding: 25px; border-radius: 10px;
        border: 1px solid #E1E4E8; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: white; border-radius: 6px;
        border: 1px solid #E5E7EB; padding: 0 25px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF; border-color: #3B82F6; color: #1D4ED8;
    }

    .alert-banner {
        padding: 20px; border-radius: 8px; font-weight: 600; margin-bottom: 15px;
    }
    .alert-critical { background-color: #FEF2F2; border-left: 5px solid #EF4444; color: #991B1B; }
    .alert-elevated { background-color: #FEF9E7; border-left: 5px solid #F1C40F; color: #7D6608; }
    .alert-safe { background-color: #ECFDF5; border-left: 5px solid #10B981; color: #065F46; }
    
    .stChatInput { position: fixed; bottom: 20px; width: 70%; left: 15%; z-index: 999; }
</style>
""", unsafe_allow_html=True)

# --- 3. CONSTANTS & PROMPTS ---
ALARM_MAP = {
    "HA": {"long": "Harsh Acceleration", "short": "HA"},
    "HB": {"long": "Harsh Braking",      "short": "HB"},
    "HC": {"long": "Harsh Cornering",    "short": "HC"},
}

PROMPT_EXEC_BRIEF = """
You are the Head of Fleet Operations. Write a high-level **Executive Briefing** (Max 4 sentences).
DATA CONTEXT:
* **12-Week Trend:** {trend_vals}
* **Latest Full Week Rate:** {last_full_val}
* **Depot Breakdown:** {depot_stats}

INSTRUCTIONS:
1. **Status Update:**
   - If rate > 5.0, state "CRITICAL ATTENTION REQUIRED".
   - If rate > 3.0, state "ELEVATED STATUS".
2. **Context:** Mention if the trend is improving or requiring monitoring.
3. **Focus Area:** Identify which Depot is the primary contributor.
4. **Tone:** Professional, constructive, and direct. Avoid alarmist language.
"""

PROMPT_WEEKLY_INSIGHT = """
You are an expert Bus Operations Analyst. Alarm type: {alarm_code}.
Using ONLY this JSON summary for a SINGLE week, produce insights.

DATA:
{payload}

Write markdown with EXACT sections:
### Executive Summary
### Operational Patterns
- **Key Concentrations:** Identify cross-category patterns.
- **Performance Context:** Compare current performance vs fleet average.
### Recommended Actions
Return a 3-row Markdown Table: | Priority | Recommended Action | Data-Driven Rationale |
(Use phrases like "Monitor depot", "Review settings", or "Engage driver" instead of "Audit" or "Investigate").
"""

PROMPT_4WEEK_DEEP = """
You are a world-class Bus Operations Analyst.
Your task is to analyze the root cause of the 4-week trend for '{alarm_code}'.

{context_str}

**DATA:**
{data_json}

**YOUR MISSION (Write in Markdown):**
1. **Trend Overview:** Summary of the 4-week fleet performance.
2. **Contributing Factors:** Identify specific Drivers, Buses, or Services influencing the trend.
3. **Operational Nexus:** Find connections (Driver -> Bus -> Service).
4. **Summary:** Brief, prioritized next steps.
"""

PROMPT_ALERT_EMAIL = """
You are drafting a professional operational alert email in **HTML format**.
SCENARIO:
* **Metric:** {alarm}
* **Projected Value:** {projection}
* **Trend Context:** {trend_context}
* **Model Insight:** {model_context}
* **Top Contributors Data:** {offender_data}

INSTRUCTIONS:
Return **ONLY** the HTML code.
Follow this exact structure:
<h2 style="color: #b91c1c;">Operational Update: {alarm} Metric Status</h2>
<p>
  <strong>Projected Value:</strong> {projection}<br>
  <strong>Context:</strong> {trend_context}
</p>
<h3 style="color: #b91c1c;">Analysis</h3>
<p>[Write 2-3 sentences analyzing the 'Top Contributors Data'. Explicitly mention the Model Distribution ({model_context}) to highlight if EVs or specific models are the main cause. Use neutral language.]</p>
<h3>Top 10 Contributors (High Frequency)</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <tr style="background-color: #f2f2f2;"><th>Depot</th><th>Service</th><th>Bus</th><th>Model</th><th>Driver</th><th>Count</th></tr>
  [Generate rows from offender_data]
</table>
<h3 style="color: #b91c1c;">Suggested Actions</h3>
<ul>
  <li>Review root cause of alarm frequency on top services.</li>
  <li>Engage with the relevant Depot to discuss driver support.</li>
  <li>Continue monitoring {alarm} metric.</li>
</ul>
<p>Best regards,<br>Data Analytics Team</p>
"""

# --- 4. AI SETUP ---
@st.cache_resource
def initialize_llm():
    try:
        if AzureChatOpenAI is None: return None
        endpoint = get_config("AZURE_ENDPOINT")
        api_key = get_config("OPENAI_API_KEY")
        deployment = get_config("AZURE_DEPLOYMENT")
        
        if not endpoint or not api_key: return None
            
        return AzureChatOpenAI(
            azure_endpoint=endpoint.rstrip("/"),
            openai_api_key=api_key,
            azure_deployment=deployment,
            api_version=get_config("AZURE_API_VERSION", "2024-02-15-preview"),
            temperature=0.2
        )
    except Exception as e:
        print(f"LLM Init Error: {e}")
        return None

# --- 5. DATA LOGIC ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

@st.cache_data
def load_data():
    """Smart Data Loader: Handles Azure URL or Local File with Absolute Paths"""
    path_tele = get_config("TELEMATICS_URL", "telematics_new_data_2207.csv")
    path_head = get_config("HEADCOUNTS_URL", "depot_headcounts.csv")
    path_excl = get_config("EXCLUSIONS_URL", "vehicle_exclusions.csv")
    
    # Absolute path fix for local files in Azure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_model_url = get_config("MODEL_URL", "").strip()
    path_model_local = os.path.join(base_dir, "model.csv")

    diagnostics = {"model_status": "Not Attempted", "model_cols": [], "merge_count": 0}

    def smart_read(path_val, is_required=False):
        if not path_val: return pd.DataFrame()
        path_str = str(path_val).strip()
        
        # A) URL Read (Azure Blob)
        if path_str.lower().startswith("http"):
            try:
                return pd.read_csv(path_str)
            except Exception as e:
                if is_required:
                    if "403" in str(e):
                        st.error(f"❌ **Azure Access Denied (403)**: SAS Token likely expired for `{path_str}`.")
                    else:
                        st.error(f"❌ **Azure Connection Error**: {e}")
                return pd.DataFrame()
        
        # B) Local File Read
        if not os.path.isabs(path_str): path_str = os.path.join(base_dir, path_str)
        if os.path.exists(path_str):
            return pd.read_csv(path_str)
        else:
            if is_required: st.error(f"❌ File Not Found: {path_str}")
            return pd.DataFrame()

    df = smart_read(path_tele, is_required=True)
    headcounts = smart_read(path_head)
    exclusions = smart_read(path_excl)

    if df.empty: return pd.DataFrame(), pd.DataFrame(), diagnostics

    # Pre-process Telematics
    df.columns = [c.lower().strip() for c in df.columns]
    for c in ['bus_no', 'driver_id', 'alarm_type', 'depot_id', 'svc_no']:
        if c in df.columns: 
            df[c] = df[c].astype(str).str.strip().str.upper()
            df[c] = df[c].replace(['NAN', 'NULL', 'NONE', ''], None)
            
            # --- FIX: Remove Decimals from Driver ID ---
            if c == 'driver_id':
                df[c] = df[c].astype(str).str.replace(r'\.0$', '', regex=True)

    # --- MODEL LOADING & EV LOGIC ---
    bus_models = pd.DataFrame()
    
    # 1. Try URL
    if path_model_url:
        try:
            bus_models = pd.read_csv(path_model_url)
            diagnostics['model_status'] = "Loaded from URL"
        except Exception as e:
            diagnostics['model_status'] = f"URL Failed: {str(e)}"
    
    # 2. Try Local if URL failed or empty
    if bus_models.empty and os.path.exists(path_model_local):
        try:
            bus_models = pd.read_csv(path_model_local)
            diagnostics['model_status'] = "Loaded from Local File"
        except Exception as e:
            diagnostics['model_status'] = f"Local Read Failed: {str(e)}"
    
    # 3. Merge Logic
    if not bus_models.empty:
        bus_models.columns = [c.lower().strip().replace(' ', '_').replace('.', '') for c in bus_models.columns]
        diagnostics['model_cols'] = list(bus_models.columns)
        
        possible_ids = ['bus_no', 'bus_number', 'vehicle_no', 'vehicle_id']
        found_id = next((c for c in bus_models.columns if c in possible_ids), None)
        
        if found_id:
            bus_models.rename(columns={found_id: 'bus_no'}, inplace=True)
            bus_models['bus_no'] = bus_models['bus_no'].astype(str).str.strip().str.upper()
            
            if 'model' in bus_models.columns:
                df = df.merge(bus_models[['bus_no', 'model']], on='bus_no', how='left')
                df['model'] = df['model'].fillna('Unknown')
                
                # --- NEW: Tag EV Buses ---
                def tag_ev(model_name):
                    if not isinstance(model_name, str): return model_name
                    upper_m = model_name.upper()
                    if 'BYD' in upper_m or 'ZHONGTONG' in upper_m:
                        if '(EV)' not in model_name:
                            return f"{model_name} (EV)"
                    return model_name
                
                df['model'] = df['model'].apply(tag_ev)
                
                diagnostics['merge_count'] = df[df['model'] != 'Unknown'].shape[0]
                diagnostics['model_status'] += " & Merged Successfully"
            else:
                diagnostics['model_status'] += " (Missing 'model' column)"
        else:
            diagnostics['model_status'] += " (No ID column found)"
            
    if 'model' not in df.columns: df['model'] = 'Unknown'

    if 'depot_id' in headcounts.columns:
        headcounts['depot_id'] = headcounts['depot_id'].astype(str).str.strip().str.upper()
    else:
        headcounts = pd.DataFrame(columns=['depot_id', 'headcount'])

    if not exclusions.empty and 'bus_no' in exclusions.columns:
        excl_list = set(exclusions['bus_no'].astype(str).str.strip().str.upper())
        df = df[~df['bus_no'].isin(excl_list)]

    df['date'] = pd.to_datetime(df.get('alarm_calendar_date'), dayfirst=True, errors='coerce')
    df['year'] = df['date'].dt.isocalendar().year
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek 
    
    return df, headcounts, diagnostics

def process_metrics(df, headcounts, alarm_type, depots, exclude_null_driver, only_completed):
    mask = (df['alarm_type'] == alarm_type) & (df['depot_id'].isin(depots)) & (df['depot_id'].notna())
    if exclude_null_driver:
        mask = mask & (df['driver_id'].notna()) & (df['driver_id'] != '0') & (df['driver_id'] != 'NAN')
    
    df_filtered = df[mask].copy()
    if df_filtered.empty: return df_filtered, pd.DataFrame(), {}, {}, 0

    weekly = df_filtered.groupby(['year', 'week']).size().reset_index(name='count')
    hc_mask = headcounts['depot_id'].isin(depots)
    total_hc = headcounts[hc_mask]['headcount'].sum()
    weekly['per_bc'] = weekly['count'] / max(1, total_hc)
    
    weekly['start_date'] = weekly.apply(lambda x: datetime.strptime(f'{int(x.year)}-W{int(x.week)}-1', "%Y-W%W-%w"), axis=1)
    weekly = weekly.sort_values('start_date')
    weekly['label'] = "W" + weekly['week'].astype(str)
    
    latest_wk = weekly.iloc[-1]['week'] if not weekly.empty else 0
    df_curr = df_filtered[df_filtered['week'] == latest_wk]
    
    if 'model' not in df_curr.columns: df_curr['model'] = 'Unknown'

    weekly_payload = {
        "week_number": int(latest_wk),
        "total_alarms": int(len(df_curr)),
        "total_headcount": int(total_hc),
        "depot_breakdown": df_curr['depot_id'].value_counts().to_dict(),
        "model_breakdown": df_curr['model'].value_counts().to_dict(),
        "top_contributors": df_curr.groupby(['depot_id', 'svc_no', 'bus_no', 'model', 'driver_id']).size().sort_values(ascending=False).head(15).reset_index(name='count').to_dict(orient='records')
    }
    
    df_4wk = df_filtered[df_filtered['week'] >= (latest_wk - 3)]
    wk_4_payload = {
        "drivers": df_4wk.groupby(['driver_id', 'week']).size().unstack(fill_value=0).sum(axis=1).to_dict(),
        "services": df_4wk.groupby(['svc_no', 'week']).size().unstack(fill_value=0).sum(axis=1).to_dict(),
        "buses": df_4wk.groupby(['bus_no', 'week']).size().unstack(fill_value=0).sum(axis=1).to_dict()
    }
    
    return df_filtered, weekly, weekly_payload, wk_4_payload, latest_wk

# --- 6. VISUALIZATIONS ---
def plot_trend_old_style(weekly_df, alarm_name):
    fig = go.Figure()
    if weekly_df.empty: return fig
    plot_df = weekly_df.copy()
    max_y = max(6, plot_df['per_bc'].max() * 1.1)
    
    fig.add_hrect(y0=0, y1=3, line_width=0, fillcolor="rgba(46, 204, 113, 0.15)", layer="below")
    fig.add_hrect(y0=3, y1=5, line_width=0, fillcolor="rgba(241, 196, 15, 0.15)", layer="below")
    fig.add_hrect(y0=5, y1=max_y, line_width=0, fillcolor="rgba(231, 76, 60, 0.15)", layer="below")
    
    fig.add_trace(go.Scatter(
        x=plot_df['label'], y=plot_df['per_bc'],
        mode="lines+markers+text", name=alarm_name,
        line=dict(color="#0072C6", width=3),
        marker=dict(size=8, color="#3498DB"),
        text=plot_df['per_bc'].round(2), textposition="top center",
        textfont=dict(color="black", size=12)
    ))
    
    fig.add_hline(y=3.0, line_width=1.5, line_dash="dash", line_color="green")
    fig.add_hline(y=5.0, line_width=1.5, line_dash="dash", line_color="red")
    
    fig.update_layout(title=f"12-Week Trend ({alarm_name})", yaxis_title="Avg per Bus Captain", yaxis_range=[0, max_y], showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=350)
    return fig

def plot_single_forecast_bar(comp_df):
    fig = go.Figure()
    colors = ['#BDC3C7' if r['status'] == 'Completed' else ('#E74C3C' if r['display_rate']>5 else '#2ECC71') for _, r in comp_df.iterrows()]
    fig.add_trace(go.Bar(x=comp_df['week_label'], y=comp_df['display_rate'], marker_color=colors, text=comp_df['display_rate'].round(2), textposition='auto'))
    fig.add_hline(y=5.0, line_dash="dash", line_color="red")
    fig.update_layout(title="Risk Timeline", margin=dict(l=20, r=20, t=40, b=20), height=350)
    return fig

# --- FIXED: Robust Power Automate Function ---
def trigger_power_automate(html_body, prediction, recipients_str):
    """
    Triggers a Power Automate flow via HTTP POST.
    Checks hardcoded variable first, then env variable.
    """
    try:
        # Check Hardcoded first, then Env
        url = HARDCODED_PA_URL
        if not url:
            url = get_config("POWER_AUTOMATE_FLOW_URL")
        
        if not url:
            st.error("❌ Configuration Error: URL missing. Please set POWER_AUTOMATE_URL in the code or Azure settings.")
            return False
        
        # Prepare Payload
        payload = {
            "subject": f"⚠️ ALERT: Rate {prediction:.2f}", 
            "body": html_body, 
            "recipient": recipients_str
        }
        
        # Send Request with explicit timeout
        response = requests.post(url, json=payload, timeout=10)
        
        # Validate Status Code
        if response.status_code in [200, 202]:
            return True
        else:
            st.error(f"❌ Transmission Failed: Server returned {response.status_code}. Details: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"❌ Connection Error: {str(e)}")
        return False

def calculate_smart_forecast(df_filtered, weekly, current_week, total_hc):
    latest_row = weekly.iloc[-1]
    current_data = df_filtered[df_filtered['week'] == current_week]
    
    if not current_data.empty:
        last_data_dt = current_data['date'].max()
        max_day_idx = last_data_dt.weekday()
        day_name = last_data_dt.strftime('%A')
    else:
        max_day_idx = 0; day_name = "Monday"

    past_weeks_ids = weekly[weekly['week'] < current_week]['week'].unique()[-12:]
    profile_ratios, weights = [], []
    
    for i, w in enumerate(past_weeks_ids):
        wk_data = df_filtered[df_filtered['week'] == w]
        if len(wk_data) < 10: continue
        count_so_far = len(wk_data[wk_data['day_of_week'] <= max_day_idx])
        profile_ratios.append(count_so_far / len(wk_data))
        weights.append(i + 1)
        
    completion_rate = np.average(profile_ratios, weights=weights) if profile_ratios else (max_day_idx + 1) / 7.0
    completion_rate = max(0.05, completion_rate)
    
    momentum_factor = 1.0
    if len(weekly) >= 5:
        recent_rates = weekly[weekly['week'] < current_week].tail(4)['per_bc'].values
        if len(recent_rates) > 1:
            slope, _ = np.polyfit(np.arange(len(recent_rates)), recent_rates, 1)
            if slope > 0: momentum_factor = 1 + (slope * 0.5)

    comparison_rows = []
    display_weeks = weekly['week'].unique()[-5:]
    w_prof, w_avg = 0.5, 0.5
    
    for w in display_weeks:
        wk_label = f"W{w}"
        is_curr = (w == current_week)
        wk_slice = df_filtered[df_filtered['week'] == w]
        hc = max(1, total_hc)
        
        if is_curr:
            count_so_far = len(wk_slice[wk_slice['day_of_week'] <= max_day_idx])
            raw_proj = count_so_far / completion_rate
            adj_proj = raw_proj * momentum_factor
            avg_4wk = weekly[weekly['week'] < w].tail(4)['per_bc'].mean()
            if pd.isna(avg_4wk): avg_4wk = count_so_far/hc
            
            w_prof = min(1.0, completion_rate + 0.15)
            w_avg = 1.0 - w_prof
            final = ((adj_proj/hc) * w_prof) + (avg_4wk * w_avg)
            
            comparison_rows.append({"week_label": f"W{w} (Fcst)", "status": "In Progress (Projected)", "display_rate": final})
        else:
            comparison_rows.append({"week_label": wk_label, "status": "Completed", "display_rate": len(wk_slice)/hc})
            
    return comparison_rows[-1]['display_rate'], pd.DataFrame(comparison_rows), {"day": day_name, "completion_rate": completion_rate, "weight_profile": w_prof, "weight_avg": w_avg}

# --- 7. MAIN APP ---
def main():
    llm = initialize_llm()
    
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        if st.button("🔄 Reset Cache & Reload Data"):
            st.cache_data.clear()
            st.rerun()
            
        alarm = st.selectbox("Alarm Type", list(ALARM_MAP.keys()))
        df, headcounts, diagnostics = load_data()
        
        with st.expander("🛠️ Diagnostics (Model Data)"):
            st.write(f"Status: {diagnostics.get('model_status')}")
            st.write(f"Matched: {diagnostics.get('merge_count')} records")
            
        if df.empty:
            st.warning("⚠️ Data missing.")
            st.stop()
            
        depot_opts = sorted(headcounts['depot_id'].unique())
        default_depots = [d for d in depot_opts if d in ["WDLAND", "KRANJI", "JURONG"]]
        if not default_depots: default_depots = depot_opts[:2]
        
        depots = st.multiselect("Depot(s)", depot_opts, default=default_depots)
        only_completed = st.checkbox("Only completed weeks", value=True)
        exclude_null_driver = st.checkbox("Exclude null drivers", value=True)

    if not depots: st.stop()
    
    # Global Filter (Full Data)
    df_filtered, weekly, wk_payload, wk4_payload, latest_wk_num = process_metrics(
        df, headcounts, alarm, depots, exclude_null_driver, only_completed
    )
    if weekly.empty: st.warning("No data found."); st.stop()
    
    hc_mask = headcounts['depot_id'].isin(depots)
    total_hc = headcounts[hc_mask]['headcount'].sum()
    
    # Calculate Forecast for Tab 3 (Default View)
    final_projected_val, comp_df, explainer = calculate_smart_forecast(df_filtered, weekly, latest_wk_num, total_hc)

    st.markdown(f"## {ALARM_MAP[alarm]['long']} Intelligence Hub")
    
    c1, c2 = st.columns([2, 1])
    with c1: st.plotly_chart(plot_trend_old_style(weekly.tail(12), alarm), use_container_width=True)
    with c2:
        st.markdown("### 📋 Executive Brief")
        if llm:
            if "exec_brief" not in st.session_state:
                with st.spinner("Analyzing..."):
                    st.session_state.exec_brief = llm.predict(PROMPT_EXEC_BRIEF.format(trend_vals=weekly.tail(12)['per_bc'].tolist(), last_full_val=weekly.iloc[-2]['per_bc'], depot_stats=wk_payload['depot_breakdown']))
            st.write(st.session_state.exec_brief)
        else: st.info("AI Not Connected")

    t1, t2, t3 = st.tabs(["📊 Weekly Deep Dive", "🧠 4-Week Pattern", "🔮 Risk Forecast"])
    
    with t1:
        c1, c2 = st.columns([1, 2])
        c1.metric("Avg per BC", f"{weekly.iloc[-1]['per_bc']:.2f}"); c1.metric("Total Events", int(weekly.iloc[-1]['count']))
        with c2:
            if llm and st.button("Generate Deep Dive"):
                with st.spinner("Analyzing..."): st.markdown(llm.predict(PROMPT_WEEKLY_INSIGHT.format(alarm_code=alarm, payload=json.dumps(wk_payload, cls=NpEncoder))))

    with t2:
        c1, c2 = st.columns([1, 2])
        avg_4wk = weekly.tail(4)['per_bc'].mean()
        c1.metric("4-Week Baseline", f"{avg_4wk:.2f}")
        with c2:
            if llm and st.button("Run Systemic Scan"):
                with st.spinner("Scanning..."): st.markdown(llm.predict(PROMPT_4WEEK_DEEP.format(alarm_code=alarm, context_str=f"Rate: {weekly.iloc[-1]['per_bc']}", data_json=json.dumps(wk4_payload, cls=NpEncoder))))

    with t3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        # --- DATE SELECTOR ---
        col_date, col_status = st.columns([1, 2])
        with col_date:
            max_dt = df_filtered['date'].max()
            if pd.isna(max_dt): max_dt = datetime.now()
            report_date = st.date_input("Drafting Date (Snapshot)", value=max_dt, max_value=datetime.now())
        
        # Snapshot Logic
        snap_date_ts = pd.to_datetime(report_date)
        df_snap = df_filtered[df_filtered['date'] <= snap_date_ts]
        
        if not df_snap.empty:
            _, weekly_snap, wk_payload_snap, _, latest_wk_snap = process_metrics(
                df, headcounts, alarm, depots, exclude_null_driver, only_completed
            )
            weekly_snap = weekly_snap[weekly_snap['week'] <= latest_wk_snap]
            
            snap_proj_val, snap_comp_df, snap_explainer = calculate_smart_forecast(df_snap, weekly_snap, latest_wk_snap, total_hc)
            
            # --- MODEL DISTRIBUTION ANALYSIS ---
            model_context_str = "Model breakdown not available."
            if 'model' in df_snap.columns:
                m_counts = df_snap['model'].value_counts(normalize=True) * 100
                top_models_str = ", ".join([f"{k}: {v:.1f}%" for k, v in m_counts.head(3).items()])
                model_context_str = f"Top Contributing Models: {top_models_str}."
            
            with col_status:
                st.metric("Projected Risk (As of Snapshot)", f"{snap_proj_val:.2f}", delta=f"Week {latest_wk_snap} Projection", delta_color="inverse")
        
            st.plotly_chart(plot_single_forecast_bar(snap_comp_df), use_container_width=True)
            st.dataframe(snap_comp_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📢 Escalation Protocol")
            c_email_btn, c_email_prev = st.columns([1, 2])
            
            with c_email_btn:
                if snap_proj_val > 5.0:
                    st.error("🚨 Risk Level Critical: Alert Generation Enabled.")
                    recipients = st.text_input("Recipients", "devi02@smrt.com.sg; fleet_ops@smrt.com.sg")
                    
                    if st.button("📝 Draft Alert Email"):
                         if llm:
                            avg_4 = weekly_snap.tail(4)['per_bc'].mean() if len(weekly_snap) > 1 else snap_proj_val
                            diff = snap_proj_val - avg_4
                            trend_txt = f"{((snap_proj_val - avg_4)/avg_4)*100:.1f}% vs 4-wk avg"
                            
                            # EXTENDED TO TOP 10 OFFENDERS
                            offender_list = wk_payload_snap['top_contributors'][:10]
                            offender_str = "\n".join([f"<tr><td>{i['depot_id']}</td><td>{i['svc_no']}</td><td>{i['bus_no']}</td><td>{i['model']}</td><td>{i['driver_id']}</td><td>{i['count']}</td></tr>" for i in offender_list])
                            
                            with st.spinner("Drafting email..."):
                                st.session_state.email_html_draft = llm.predict(PROMPT_ALERT_EMAIL.format(
                                    alarm=alarm, projection=f"{snap_proj_val:.2f}", 
                                    trend_context=trend_txt, model_context=model_context_str, offender_data=offender_str
                                ))
                         else: st.error("AI not connected")

                    if "email_html_draft" in st.session_state:
                        if st.button("📨 Send via Power Automate"):
                            if trigger_power_automate(st.session_state.email_html_draft, snap_proj_val, recipients):
                                st.success("Alert Sent Successfully")
                            else: st.error("Transmission Failed")
                else: st.success(f"Risk {snap_proj_val:.2f} is below Critical Threshold (5.0). Escalation inactive.")

            with c_email_prev:
                if "email_html_draft" in st.session_state and snap_proj_val > 5.0:
                    st.markdown("**Email Preview:**")
                    st.components.v1.html(st.session_state.email_html_draft, height=300, scrolling=True)
        else:
            st.warning("No data found before selected date.")

        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("🧠 Forecast Logic"):
            if 'snap_explainer' in locals():
                st.markdown(f"Day: **{snap_explainer['day']}**. Completion: **{snap_explainer['completion_rate']*100:.0f}%**.")

    st.markdown("---")
    st.subheader("💬 Root Cause Analyst")
    
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "I have access to the full dataset. Ask me 'What if we remove BYD model?'"}]
    
    for msg in st.session_state.chat_log:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
    if user_val := st.chat_input("Ask a complex question...", key="chat_widget"):
        st.session_state.chat_log.append({"role": "user", "content": user_val})
        with st.chat_message("user"): st.markdown(user_val)
        with st.chat_message("assistant"):
            if llm:
                with st.spinner("Analyzing full dataset..."):
                    full_chat_context = {
                        "trend_history": weekly[['year', 'week', 'count', 'per_bc']].to_dict(orient='records'),
                        "current_week_deep_dive": wk_payload,
                        "4_week_summary": wk4_payload
                    }
                    system_instructions = f"""
                    You are a data analyst using the full context provided (History + Current).
                    Context: {json.dumps(full_chat_context, cls=NpEncoder)}
                    **Handling 'What-If' Questions:**
                    1. If user asks "What if we remove [MODEL]?", check 'current_week_deep_dive' -> 'model_breakdown'.
                    2. Subtract that model's alarms from 'total_alarms'.
                    3. Recalculate rate = (New Total / total_headcount).
                    **Handling History Questions:** Use 'trend_history'.
                    """
                    response = llm.predict(f"{system_instructions}\nUser Question: {user_val}")
                    st.markdown(response)
                    st.session_state.chat_log.append({"role": "assistant", "content": response})
            else:
                st.write("AI Not Connected.")

if __name__ == "__main__":
    main()
