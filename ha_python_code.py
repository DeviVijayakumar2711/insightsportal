import os
import json
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Try to import Azure OpenAI
try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    AzureChatOpenAI = None

load_dotenv()

# --- CONFIGURATION (POWER AUTOMATE LINK) ---
HARDCODED_PA_URL = "https://defaultc88daf55f4aa4f79a143526402ab9a.23.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/9c97356356884121962f780d0b6e5e89/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=SbOZpTqO5gOax1mz06R0dJBVvLt3mN2t2tp3lDGfiv4"

def get_config(key, default=""):
    if key in os.environ: return os.environ[key]
    if os.path.exists(".streamlit/secrets.toml"):
        try: return st.secrets.get(key, default)
        except: pass
    return default

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fleet Operations Center", page_icon="🚍", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F4F6F9; }
    .metric-card { background-color: white; padding: 25px; border-radius: 10px; border: 1px solid #E1E4E8; margin-bottom: 20px; }
    .stChatInput { position: fixed; bottom: 20px; width: 70%; left: 15%; z-index: 999; }
</style>
""", unsafe_allow_html=True)

# --- PROMPTS ---
PROMPT_EXEC_BRIEF = """
You are the Head of Fleet Operations. Write a high-level **Executive Briefing** (Max 4 sentences).
DATA: Trend {trend_vals}, Latest Rate {last_full_val}, Depots {depot_stats}.
Tone: Professional and constructive.
"""

PROMPT_WEEKLY_INSIGHT = """
You are an expert Bus Operations Analyst. Alarm: {alarm_code}. Data: {payload}.
Sections: ### Executive Summary, ### Operational Patterns, ### Recommended Actions (Monitor/Review).
"""

PROMPT_4WEEK_DEEP = """
Analyze root cause of 4-week trend for '{alarm_code}'. Data: {data_json}. {context_str}.
"""

PROMPT_ALERT_EMAIL = """
You are drafting a professional operational update email in **HTML format**.
SCENARIO:
* **Metric:** {alarm}
* **Projected Value:** {projection}
* **Trend Context:** {trend_context}
* **Breakdown Insight:** {breakdown_context}
* **Top Contributing Factors:** {offender_data}

INSTRUCTIONS:
Return **ONLY** the HTML code.
Structure:
<h2 style="color: #b91c1c;">Operational Update: {alarm} Metric Status</h2>
<p><strong>Projected Value:</strong> {projection}<br><strong>Context:</strong> {trend_context}</p>

<p>{breakdown_context}</p>

<h3 style="color: #b91c1c;">Analysis</h3>
<p>[Write 2 sentences analyzing the data neutral language.]</p>

<h3>Top Contributing Factors</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <tr style="background-color: #f2f2f2;"><th>Depot</th><th>Service</th><th>Bus</th><th>Model</th><th>Driver</th><th>Total HA Count</th></tr>
  {offender_data}
</table>
<h3 style="color: #b91c1c;">Suggested Actions</h3>
<ul>
  <li>Review root cause of alarm frequency on top services.</li>
  <li>Engage with the relevant Depot to discuss driver support.</li>
</ul>
<p>Best regards,<br>Data Analytics Team</p>
"""

# --- DATA LOGIC ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

@st.cache_data
def load_data():
    path_tele = get_config("TELEMATICS_URL", "telematics_new_data_2207.csv")
    path_head = get_config("HEADCOUNTS_URL", "depot_headcounts.csv")
    path_excl = get_config("EXCLUSIONS_URL", "vehicle_exclusions.csv")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_model_url = get_config("MODEL_URL", "")
    path_model_local = os.path.join(base_dir, "model.csv")

    def smart_read(p, is_req=False):
        if not p: return pd.DataFrame()
        if str(p).startswith("http"):
            try: return pd.read_csv(p)
            except: return pd.DataFrame()
        p = os.path.join(base_dir, p) if not os.path.isabs(p) else p
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    df = smart_read(path_tele, True)
    headcounts = smart_read(path_head)
    exclusions = smart_read(path_excl)
    bus_models = smart_read(path_model_url) if path_model_url else smart_read(path_model_local)

    if df.empty: return pd.DataFrame(), pd.DataFrame(), {}

    df.columns = [c.lower().strip() for c in df.columns]
    for c in ['bus_no', 'driver_id', 'alarm_type', 'depot_id', 'svc_no']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper().str.replace(r'\.0$', '', regex=True)

    if not bus_models.empty:
        bus_models.columns = [c.lower().strip().replace(' ', '_') for c in bus_models.columns]
        found_id = next((c for c in bus_models.columns if c in ['bus_no', 'bus_number', 'vehicle_no']), None)
        if found_id and 'model' in bus_models.columns:
            bus_models.rename(columns={found_id: 'bus_no'}, inplace=True)
            df = df.merge(bus_models[['bus_no', 'model']], on='bus_no', how='left')
            def tag_ev(m):
                if not isinstance(m, str): return "Unknown"
                return f"{m} (EV)" if "BYD" in m.upper() or "ZHONGTONG" in m.upper() else m
            df['model'] = df['model'].apply(tag_ev)
    
    df['model'] = df.get('model', 'Unknown').fillna('Unknown')
    df['date'] = pd.to_datetime(df.get('alarm_calendar_date'), dayfirst=True, errors='coerce')
    df['year'] = df['date'].dt.isocalendar().year
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    return df, headcounts, {}

def process_metrics(df, headcounts, alarm_type, depots, exclude_null, only_comp):
    mask = (df['alarm_type'] == alarm_type) & (df['depot_id'].isin(depots))
    if exclude_null: mask &= (df['driver_id'].notna()) & (df['driver_id'] != '0')
    df_filtered = df[mask].copy()
    if df_filtered.empty: return df_filtered, pd.DataFrame(), {}, {}, 0
    
    weekly = df_filtered.groupby(['year', 'week']).size().reset_index(name='count')
    total_hc = headcounts[headcounts['depot_id'].isin(depots)]['headcount'].sum()
    weekly['per_bc'] = weekly['count'] / max(1, total_hc)
    latest_wk = weekly.iloc[-1]['week']
    df_curr = df_filtered[df_filtered['week'] == latest_wk]

    wk_payload = {
        "week": int(latest_wk), "total_alarms": int(len(df_curr)), "total_hc": int(total_hc),
        "depot_breakdown": df_curr['depot_id'].value_counts().to_dict(),
        "model_breakdown": df_curr['model'].value_counts().to_dict(),
        "top_contributors": df_curr.groupby(['depot_id', 'svc_no', 'bus_no', 'model', 'driver_id']).size().sort_values(ascending=False).head(15).reset_index(name='count').to_dict(orient='records')
    }
    return df_filtered, weekly, wk_payload, {}, latest_wk

def calculate_smart_forecast(df_filtered, weekly, current_week, total_hc):
    current_data = df_filtered[df_filtered['week'] == current_week]
    max_day_idx = current_data['date'].max().weekday() if not current_data.empty else 0
    comp_rate = max(0.05, (max_day_idx + 1) / 7.0)
    hc = max(1, total_hc)
    proj = (len(current_data) / comp_rate) / hc
    
    comp_df = pd.DataFrame([
        {"week_label": f"W{current_week} (Fcst)", "status": "In Progress", "display_rate": proj}
    ])
    return proj, comp_df, {"day": "N/A", "completion_rate": comp_rate}

# --- APP START ---
def main():
    llm = initialize_llm()
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        if st.button("🔄 Reset Cache"): st.cache_data.clear(); st.rerun()
        alarm = st.selectbox("Alarm Type", list(ALARM_MAP.keys()))
        df, headcounts, diag = load_data()
        depot_opts = sorted(headcounts['depot_id'].unique())
        depots = st.multiselect("Depots", depot_opts, default=depot_opts[:3])

    if not depots: st.stop()
    df_filtered, weekly, wk_payload, _, latest_wk = process_metrics(df, headcounts, alarm, depots, True, True)
    total_hc = headcounts[headcounts['depot_id'].isin(depots)]['headcount'].sum()

    st.markdown(f"## {ALARM_MAP[alarm]['long']} Intelligence Hub")
    t1, t2, t3 = st.tabs(["📊 Weekly", "🧠 Pattern", "🔮 Risk & Alert"])

    with t3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        max_dt = df_filtered['date'].max()
        report_date = st.date_input("Drafting Date", value=max_dt if not pd.isna(max_dt) else datetime.now())
        df_snap = df_filtered[df_filtered['date'] <= pd.to_datetime(report_date)]
        
        _, weekly_snap, wk_p_snap, _, wk_num_snap = process_metrics(df_snap, headcounts, alarm, depots, True, True)
        proj_val, snap_comp_df, _ = calculate_smart_forecast(df_snap, weekly_snap, wk_num_snap, total_hc)
        
        st.metric("Projected Risk", f"{proj_val:.2f}", delta_color="inverse")
        
        # --- EMAIL BREAKDOWN LOGIC ---
        m_counts = df_snap[df_snap['week'] == wk_num_snap]['model'].value_counts(normalize=True) * 100
        d_counts = df_snap[df_snap['week'] == wk_num_snap]['depot_id'].value_counts(normalize=True) * 100
        
        model_str = ", ".join([f"{k}: {v:.1f}%" for k, v in m_counts.head(3).items()])
        depot_str = ", ".join([f"{k}: {v:.1f}%" for k, v in d_counts.head(3).items()])
        breakdown_text = f"Projected Week Model Distribution: {model_str}<br>Projected Week Depot Distribution: {depot_str}"
        
        c1, c2 = st.columns([1, 2])
        with c1:
            if proj_val > 5.0:
                st.error("🚨 Critical Risk Enabled")
                recipients = st.text_input("Recipients", "devi02@smrt.com.sg")
                if st.button("📝 Draft Alert Email"):
                    if llm:
                        offenders = wk_p_snap['top_contributors'][:10]
                        off_rows = "\n".join([f"<tr><td>{i['depot_id']}</td><td>{i['svc_no']}</td><td>{i['bus_no']}</td><td>{i['model']}</td><td>{i['driver_id']}</td><td>{i['count']}</td></tr>" for i in offenders])
                        st.session_state.email_draft = llm.predict(PROMPT_ALERT_EMAIL.format(
                            alarm=alarm, projection=f"{proj_val:.2f}", trend_context="High", 
                            breakdown_context=breakdown_text, offender_data=off_rows
                        ))
                if "email_draft" in st.session_state and st.button("📨 Send via Power Automate"):
                    res = requests.post(HARDCODED_PA_URL, json={"subject": f"Alert {proj_val:.2f}", "body": st.session_state.email_draft, "recipient": recipients})
                    if res.status_code in [200, 202]: st.success("Sent!")
        with c2:
            if "email_draft" in st.session_state: st.components.v1.html(st.session_state.email_draft, height=400, scrolling=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__": main()
