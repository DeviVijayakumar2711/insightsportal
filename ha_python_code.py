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

# ==========================================
# 🔧 CONFIGURATION (POWER AUTOMATE LINK)
# ==========================================
HARDCODED_PA_URL = "https://defaultc88daf55f4aa4f79a143526402ab9a.23.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/9c97356356884121962f780d0b6e5e89/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=SbOZpTqO5gOax1mz06R0dJBVvLt3mN2t2tp3lDGfiv4"

# --- 1. CONFIGURATION HELPER ---
def get_config(key, default=""):
    if key in os.environ: return os.environ[key]
    if os.path.exists(".streamlit/secrets.toml"):
        try: return st.secrets.get(key, default)
        except: pass
    return default

# --- 2. PAGE CONFIG & CSS ---
st.set_page_config(page_title="Fleet Operations Center", page_icon="🚍", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F4F6F9; }
    .metric-card { background-color: white; padding: 25px; border-radius: 10px; border: 1px solid #E1E4E8; margin-bottom: 20px; }
    .stChatInput { position: fixed; bottom: 20px; width: 70%; left: 15%; z-index: 999; }
</style>
""", unsafe_allow_html=True)

# --- 3. PROMPTS ---
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

# Updated Email Prompt is handled dynamically in Python for the Consolidated Tab, 
# keeping the single alert prompt for the standard tab.
PROMPT_ALERT_EMAIL = """
You are drafting a professional operational update email in **HTML format**.
SCENARIO:
* **Metric:** {alarm}
* **Alert Title:** {alert_title} 
* **Header Color:** {header_color}
* **Projected Value:** {projection}
* **Trend Context:** {trend_context}
* **Breakdown Insight:** {breakdown_context}
* **Top Contributing Factors:** {offender_data}

INSTRUCTIONS:
Return **ONLY** the HTML code.
Follow this exact structure:
<h2 style="color: {header_color};">{alert_title}</h2>
<p>
  <strong>Projected value for end of the week:</strong> {projection}<br>
  <strong>Context:</strong> {trend_context}
</p>

<p>{breakdown_context}</p>

<h3 style="color: {header_color};">Analysis</h3>
<p>[Write 2 sentences analyzing the 'Top Contributing Factors'. Explicitly mention if specific models (like EV) are driving the trend. Use neutral language.]</p>

<h3>Top Contributing Factors</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <tr style="background-color: #f2f2f2;"><th>Depot</th><th>Service</th><th>Bus</th><th>Model</th><th>Driver</th><th>Total HA Count</th></tr>
  {offender_data}
</table>
<h3 style="color: {header_color};">Action Required</h3>
<ul>
  <li>Depot Head to engage with driver and monitor.</li>
  <li>Talk with SIS to check on data issues/sensor issues for the list flagged out.</li>
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
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

@st.cache_data
def load_data():
    """Robust Data Loader with Diagnostics"""
    path_tele = get_config("TELEMATICS_URL", "telematics_new_data_2207.csv")
    path_head = get_config("HEADCOUNTS_URL", "depot_headcounts.csv")
    path_excl = get_config("EXCLUSIONS_URL", "vehicle_exclusions.csv")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_model_url = get_config("MODEL_URL", "").strip()
    path_model_local = os.path.join(base_dir, "model.csv")

    diagnostics = {"model_status": "Not Attempted", "model_cols": [], "merge_count": 0}

    def smart_read(path_val, is_required=False):
        if not path_val: return pd.DataFrame()
        path_str = str(path_val).strip()
        
        if path_str.lower().startswith("http"):
            try: return pd.read_csv(path_str)
            except Exception as e:
                if is_required: st.error(f"Read Error: {e}")
                return pd.DataFrame()
        
        if not os.path.isabs(path_str): path_str = os.path.join(base_dir, path_str)
        if os.path.exists(path_str): return pd.read_csv(path_str)
        else:
            if is_required: st.error(f"File Not Found: {path_str}")
            return pd.DataFrame()

    df = smart_read(path_tele, True)
    headcounts = smart_read(path_head)
    exclusions = smart_read(path_excl)
    
    bus_models = pd.DataFrame()
    if path_model_url:
        try: bus_models = pd.read_csv(path_model_url); diagnostics['model_status'] = "URL Loaded"
        except: pass
    
    if bus_models.empty and os.path.exists(path_model_local):
        try: bus_models = pd.read_csv(path_model_local); diagnostics['model_status'] = "Local Loaded"
        except: pass

    if df.empty: return pd.DataFrame(), pd.DataFrame(), diagnostics

    df.columns = [c.lower().strip() for c in df.columns]
    for c in ['bus_no', 'driver_id', 'alarm_type', 'depot_id', 'svc_no']:
        if c in df.columns: 
            df[c] = df[c].astype(str).str.strip().str.upper().replace(['NAN', 'NULL'], None)
            if c == 'driver_id': df[c] = df[c].str.replace(r'\.0$', '', regex=True)

    if not bus_models.empty:
        bus_models.columns = [c.lower().strip().replace(' ', '_') for c in bus_models.columns]
        found_id = next((c for c in bus_models.columns if c in ['bus_no', 'bus_number', 'vehicle_no']), None)
        
        if found_id:
            bus_models.rename(columns={found_id: 'bus_no'}, inplace=True)
            bus_models['bus_no'] = bus_models['bus_no'].astype(str).str.strip().str.upper()
            
            if 'model' in bus_models.columns:
                df = df.merge(bus_models[['bus_no', 'model']], on='bus_no', how='left')
                df['model'] = df['model'].fillna('Unknown')
                def tag_ev(m):
                    if not isinstance(m, str): return "Unknown"
                    upper_m = m.upper()
                    if ('BYD' in upper_m or 'ZHONGTONG' in upper_m) and '(EV)' not in m: return f"{m} (EV)"
                    return m
                df['model'] = df['model'].apply(tag_ev)
                diagnostics['merge_count'] = len(df[df['model'] != 'Unknown'])

    if 'model' not in df.columns: df['model'] = 'Unknown'

    if not exclusions.empty and 'bus_no' in exclusions.columns:
        excl_list = set(exclusions['bus_no'].astype(str).str.strip().str.upper())
        df = df[~df['bus_no'].isin(excl_list)]

    if 'depot_id' in headcounts.columns:
        headcounts['depot_id'] = headcounts['depot_id'].astype(str).str.strip().str.upper()
    else:
        headcounts = pd.DataFrame(columns=['depot_id', 'headcount'])

    df['date'] = pd.to_datetime(df.get('alarm_calendar_date'), dayfirst=True, errors='coerce')
    df['year'] = df['date'].dt.isocalendar().year
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek 
    
    return df, headcounts, diagnostics

def process_metrics(df, headcounts, alarm_type, depots, exclude_null, only_comp):
    mask = (df['alarm_type'] == alarm_type) & (df['depot_id'].isin(depots))
    if exclude_null: mask &= (df['driver_id'].notna()) & (df['driver_id'] != '0')
    
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

    wk_payload = {
        "week": int(latest_wk),
        "total_alarms": int(len(df_curr)),
        "total_headcount": int(total_hc),
        "depot_breakdown": df_curr['depot_id'].value_counts().to_dict(),
        "model_breakdown": df_curr['model'].value_counts().to_dict(),
        "top_contributors": df_curr.groupby(['depot_id', 'svc_no', 'bus_no', 'model', 'driver_id']).size().sort_values(ascending=False).head(15).reset_index(name='count').to_dict(orient='records')
    }
    
    df_4wk = df_filtered[df_filtered['week'] >= (latest_wk - 3)]
    wk4_payload = {
        "drivers": df_4wk.groupby(['driver_id', 'week']).size().unstack(fill_value=0).sum(axis=1).to_dict()
    }
    
    return df_filtered, weekly, wk_payload, wk4_payload, latest_wk

def calculate_smart_forecast(df_filtered, weekly, current_week, total_hc):
    latest_row = weekly.iloc[-1]
    current_data = df_filtered[df_filtered['week'] == current_week]
    
    if not current_data.empty:
        last_data_dt = current_data['date'].max()
        max_day_idx = last_data_dt.weekday()
        day_name = last_data_dt.strftime('%A')
    else:
        max_day_idx = 0; day_name = "Monday"

    past_weeks = weekly[weekly['week'] < current_week]['week'].unique()[-12:]
    ratios = []
    for w in past_weeks:
        wd = df_filtered[df_filtered['week'] == w]
        if len(wd) > 10:
            ratios.append(len(wd[wd['day_of_week'] <= max_day_idx]) / len(wd))
            
    comp_rate = np.mean(ratios) if ratios else (max_day_idx + 1) / 7.0
    comp_rate = max(0.05, comp_rate)
    hc = max(1, total_hc)
    
    momentum = 1.0
    if len(weekly) >= 5:
        rates = weekly[weekly['week'] < current_week].tail(4)['per_bc'].values
        if len(rates) > 1:
            s, _ = np.polyfit(np.arange(len(rates)), rates, 1)
            if s > 0: momentum = 1 + (s * 0.5)

    comp_rows = []
    display_weeks = weekly['week'].unique()[-5:]
    
    for w in display_weeks:
        is_curr = (w == current_week)
        wk_slice = df_filtered[df_filtered['week'] == w]
        if is_curr:
            count_sofar = len(wk_slice[wk_slice['day_of_week'] <= max_day_idx])
            raw = count_sofar / comp_rate
            adj = raw * momentum
            avg_4 = weekly[weekly['week'] < w].tail(4)['per_bc'].mean()
            if pd.isna(avg_4): avg_4 = count_sofar/hc
            w_prof = min(1.0, comp_rate + 0.15)
            final = ((adj/hc) * w_prof) + (avg_4 * (1-w_prof))
            comp_rows.append({"week_label": f"W{w} (Fcst)", "status": "In Progress", "display_rate": final})
        else:
            comp_rows.append({"week_label": f"W{w}", "status": "Completed", "display_rate": len(wk_slice)/hc})
            
    return comp_rows[-1]['display_rate'], pd.DataFrame(comp_rows), {"day": day_name, "completion_rate": comp_rate}

def plot_trend_old_style(weekly_df, alarm_name):
    fig = go.Figure()
    if weekly_df.empty: return fig
    max_y = max(6, weekly_df['per_bc'].max() * 1.1)
    
    fig.add_hrect(y0=0, y1=3, line_width=0, fillcolor="rgba(46, 204, 113, 0.15)", layer="below")
    fig.add_hrect(y0=3, y1=5, line_width=0, fillcolor="rgba(241, 196, 15, 0.15)", layer="below")
    fig.add_hrect(y0=5, y1=max_y, line_width=0, fillcolor="rgba(231, 76, 60, 0.15)", layer="below")
    
    fig.add_trace(go.Scatter(
        x=weekly_df['label'], y=weekly_df['per_bc'], mode="lines+markers+text",
        line=dict(color="#0072C6", width=3), marker=dict(size=8, color="#3498DB"),
        text=weekly_df['per_bc'].round(2), textposition="top center"
    ))
    fig.update_layout(title=f"12-Week Trend ({alarm_name})", yaxis_range=[0, max_y], height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_single_forecast_bar(comp_df):
    fig = go.Figure()
    cols = ['#BDC3C7' if r['status'] == 'Completed' else ('#E74C3C' if r['display_rate']>5 else '#2ECC71') for _, r in comp_df.iterrows()]
    fig.add_trace(go.Bar(x=comp_df['week_label'], y=comp_df['display_rate'], marker_color=cols, text=comp_df['display_rate'].round(2), textposition='auto'))
    fig.add_hline(y=5.0, line_dash="dash", line_color="red")
    fig.update_layout(title="Risk Timeline", height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- 7. MAIN APP ---
def main():
    llm = initialize_llm()
    
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        if st.button("🔄 Reset Cache & Reload"):
            st.cache_data.clear()
            st.rerun()
            
        alarm = st.selectbox("Alarm Type (Specific Tabs)", list(ALARM_MAP.keys()))
        df, headcounts, diag = load_data()
        
        with st.expander("🛠️ Diagnostics"):
            st.write(f"Status: {diag.get('model_status')}")
            st.write(f"Matched: {diag.get('merge_count')} records")
            
        if df.empty: st.warning("⚠️ Data Missing. Check Azure URLs."); st.stop()
            
        depot_opts = sorted(headcounts['depot_id'].unique())
        default_depots = [d for d in depot_opts if d in ["WDLAND", "KRANJI", "JURONG"]]
        if not default_depots: default_depots = depot_opts[:2]
        
        depots = st.multiselect("Depots", depot_opts, default=default_depots)
        only_comp = st.checkbox("Only completed weeks", value=True)
        excl_null = st.checkbox("Exclude null drivers", value=True)

    if not depots: st.stop()
    
    # Process Single Alarm for Tabs 1-3
    df_filtered, weekly, wk_payload, wk4_payload, latest_wk = process_metrics(
        df, headcounts, alarm, depots, excl_null, only_comp
    )
    
    hc_mask = headcounts['depot_id'].isin(depots)
    total_hc = headcounts[hc_mask]['headcount'].sum()
    
    # Forecast Single Alarm
    if not weekly.empty:
        proj_val, comp_df, explainer = calculate_smart_forecast(df_filtered, weekly, latest_wk, total_hc)
    else:
        proj_val, comp_df = 0, pd.DataFrame()

    st.markdown(f"## {ALARM_MAP[alarm]['long']} Intelligence Hub")
    
    # Tabs
    t1, t2, t3, t4 = st.tabs(["📊 Weekly Deep Dive", "🧠 4-Week Pattern", "🔮 Risk Forecast", "📧 Consolidated Alert"])
    
    # --- TAB 1, 2, 3: Standard Single Alarm ---
    with t1:
        if not weekly.empty:
            c1, c2 = st.columns([2, 1])
            with c1: st.plotly_chart(plot_trend_old_style(weekly.tail(12), alarm), use_container_width=True)
            with c2: st.metric("Avg per BC", f"{weekly.iloc[-1]['per_bc']:.2f}")
            if llm and st.button("Generate Deep Dive"):
                with st.spinner("Analyzing..."): st.markdown(llm.predict(PROMPT_WEEKLY_INSIGHT.format(alarm_code=alarm, payload=json.dumps(wk_payload, cls=NpEncoder))))
    with t2:
        if not weekly.empty:
            st.metric("4-Week Baseline", f"{weekly.tail(4)['per_bc'].mean():.2f}")
            if llm and st.button("Run Systemic Scan"):
                with st.spinner("Scanning..."): st.markdown(llm.predict(PROMPT_4WEEK_DEEP.format(alarm_code=alarm, context_str=f"Rate: {weekly.iloc[-1]['per_bc']}", data_json=json.dumps(wk4_payload, cls=NpEncoder))))
    with t3:
        if not weekly.empty:
            st.metric("Projected Risk", f"{proj_val:.2f}", delta_color="inverse")
            st.plotly_chart(plot_single_forecast_bar(comp_df), use_container_width=True)

    # --- TAB 4: CONSOLIDATED GENERAL ALERT ---
    with t4:
        st.markdown("### 📧 General Consolidated Alert (HA, HB, HC)")
        st.info("This tab scans ALL metrics (HA, HB, HC) and generates a single combined email if they are trending > 3.0.")
        
        c_gen_date, c_gen_act = st.columns([1, 2])
        with c_gen_date:
            gen_report_date = st.date_input("Drafting Date (General)", value=datetime.now())
        
        if st.button("🚀 Scan All Metrics & Draft Email"):
            active_alerts = []
            alerts_meta = []
            max_risk = 0.0
            
            # Loop through all 3 alarm types
            for a_type in ["HA", "HB", "HC"]:
                # Process each metric independently
                _df_f, _wk, _pl, _, _lwk = process_metrics(df, headcounts, a_type, depots, excl_null, only_comp)
                
                # Filter by date snapshot
                _df_snap = _df_f[_df_f['date'] <= pd.to_datetime(gen_report_date)]
                
                if not _df_snap.empty:
                    # Recalculate metrics for snapshot
                    _wk_snap = _df_snap.groupby(['year', 'week']).size().reset_index(name='count')
                    _wk_snap['per_bc'] = _wk_snap['count'] / max(1, total_hc)
                    _latest_wk_snap = _wk_snap.iloc[-1]['week']
                    
                    # Calculate Projection
                    _proj, _, _ = calculate_smart_forecast(_df_snap, _wk_snap, _latest_wk_snap, total_hc)
                    
                    # Check Threshold > 3.0
                    if _proj > 3.0:
                        max_risk = max(max_risk, _proj)
                        
                        # Get Top 10 Contributors for this specific alarm
                        _curr_wk_data = _df_snap[_df_snap['week'] == _latest_wk_snap]
                        _top_10 = _curr_wk_data.groupby(['depot_id', 'svc_no', 'bus_no', 'model', 'driver_id']).size().sort_values(ascending=False).head(10).reset_index(name='count')
                        
                        # Format Table Rows
                        _rows = ""
                        for _, row in _top_10.iterrows():
                            _rows += f"<tr><td>{row['depot_id']}</td><td>{row['svc_no']}</td><td>{row['bus_no']}</td><td>{row['model']}</td><td>{row['driver_id']}</td><td>{row['count']}</td></tr>"
                        
                        risk_label = "Medium Risk (>3.0)" if _proj <= 5.0 else "Critical Risk (>5.0)"
                        alerts_meta.append(f"Projected value for end of the week ({a_type}): {_proj:.2f} [{risk_label}]")
                        
                        # Create Table Block
                        active_alerts.append(f"""
                        <h3>Top Contributing Factors for {a_type}</h3>
                        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                          <tr style="background-color: #f2f2f2;"><th>Depot</th><th>Service</th><th>Bus</th><th>Model</th><th>Driver</th><th>Total {a_type} Count</th></tr>
                          {_rows}
                        </table>
                        """)

            # --- BUILD EMAIL ---
            if active_alerts:
                is_crit = max_risk > 5.0
                header_col = "#b91c1c" if is_crit else "#d97706"
                header_title = "Operational Alert: " + " & ".join([x.split('(')[1].split(')')[0] for x in alerts_meta]) + f" Trending to {'Red' if is_crit else 'Yellow'} Zone"
                
                # Manual HTML Construction (No LLM needed for strict formatting)
                projection_line = "&nbsp;&nbsp;&nbsp;".join(alerts_meta)
                tables_html = "".join(active_alerts)
                
                final_email_html = f"""
                <h2 style="color: {header_col};">Operational Alert: HA (Harsh Acceleration) & HC (Harsh Cornering) Trending to {'Red' if is_crit else 'Yellow'} Zone</h2>
                <p><strong>{projection_line}</strong></p>
                
                <h3 style="color: {header_col};">Analysis</h3>
                <p>Below are the top contributing factors for the flagged alarms this week.</p>
                
                {tables_html}
                
                <h3 style="color: {header_col};">Action Required</h3>
                <ul>
                  <li>Depot Head to engage with driver and monitor for the week.</li>
                  <li>Talk with SIS to check on data / sensor issues for the list flagged out.</li>
                </ul>
                <p>Best regards,<br>Data Analytics Team</p>
                """
                
                st.session_state.gen_email = final_email_html
                st.success(f"Generated Alert for {len(active_alerts)} metrics.")
            else:
                st.success("✅ All systems Green. No alarms trending > 3.0.")

        # Show Preview & Send
        if "gen_email" in st.session_state:
            recipients = st.text_input("Recipients (General)", "devi02@smrt.com.sg; fleet_ops@smrt.com.sg")
            if st.button("📨 Send General Alert via Power Automate"):
                try:
                    res = requests.post(HARDCODED_PA_URL, json={"subject": "Consolidated Operational Alert", "body": st.session_state.gen_email, "recipient": recipients}, timeout=10)
                    if res.status_code in [200, 202]: st.success("Sent Successfully!")
                    else: st.error(f"Failed: {res.status_code}")
                except Exception as e: st.error(f"Error: {e}")
            
            st.markdown("**Email Preview:**")
            st.components.v1.html(st.session_state.gen_email, height=600, scrolling=True)

    # Chatbot Logic (Same as before)
    st.markdown("---")
    st.subheader("💬 Root Cause Analyst")
    if "chat_log" not in st.session_state: st.session_state.chat_log = [{"role": "assistant", "content": "Ask me about specific weeks or trends."}]
    for m in st.session_state.chat_log: st.chat_message(m["role"]).markdown(m["content"])
    
    if u := st.chat_input("Ask analyst..."):
        st.session_state.chat_log.append({"role": "user", "content": u})
        st.chat_message("user").markdown(u)
        with st.chat_message("assistant"):
            if llm:
                with st.spinner("Analyzing..."):
                    recent_weeks = weekly['week'].tail(5).tolist()
                    hist_data = {}
                    for w in recent_weeks:
                        d = df_filtered[df_filtered['week'] == w]
                        if not d.empty:
                            hist_data[f"Week {w}"] = {
                                "total": int(len(d)),
                                "depot": d['depot_id'].value_counts().head(5).to_dict(),
                                "model": d['model'].value_counts().head(5).to_dict(),
                                "svc": d['svc_no'].value_counts().head(5).to_dict(),
                                "driver": d['driver_id'].value_counts().head(5).to_dict(),
                                "bus": d['bus_no'].value_counts().head(5).to_dict()
                            }
                    ctx = { "trend": weekly[['year', 'week', 'count', 'per_bc']].to_dict('records'), "history_breakdown": hist_data, "curr": wk_payload }
                    sys = f"Analyst. Context: {json.dumps(ctx, cls=NpEncoder)}. Answer user question."
                    resp = llm.predict(f"{sys}\nQ: {u}")
                    st.markdown(resp); st.session_state.chat_log.append({"role": "assistant", "content": resp})
            else: st.write("AI Not Connected")

if __name__ == "__main__":
    main()
