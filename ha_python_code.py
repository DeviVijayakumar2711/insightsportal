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

# --- UPDATED EMAIL PROMPT (SIMPLIFIED) ---
PROMPT_ALERT_EMAIL = """
You are drafting a professional operational alert email in **HTML format**.
SCENARIO:
* **Metric:** {alarm}
* **Projected Value:** {projection}
* **Top Contributors Data:** {offender_data}
* **Header Color:** {header_color}

INSTRUCTIONS:
Return **ONLY** the HTML code.
Follow this exact structure:
<h2 style="color: {header_color};">Operational Alert: {alarm} Trending to Red Zone</h2>
<p>
  <strong>Projected value for end of the week:</strong> {projection}
</p>
<p>Below are the key contributors for the alarm count this week:</p>

<h3>Top Contributing Factors</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <tr style="background-color: #f2f2f2;"><th>Depot</th><th>Service</th><th>Bus</th><th>Model</th><th>Driver</th><th>Total HA Count</th></tr>
  {offender_data}
</table>

<h3>Action Required</h3>
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
    
    # Robust Pathing for Azure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_model_url = get_config("MODEL_URL", "").strip()
    path_model_local = os.path.join(base_dir, "model.csv")

    diagnostics = {"model_status": "Not Attempted", "model_cols": [], "merge_count": 0}

    def smart_read(path_val, is_required=False):
        if not path_val: return pd.DataFrame()
        path_str = str(path_val).strip()
        
        # A) URL Read
        if path_str.lower().startswith("http"):
            try:
                return pd.read_csv(path_str)
            except Exception as e:
                if is_required: st.error(f"Read Error: {e}")
                return pd.DataFrame()
        
        # B) Local Read
        if not os.path.isabs(path_str): path_str = os.path.join(base_dir, path_str)
        if os.path.exists(path_str):
            return pd.read_csv(path_str)
        else:
            if is_required: st.error(f"File Not Found: {path_str}")
            return pd.DataFrame()

    df = smart_read(path_tele, True)
    headcounts = smart_read(path_head)
    exclusions = smart_read(path_excl)
    
    # Model Loading (Hybrid)
    bus_models = pd.DataFrame()
    if path_model_url:
        try: bus_models = pd.read_csv(path_model_url); diagnostics['model_status'] = "URL Loaded"
        except: pass
    
    if bus_models.empty and os.path.exists(path_model_local):
        try: bus_models = pd.read_csv(path_model_local); diagnostics['model_status'] = "Local Loaded"
        except: pass

    if df.empty: return pd.DataFrame(), pd.DataFrame(), diagnostics

    # Clean Telematics
    df.columns = [c.lower().strip() for c in df.columns]
    for c in ['bus_no', 'driver_id', 'alarm_type', 'depot_id', 'svc_no']:
        if c in df.columns: 
            df[c] = df[c].astype(str).str.strip().str.upper().replace(['NAN', 'NULL'], None)
            if c == 'driver_id': df[c] = df[c].str.replace(r'\.0$', '', regex=True)

    # Merge Models
    if not bus_models.empty:
        bus_models.columns = [c.lower().strip().replace(' ', '_') for c in bus_models.columns]
        found_id = next((c for c in bus_models.columns if c in ['bus_no', 'bus_number', 'vehicle_no']), None)
        
        if found_id:
            bus_models.rename(columns={found_id: 'bus_no'}, inplace=True)
            bus_models['bus_no'] = bus_models['bus_no'].astype(str).str.strip().str.upper()
            
            if 'model' in bus_models.columns:
                df = df.merge(bus_models[['bus_no', 'model']], on='bus_no', how='left')
                df['model'] = df['model'].fillna('Unknown')
                
                # EV Tagging Logic
                def tag_ev(m):
                    if not isinstance(m, str): return "Unknown"
                    upper_m = m.upper()
                    if ('BYD' in upper_m or 'ZHONGTONG' in upper_m) and '(EV)' not in m:
                        return f"{m} (EV)"
                    return m
                
                df['model'] = df['model'].apply(tag_ev)
                diagnostics['merge_count'] = len(df[df['model'] != 'Unknown'])

    if 'model' not in df.columns: df['model'] = 'Unknown'

    # Filter Exclusions
    if not exclusions.empty and 'bus_no' in exclusions.columns:
        excl_list = set(exclusions['bus_no'].astype(str).str.strip().str.upper())
        df = df[~df['bus_no'].isin(excl_list)]

    # Clean Headcounts
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
            
        alarm = st.selectbox("Alarm Type", list(ALARM_MAP.keys()))
        df, headcounts, diag = load_data()
        
        with st.expander("🛠️ Diagnostics"):
            st.write(f"Status: {diag.get('model_status')}")
            st.write(f"Matched: {diag.get('merge_count')} records")
            
        if df.empty:
            st.warning("⚠️ Data Missing. Check Azure URLs.")
            st.stop()
            
        depot_opts = sorted(headcounts['depot_id'].unique())
        default_depots = [d for d in depot_opts if d in ["WDLAND", "KRANJI", "JURONG"]]
        if not default_depots: default_depots = depot_opts[:2]
        
        depots = st.multiselect("Depots", depot_opts, default=default_depots)
        only_comp = st.checkbox("Only completed weeks", value=True)
        excl_null = st.checkbox("Exclude null drivers", value=True)

    if not depots: st.stop()
    
    df_filtered, weekly, wk_payload, wk4_payload, latest_wk = process_metrics(
        df, headcounts, alarm, depots, excl_null, only_comp
    )
    if weekly.empty: st.warning("No Data Found"); st.stop()
    
    hc_mask = headcounts['depot_id'].isin(depots)
    total_hc = headcounts[hc_mask]['headcount'].sum()
    
    # Calculate Forecast
    proj_val, comp_df, explainer = calculate_smart_forecast(df_filtered, weekly, latest_wk, total_hc)

    st.markdown(f"## {ALARM_MAP[alarm]['long']} Intelligence Hub")
    
    c1, c2 = st.columns([2, 1])
    with c1: st.plotly_chart(plot_trend_old_style(weekly.tail(12), alarm), use_container_width=True)
    with c2:
        st.markdown("### 📋 Executive Brief")
        if llm:
            if "exec_brief" not in st.session_state:
                with st.spinner("Drafting..."):
                    st.session_state.exec_brief = llm.predict(PROMPT_EXEC_BRIEF.format(trend_vals=weekly.tail(12)['per_bc'].tolist(), last_full_val=weekly.iloc[-2]['per_bc'], depot_stats=wk_payload['depot_breakdown']))
            st.write(st.session_state.exec_brief)
        else: st.info("AI Not Connected")

    t1, t2, t3 = st.tabs(["📊 Weekly Deep Dive", "🧠 4-Week Pattern", "🔮 Risk Forecast"])
    
    with t1:
        c1, c2 = st.columns([1, 2])
        c1.metric("Avg per BC", f"{weekly.iloc[-1]['per_bc']:.2f}")
        with c2:
            if llm and st.button("Generate Deep Dive"):
                with st.spinner("Analyzing..."): st.markdown(llm.predict(PROMPT_WEEKLY_INSIGHT.format(alarm_code=alarm, payload=json.dumps(wk_payload, cls=NpEncoder))))

    with t2:
        c1, c2 = st.columns([1, 2])
        c1.metric("4-Week Baseline", f"{weekly.tail(4)['per_bc'].mean():.2f}")
        with c2:
            if llm and st.button("Run Systemic Scan"):
                with st.spinner("Scanning..."): st.markdown(llm.predict(PROMPT_4WEEK_DEEP.format(alarm_code=alarm, context_str=f"Rate: {weekly.iloc[-1]['per_bc']}", data_json=json.dumps(wk4_payload, cls=NpEncoder))))

    with t3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        c_date, c_stat = st.columns([1, 2])
        with c_date:
            max_dt = df_filtered['date'].max()
            if pd.isna(max_dt): max_dt = datetime.now()
            report_date = st.date_input("Drafting Date (Snapshot)", value=max_dt, max_value=datetime.now())
        
        # Snapshot Logic
        snap_dt = pd.to_datetime(report_date)
        df_snap = df_filtered[df_filtered['date'] <= snap_dt]
        
        if not df_snap.empty:
            _, weekly_snap, wk_p_snap, _, wk_num_snap = process_metrics(df_snap, headcounts, alarm, depots, excl_null, only_comp)
            weekly_snap = weekly_snap[weekly_snap['week'] <= wk_num_snap]
            
            snap_val, snap_comp_df, snap_exp = calculate_smart_forecast(df_snap, weekly_snap, wk_num_snap, total_hc)
            
            with c_stat:
                st.metric("Projected Risk (As of Snapshot)", f"{snap_val:.2f}", delta=f"W{wk_num_snap} Projection", delta_color="inverse")
            
            st.plotly_chart(plot_single_forecast_bar(snap_comp_df), use_container_width=True)
            
            st.markdown("---")
            st.subheader("📢 Escalation Protocol")
            c_email_btn, c_email_prev = st.columns([1, 2])
            
            with c_email_btn:
                # Trigger Yellow (>3.0) and Red (>5.0)
                if snap_val > 3.0:
                    is_critical = snap_val > 5.0
                    header_col = "#b91c1c" if is_critical else "#d97706"
                    
                    if is_critical:
                        st.error("🚨 Critical Risk (>5.0): Alert Generation Enabled.")
                    else:
                        st.warning("⚠️ Elevated Risk (>3.0): Alert Generation Enabled.")
                        
                    recipients = st.text_input("Recipients", "devi02@smrt.com.sg; fleet_ops@smrt.com.sg")
                    
                    if st.button("📝 Draft Alert Email"):
                        if llm:
                            # 2. Top 10 Offenders
                            offenders = wk_p_snap['top_contributors'][:10]
                            off_rows = "\n".join([f"<tr><td>{i['depot_id']}</td><td>{i['svc_no']}</td><td>{i['bus_no']}</td><td>{i['model']}</td><td>{i['driver_id']}</td><td>{i['count']}</td></tr>" for i in offenders])
                            
                            # 3. Generate Email
                            with st.spinner("Drafting..."):
                                st.session_state.email_draft = llm.predict(PROMPT_ALERT_EMAIL.format(
                                    alarm=alarm, projection=f"{snap_val:.2f}", 
                                    trend_context=f"Risk Level: {'Critical' if is_critical else 'Elevated'}", 
                                    breakdown_context="", # Removing complex distribution string to keep it simple as per request
                                    offender_data=off_rows,
                                    header_color=header_col
                                ))
                        else: st.error("AI Not Connected")
                    
                    if "email_draft" in st.session_state:
                        if st.button("📨 Send via Power Automate"):
                            try:
                                res = requests.post(HARDCODED_PA_URL, json={"subject": f"Alert {snap_val:.2f}", "body": st.session_state.email_draft, "recipient": recipients}, timeout=10)
                                if res.status_code in [200, 202]: st.success("Sent Successfully!")
                                else: st.error(f"Failed: {res.status_code}")
                            except Exception as e: st.error(f"Error: {e}")
                else: st.success(f"Risk {snap_val:.2f} is Safe (<3.0). Escalation inactive.")
            
            with c_email_prev:
                if "email_draft" in st.session_state and snap_val > 3.0:
                    st.markdown("**Email Preview:**")
                    st.components.v1.html(st.session_state.email_draft, height=400, scrolling=True)
        else: st.warning("No Data found for selected snapshot.")
        st.markdown('</div>', unsafe_allow_html=True)

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
                    # FIX: Inject Detailed Historical Breakdown (Bus, Svc, Driver)
                    recent_weeks = weekly['week'].tail(5).tolist()
                    hist_data = {}
                    for w in recent_weeks:
                        d = df_filtered[df_filtered['week'] == w]
                        if not d.empty:
                            hist_data[f"W{w}"] = {
                                "total": int(len(d)),
                                "depot": d['depot_id'].value_counts().head(3).to_dict(),
                                "model": d['model'].value_counts().head(3).to_dict(),
                                "svc": d['svc_no'].value_counts().head(3).to_dict(),
                                "driver": d['driver_id'].value_counts().head(3).to_dict(),
                                "bus": d['bus_no'].value_counts().head(3).to_dict()
                            }

                    ctx = {
                        "trend": weekly[['year', 'week', 'count', 'per_bc']].to_dict('records'),
                        "history_breakdown": hist_data,
                        "current_deep_dive": wk_payload
                    }
                    sys = f"Analyst. Context: {json.dumps(ctx, cls=NpEncoder)}. Answer user question."
                    resp = llm.predict(f"{sys}\nQ: {u}")
                    st.markdown(resp); st.session_state.chat_log.append({"role": "assistant", "content": resp})
            else: st.write("AI Not Connected")

if __name__ == "__main__":
    main()
