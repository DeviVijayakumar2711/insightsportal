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

# --- 1. CONFIGURATION HELPER (Azure + Local) ---
def get_config(key, default=""):
    """
    Retrieves configuration from Azure Environment Variables first.
    Completely prevents 'No secrets file found' error by checking file existence first.
    """
    # 1. Check OS Environment (Azure App Settings) - PRIORITY
    if key in os.environ:
        return os.environ[key]
    
    # 2. Check Streamlit Secrets (Local only)
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
1. **SEVERITY FIRST:**
   - If rate > 5.0, you MUST start with "CRITICAL STATUS".
   - If rate > 3.0, start with "ELEVATED STATUS".
2. **Trend Context:** Only after establishing the critical nature, mention the trend. (e.g., "Although trending down, the rate of {last_full_val} remains critically high...").
3. **Primary Contributor:** Identify which Depot is driving the numbers.
4. **Tone:** Urgent, professional, and direct.
"""

PROMPT_WEEKLY_INSIGHT = """
You are an expert Bus Operations Analyst. Alarm type: {alarm_code}.
Using ONLY this JSON summary for a SINGLE week, produce insights.

DATA:
{payload}

Write markdown with EXACT sections:
### Executive Summary
### Actionable Anomaly Detection
- **Nexus of Risk:** Identify cross-category patterns.
- **Category-Specific Insights:** Compare performance vs fleet average.
- **Low Workload, High Rate:** Identify drivers/buses with few trips but high alarms.
### Prioritized Recommendations
Return a 3-row Markdown Table: | Priority | Recommended Action | Data-Driven Rationale |
"""

PROMPT_4WEEK_DEEP = """
You are a world-class Bus Operations Analyst.
Your task is to analyze the root cause of the 4-week trend for '{alarm_code}'.

{context_str}

**DATA:**
{data_json}

**YOUR MISSION (Write in Markdown):**
1. **Overall Trend Diagnosis:** Summary of the 4-week fleet trend.
2. **Key Spike Drivers:** Identify specific Drivers, Buses, or Services causing the trend.
3. **Hidden Insights:** Find the nexus (Driver -> Bus -> Service).
4. **Actionable Summary:** Brief, prioritized summary.
"""

PROMPT_ALERT_EMAIL = """
You are drafting a professional operational alert email in **HTML format**.
SCENARIO:
* **Metric:** {alarm}
* **Projected Value:** {projection}
* **Trend Context:** {trend_context}
* **Top Offenders Data:** {offender_data}

INSTRUCTIONS:
Return **ONLY** the HTML code.
Follow this exact structure:
<h2 style="color: #b91c1c;">Operational Alert: {alarm} Metric Trending into RED ZONE</h2>
<p>
  <strong>Projected Value:</strong> {projection}<br>
  <strong>Percentage Increase:</strong> {trend_context}
</p>
<h3 style="color: #b91c1c;">Root Cause Analysis</h3>
<p>[Write 2 sentences analyzing the 'Top Offenders Data']</p>
<h3>Top Offenders</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <tr style="background-color: #f2f2f2;"><th>Depot</th><th>Service</th><th>Bus</th><th>Model</th><th>Driver</th><th>Count</th></tr>
  [Generate rows from offender_data]
</table>
<h3 style="color: #b91c1c;">Action Required</h3>
<ul>
  <li>Investigate root cause of high alarm frequency on top offending Services.</li>
  <li>Engage with the relevant Depot to address driver-specific alarm patterns.</li>
  <li>Monitor {alarm} metric closely over the next 48 hours.</li>
  <li>Implement corrective actions to mitigate further escalation.</li>
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
    """Smart Data Loader: Handles Azure URL or Local File"""
    path_tele = get_config("TELEMATICS_URL", "telematics_new_data_2207.csv")
    path_head = get_config("HEADCOUNTS_URL", "depot_headcounts.csv")
    path_excl = get_config("EXCLUSIONS_URL", "vehicle_exclusions.csv")

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
        else:
            if os.path.exists(path_str):
                return pd.read_csv(path_str)
            else:
                if is_required:
                    st.error(f"❌ **Data Not Found**: Could not find `{path_str}` local or remote.")
                return pd.DataFrame()

    df = smart_read(path_tele, is_required=True)
    headcounts = smart_read(path_head)
    exclusions = smart_read(path_excl)

    if df.empty: return pd.DataFrame(), pd.DataFrame()

    # Pre-process
    df.columns = [c.lower().strip() for c in df.columns]
    for c in ['bus_no', 'driver_id', 'alarm_type', 'depot_id', 'svc_no']:
        if c in df.columns: 
            df[c] = df[c].astype(str).str.strip().str.upper()
            df[c] = df[c].replace(['NAN', 'NULL', 'NONE', ''], None)
    
    # --- MODEL LOADING LOGIC (MATCHING LOCAL CODE) ---
    try:
        # NOTE: model.csv must be in the deployment zip for this to work
        bus_models = pd.read_csv("model.csv")
        
        # Clean columns
        bus_models.columns = [c.lower().strip() for c in bus_models.columns]
        if 'bus_number' in bus_models.columns:
            bus_models.rename(columns={'bus_number': 'bus_no'}, inplace=True)
        
        # Merge logic
        if 'bus_no' in bus_models.columns:
            bus_models['bus_no'] = bus_models['bus_no'].astype(str).str.strip().str.upper()
            if 'model' in bus_models.columns:
                df = df.merge(bus_models[['bus_no', 'model']], on='bus_no', how='left')
                df['model'] = df['model'].fillna('Unknown')
    except Exception:
        # Fallback if model.csv is missing
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
    
    return df, headcounts

def process_metrics(df, headcounts, alarm_type, depots, exclude_null_driver, only_completed):
    mask = (df['alarm_type'] == alarm_type) & (df['depot_id'].isin(depots)) & (df['depot_id'].notna())
    if exclude_null_driver:
        mask = mask & (df['driver_id'].notna()) & (df['driver_id'] != '0') & (df['driver_id'] != 'NAN')
    
    df_filtered = df[mask].copy()
    if df_filtered.empty:
        return df_filtered, pd.DataFrame(), {}, {}, 0

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
        "depot_breakdown": df_curr['depot_id'].value_counts().to_dict(),
        "top_15_toxic_combinations": df_curr.groupby(['depot_id', 'svc_no', 'bus_no', 'model', 'driver_id']).size().sort_values(ascending=False).head(15).reset_index(name='count').to_dict(orient='records')
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
    # MATCHES LOCAL: No 3rd argument, draws raw data
    fig = go.Figure()
    if weekly_df.empty: return fig
    
    max_y = max(6, weekly_df['per_bc'].max() * 1.1)
    
    fig.add_hrect(y0=0, y1=3, line_width=0, fillcolor="rgba(46, 204, 113, 0.15)", layer="below")
    fig.add_hrect(y0=3, y1=5, line_width=0, fillcolor="rgba(241, 196, 15, 0.15)", layer="below")
    fig.add_hrect(y0=5, y1=max_y, line_width=0, fillcolor="rgba(231, 76, 60, 0.15)", layer="below")
    
    fig.add_trace(go.Scatter(
        x=weekly_df['label'], y=weekly_df['per_bc'],
        mode="lines+markers+text", name=alarm_name,
        line=dict(color="#0072C6", width=3),
        marker=dict(size=8, color="#3498DB"),
        text=weekly_df['per_bc'].round(2), textposition="top center",
        textfont=dict(color="black", size=12)
    ))
    
    fig.add_hline(y=3.0, line_width=1.5, line_dash="dash", line_color="green")
    fig.add_hline(y=5.0, line_width=1.5, line_dash="dash", line_color="red")
    
    fig.update_layout(title=f"12-Week Trend ({alarm_name})", yaxis_title="Avg per Bus Captain", yaxis_range=[0, max_y], showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=350)
    return fig

def plot_single_forecast_bar(comp_df):
    fig = go.Figure()
    colors = []
    for idx, row in comp_df.iterrows():
        if row['status'] == 'Completed':
            colors.append('#BDC3C7')
        else:
            val = row['display_rate']
            if val > 5: colors.append('#E74C3C')
            elif val > 3: colors.append('#F1C40F')
            else: colors.append('#2ECC71')

    fig.add_trace(go.Bar(
        x=comp_df['week_label'], 
        y=comp_df['display_rate'],
        marker_color=colors,
        text=comp_df['display_rate'].round(2),
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>Rate: %{y:.2f}<br>%{customdata}<extra></extra>",
        customdata=comp_df['status']
    ))
    
    fig.add_hline(y=5.0, line_width=2, line_dash="dash", line_color="red", annotation_text="Critical Limit")
    fig.add_hline(y=3.0, line_width=2, line_dash="dash", line_color="orange", annotation_text="Warning Limit")

    fig.update_layout(title="Risk Timeline: Past Actuals vs Current Forecast", margin=dict(l=20, r=20, t=40, b=20), height=350)
    return fig

# --- 7. AUTOMATION ---
def trigger_power_automate(html_body, prediction, recipients_str):
    try:
        url = get_config("POWER_AUTOMATE_FLOW_URL")
        if not url:
            st.error("Power Automate URL is missing.")
            return False
            
        payload = {
            "subject": f"⚠️ ALERT: Rate Projected to {prediction:.2f} (Red Zone)", 
            "body": html_body, 
            "recipient": recipients_str
        }
        res = requests.post(url, json=payload, timeout=6)
        return res.status_code == 202
    except Exception as e:
        st.error(f"Automation Error: {e}")
        return False

# --- 8. GOLD TIER FORECASTING LOGIC (Bayesian) ---
def calculate_smart_forecast(df_filtered, weekly, current_week, total_hc):
    """
    UPGRADED ALGORITHM: Adaptive Bayesian Profile + Momentum Drift
    """
    # --- A. Context & Data Prep ---
    current_data = df_filtered[df_filtered['week'] == current_week]
    if current_data.empty:
        # Fallback if no data yet for current week
        last_data_dt = datetime.now()
        max_day_idx = last_data_dt.weekday()
    else:
        last_data_dt = current_data['date'].max()
        max_day_idx = last_data_dt.weekday() # 0=Mon, 6=Sun
        
    day_name = last_data_dt.strftime('%A')
    
    # --- B. Smart Profiling (Exponential Decay) ---
    past_weeks_ids = weekly[weekly['week'] < current_week]['week'].unique()[-12:]
    
    profile_ratios = []
    weights = []
    
    for i, w in enumerate(past_weeks_ids):
        wk_data = df_filtered[df_filtered['week'] == w]
        total_wk_count = len(wk_data)
        if total_wk_count < 10: continue
        
        count_so_far = len(wk_data[wk_data['day_of_week'] <= max_day_idx])
        ratio = count_so_far / total_wk_count
        
        profile_ratios.append(ratio)
        weights.append(i + 1) 
    
    if profile_ratios:
        completion_rate = np.average(profile_ratios, weights=weights)
    else:
        completion_rate = (max_day_idx + 1) / 7.0
        
    completion_rate = max(0.05, completion_rate) # Clamp
    
    # --- C. Momentum Calculation (Trend Detection) ---
    momentum_factor = 1.0
    if len(weekly) >= 5:
        recent_rates = weekly[weekly['week'] < current_week].tail(4)['per_bc'].values
        if len(recent_rates) > 1:
            x = np.arange(len(recent_rates))
            slope, _ = np.polyfit(x, recent_rates, 1)
            if slope > 0:
                momentum_factor = 1 + (slope * 0.5)

    # --- D. The Projection Calculation ---
    comparison_rows = []
    display_weeks = weekly['week'].unique()[-5:]
    
    # Use real variables for explainer later
    explainer_weight_profile = 0.5
    explainer_weight_avg = 0.5
    
    for w in display_weeks:
        wk_label = f"W{w}"
        is_current = (w == current_week)
        wk_slice = df_filtered[df_filtered['week'] == w]
        hc = max(1, total_hc)
        
        if is_current:
            wk_label = f"W{w} (Fcst)"
            count_so_far = len(wk_slice[wk_slice['day_of_week'] <= max_day_idx])
            
            # 1. Profile Projection
            raw_proj = count_so_far / completion_rate
            
            # 2. Apply Momentum
            adjusted_proj = raw_proj * momentum_factor
            
            # 3. Weighted Ensemble (Bayesian)
            avg_4wk = weekly[weekly['week'] < w].tail(4)['per_bc'].mean()
            if pd.isna(avg_4wk): avg_4wk = count_so_far/hc
            
            # Dynamic Confidence Weight
            weight_profile = min(1.0, completion_rate + 0.15)
            weight_avg = 1.0 - weight_profile
            
            explainer_weight_profile = weight_profile
            explainer_weight_avg = weight_avg
            
            final_rate = ((adjusted_proj/hc) * weight_profile) + (avg_4wk * weight_avg)
            
            comparison_rows.append({
                "week_label": wk_label,
                "status": "In Progress (Projected)",
                "display_rate": final_rate,
                "count_actual": count_so_far,
                "momentum": f"{momentum_factor:.2f}x"
            })
        else:
            actual_total = len(wk_slice)
            comparison_rows.append({
                "week_label": wk_label,
                "status": "Completed",
                "display_rate": actual_total / hc,
                "count_actual": actual_total,
                "momentum": "-"
            })
            
    comp_df = pd.DataFrame(comparison_rows)
    final_proj_val = comp_df.iloc[-1]['display_rate']
    
    explainer = {
        "day": day_name,
        "completion_rate": completion_rate,
        "momentum": momentum_factor,
        "weight_profile": explainer_weight_profile,
        "weight_avg": explainer_weight_avg
    }
    
    return final_proj_val, comp_df, explainer

# --- 9. MAIN APP ---
def main():
    llm = initialize_llm()
    
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        alarm = st.selectbox("Alarm Type", list(ALARM_MAP.keys()))
        
        df, headcounts = load_data()
        
        # Only stop if DF is totally empty (meaning load failed)
        if df.empty:
            st.warning("⚠️ Data is waiting to be loaded. Please ensure TELEMATICS_URL is set in Azure.")
            st.stop()
            
        depot_opts = sorted(headcounts['depot_id'].unique())
        default_depots = [d for d in depot_opts if d in ["WDLAND", "KRANJI", "JURONG"]]
        if not default_depots and len(depot_opts) > 0: default_depots = depot_opts[:2]
        
        depots = st.multiselect("Depot(s)", depot_opts, default=default_depots)
        only_completed = st.checkbox("Only completed weeks", value=True)
        exclude_null_driver = st.checkbox("Exclude null drivers", value=True)
        
        st.markdown("---")
        with st.expander("⚙️ Advanced: Prompt Tuning"):
            prompt_wk = st.text_area("Weekly Insight", PROMPT_WEEKLY_INSIGHT)
            prompt_4wk = st.text_area("4-Week Pattern", PROMPT_4WEEK_DEEP)

    if not depots: st.stop()
    
    # Process
    df_filtered, weekly, wk_payload, wk4_payload, latest_wk_num = process_metrics(
        df, headcounts, alarm, depots, exclude_null_driver, only_completed
    )
    if weekly.empty: st.warning("No data found."); st.stop()
    
    # Headcount
    hc_mask = headcounts['depot_id'].isin(depots)
    total_hc = headcounts[hc_mask]['headcount'].sum()

    # =========================================================
    # CALCULATE PROJECTION (Using GOLD TIER - LOCAL LOGIC)
    # =========================================================
    final_projected_val, comp_df, forecast_explainer = calculate_smart_forecast(
        df_filtered, weekly, latest_wk_num, total_hc
    )

    # =========================================================
    # VISUALS
    # =========================================================
    st.markdown(f"## {ALARM_MAP[alarm]['long']} Intelligence Hub")
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col_chart, col_text = st.columns([2, 1])
        with col_chart:
            # MATCHING LOCAL: Passed raw weekly df only (removed the 3rd arg)
            st.plotly_chart(plot_trend_old_style(weekly.tail(12), alarm), use_container_width=True)
        with col_text:
            st.markdown("### 📋 Executive Brief")
            if llm:
                if "exec_brief" not in st.session_state:
                    with st.spinner("Analyzing trend..."):
                        trend_list = weekly.tail(12)['per_bc'].tolist()
                        last_full_val = trend_list[-2] if len(trend_list) > 1 else trend_list[-1]
                        prompt = PROMPT_EXEC_BRIEF.format(trend_vals=trend_list, last_full_val=last_full_val, depot_stats=wk_payload['depot_breakdown'])
                        st.session_state.exec_brief = llm.predict(prompt)
                st.write(st.session_state.exec_brief)
            else:
                st.info("AI Not Connected")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Operations Center")
    t1, t2, t3 = st.tabs(["📊 Weekly Deep Dive", "🧠 4-Week Pattern", "🔮 Risk Forecast & Alert"])
    
    latest_row = weekly.iloc[-1]

    with t1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Avg per BC", f"{latest_row['per_bc']:.2f}", delta="vs prev week")
            st.metric("Total Events", int(latest_row['count']))
        with c2:
            st.markdown("#### 🕵️‍♂️ Comprehensive Weekly Analysis")
            if llm:
                json_str = json.dumps(wk_payload, indent=2, cls=NpEncoder)
                prompt = prompt_wk.format(alarm_code=alarm, payload=json_str)
                if st.button("Generate Deep Dive"):
                    with st.spinner("Analyzing data..."):
                        st.markdown(llm.predict(prompt))
            else:
                st.write("AI needed.")
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_4wk_disp = weekly.tail(4)['per_bc'].mean()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("4-Week Baseline", f"{avg_4wk_disp:.2f}")
            st.caption("Average rate over last month")
        with c2:
            st.markdown("#### 🧠 Systemic Trend Analysis")
            if llm:
                json_str = json.dumps(wk4_payload, indent=2, cls=NpEncoder)
                context_str = f"Current Rate: {latest_row['per_bc']}. 4-Week Avg: {avg_4wk_disp}."
                prompt = PROMPT_4WEEK_DEEP.format(alarm_code=alarm, context_str=context_str, data_json=json_str)
                if st.button("Run Systemic Scan"):
                    with st.spinner("Scanning trends..."):
                        st.markdown(llm.predict(prompt))
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: UPDATED RISK FORECAST ---
    with t3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        # 1. RISK HEADER
        if final_projected_val > 5.0:
            st.markdown(f"""<div class="alert-banner alert-critical">🚨 <b>CRITICAL RISK:</b> Projected Rate {final_projected_val:.2f} > 5.0</div>""", unsafe_allow_html=True)
        elif final_projected_val > 3.0:
            st.markdown(f"""<div class="alert-banner alert-elevated">⚠️ <b>ELEVATED RISK:</b> Projected Rate {final_projected_val:.2f} > 3.0</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="alert-banner alert-safe">✅ <b>SAFE:</b> Projected Rate {final_projected_val:.2f}</div>""", unsafe_allow_html=True)

        # 2. FORECAST BAR CHART (Past vs Future)
        st.plotly_chart(plot_single_forecast_bar(comp_df), use_container_width=True)
        
        # 3. DETAILED DATA TABLE
        st.markdown("#### 📅 Detailed Forecast Data")
        table_df = comp_df.copy()
        table_df.columns = ["Week", "Status", "Rate (Recorded/Proj)", "Raw Alarms", "Momentum Adj."]
        st.dataframe(table_df, use_container_width=True)
        
        # 4. METHODOLOGY EXPLAINER (WITH CONCRETE EXAMPLE)
        with st.expander("🧠 How 'Gold Tier' Forecast Works (Logic & Example)"):
            st.markdown(f"""
            This system uses a **Bayesian Hybrid Model**, acting as a "Trust Slider" between historical stability and real-time data.
            
            ### The Logic:
            * **Early Week (e.g., Monday):** We trust **History (12-Week Avg)** more because the current data is noisy.
            * **Late Week (e.g., Friday):** We trust the **Current Trend** more because we have real evidence.
            
            ### 📝 Real-Time Example (Based on Current Data):
            * **Current Day:** {forecast_explainer['day']} (Week is **{forecast_explainer['completion_rate']*100:.0f}%** complete).
            * **Trust Allocation:** The model assigns **{forecast_explainer['weight_profile']*100:.0f}% weight** to the Current Trend and **{forecast_explainer['weight_avg']*100:.0f}% weight** to History.
            
            **Example Calculation:**
            1.  **Current Linear Trend:** "If we have 80 faults by Wednesday, we project ~180 by Sunday."
            2.  **Historical Avg:** "Usually, we end the week with 100 faults."
            3.  **Hybrid Blend:** `(180 * {forecast_explainer['weight_profile']:.2f}) + (100 * {forecast_explainer['weight_avg']:.2f})`
            4.  **Final Forecast:** **{final_projected_val:.2f}** (This smooths out the panic while keeping the alert valid).
            """)

        st.markdown("---")
        
        # 5. EMAIL ACTION (Restricted to Red Zone)
        st.subheader("📢 Escalation Protocol")
        c_email_btn, c_email_prev = st.columns([1, 2])
        
        with c_email_btn:
            if final_projected_val > 5.0:
                st.info("Risk Level Critical: Alert Generation Enabled.")
                recipients = st.text_input("Recipients", "devi02@smrt.com.sg; fleet_ops@smrt.com.sg")
                
                if st.button("📝 Draft Alert Email"):
                     if llm:
                        avg_4 = weekly.tail(4)['per_bc'].mean()
                        diff = final_projected_val - avg_4
                        pct_diff = (diff / avg_4) * 100 if avg_4 > 0 else 0
                        trend_txt = f"{abs(pct_diff):.1f}% {'higher' if diff > 0 else 'lower'} than 4-wk average"
                        
                        offender_list = wk_payload['top_15_toxic_combinations'][:5]
                        # UPDATED STRING TO INCLUDE MODEL
                        offender_str = "\n".join([f"<tr><td>{i['depot_id']}</td><td>{i['svc_no']}</td><td>{i['bus_no']}</td><td>{i['model']}</td><td>{i['driver_id']}</td><td>{i['count']}</td></tr>" for i in offender_list])
                        
                        with st.spinner("Drafting email..."):
                            st.session_state.email_html_draft = llm.predict(PROMPT_ALERT_EMAIL.format(
                                alarm=alarm, projection=f"{final_projected_val:.2f}", 
                                trend_context=trend_txt, offender_data=offender_str
                            ))
                     else:
                          st.error("AI not connected")

                if "email_html_draft" in st.session_state:
                    if st.button("📨 Send via Power Automate"):
                        if trigger_power_automate(st.session_state.email_html_draft, final_projected_val, recipients):
                            st.success("Alert Sent Successfully")
                        else:
                            st.error("Transmission Failed")
            else:
                st.success("Current Risk is below Critical Threshold (5.0). Escalation Protocol is inactive.")

        with c_email_prev:
            if "email_html_draft" in st.session_state and final_projected_val > 5.0:
                st.markdown("**Email Preview:**")
                st.components.v1.html(st.session_state.email_html_draft, height=300, scrolling=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💬 Root Cause Analyst")
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "I have access to the full dataset. Ask me 'Why did W44 spike?'"}]
    for msg in st.session_state.chat_log:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    user_val = st.chat_input("Ask a complex question...", key="chat_widget")
    if user_val:
        st.session_state.chat_log.append({"role": "user", "content": user_val})
        with st.chat_message("user"): st.markdown(user_val)
        with st.chat_message("assistant"):
            if llm:
                with st.spinner("Analyzing full dataset..."):
                    raw_context = json.dumps(wk_payload, indent=2, cls=NpEncoder)
                    chat_prompt = f"Senior Data Scientist. Question: {user_val}. Context: {raw_context}"
                    response = llm.predict(chat_prompt)
                    st.markdown(response)
                    st.session_state.chat_log.append({"role": "assistant", "content": response})
            else:
                st.write("AI Not Connected.")

if __name__ == "__main__":
    main()
