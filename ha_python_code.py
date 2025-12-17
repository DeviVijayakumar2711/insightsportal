import os
import json
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Try to import Azure OpenAI, handle failure gracefully
try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    AzureChatOpenAI = None

# Load local .env file (for local development only)
load_dotenv()

# --- 1. CONFIGURATION HELPER (ULTRA-SAFE FIX) ---
def get_config(key, default=""):
    """
    Retrieves configuration from Azure Environment Variables first.
    Completely prevents 'No secrets file found' error by checking file existence first.
    """
    # 1. Check OS Environment (Azure App Settings) - PRIORITY
    # We check directly in os.environ to avoid touching st.secrets if possible
    if key in os.environ:
        return os.environ[key]
    
    # 2. Check Streamlit Secrets (Local only)
    # FIX: Check if the file actually exists using OS before accessing st.secrets.
    # Accessing st.secrets when the file is missing triggers the Red UI Error immediately.
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
    
    .email-preview {
        background-color: #ffffff; border: 1px solid #d1d5db; padding: 20px;
        border-radius: 8px; font-family: Arial, sans-serif; color: #374151; white-space: pre-wrap;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: white; border-radius: 6px;
        border: 1px solid #E5E7EB; padding: 0 25px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF; border-color: #3B82F6; color: #1D4ED8;
    }

    .alert-banner {
        background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 20px;
        border-radius: 8px; color: #991B1B; font-weight: 600; margin-bottom: 15px;
    }
    
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

**DATA CONTEXT:**
* **12-Week Trend:** {trend_vals}
* **Latest Full Week Rate:** {last_full_val}
* **Depot Breakdown:** {depot_stats}
* **NOTE:** The final value in the trend list is likely an INCOMPLETE week (very low). **IGNORE IT** for the trend direction. Focus on the weeks prior.

**INSTRUCTIONS:**
1.  **Trend Diagnosis:** State if the trend (excluding the incomplete final week) is Stable, Increasing, or Decreasing.
2.  **Primary Contributor:** Identify which Depot is driving the numbers based on the breakdown.
3.  **Verdict:** If the rate is > 4.0, state "CRITICAL ATTENTION REQUIRED". If > 3.0, "Elevated Risk".
4.  **Tone:** Professional, direct, no fluff.
"""

PROMPT_WEEKLY_INSIGHT = """
You are an expert Bus Operations Analyst. Alarm type: {alarm_code}.
Using ONLY this JSON summary for a SINGLE week, produce insights.

DATA:
{payload}

Write markdown with EXACT sections:
### Executive Summary
### Actionable Anomaly Detection
- **Nexus of Risk:** Identify cross-category patterns (e.g. Driver X on Bus Y at Depot Z).
- **Category-Specific Insights:** Compare performance vs fleet average.
- **Low Workload, High Rate:** Identify drivers/buses with few trips but high alarms.
### Prioritized Recommendations
Return a 3-row Markdown Table: | Priority | Recommended Action | Data-Driven Rationale |
"""

PROMPT_4WEEK_DEEP = """
You are a world-class Bus Operations Analyst, an expert in diagnosing fleet performance trends.
Your task is to analyze the root cause of the 4-week trend for '{alarm_code}'.

{context_str}

**DATA:**
Below are the contributing entities (Drivers, Buses, Services) over this 4-week period.
The data shows their *total event counts* per week.

{data_json}

**YOUR MISSION (Write in Markdown):**
1. **Overall Trend Diagnosis:**
   * Start with a summary of the 4-week fleet trend.
   * State if this is an improvement, a consistent problem, or a new spike.

2. **Key Spike Drivers (Entities):**
   * Analyze the data to find the *root cause*.
   * Identify the *specific* Drivers, Buses, or Services that are the main cause.
   * Example: "The spike was driven by 3 specific drivers (IDs...)".

3. **Hidden Insights & Combinations (Most Important):**
   * **I. Driver -> Service:** Are top drivers concentrated on top services?
   * **II. Driver -> Bus -> Service:** Find the full nexus.
   * **III. Bus -> Service:** Are certain buses problematic only on specific routes?
   * Use bullet points.

4. **Actionable Summary:**
   * Provide a brief, prioritized summary of what to do next.
"""

PROMPT_ALERT_EMAIL = """
You are drafting a professional operational alert email in **HTML format**.

**SCENARIO:**
* **Metric:** {alarm}
* **Status:** Trending into **RED ZONE** (Projected: {projection}).
* **Explanation:** {trend_context}
* **REAL DATA - Top Offenders:** {offender_data}

**INSTRUCTIONS:**
1.  **Format:** Return **ONLY** the HTML code for the email body.
2.  **Situation:** Explicitly state the projected value and the % increase over the 4-week average.
3.  **Root Cause Analysis:** * Write 1-2 sentences summarizing *why* the spike is happening based on the offender data (e.g., "High frequency of alarms from Kranji Depot on Service 991").
    * **CRITICAL:** Create an HTML TABLE (`<table border="1" style="border-collapse: collapse; width: 100%;">`) populated **EXACTLY** with the "REAL DATA - Top Offenders" provided above. Do NOT invent data. Columns: Depot, Service, Bus, Driver, Count.
4.  **Action Required:** Bullet points for immediate next steps.
5.  **Sign-off:** "Best regards,<br>Data Analytics Team"

**Style:** Professional, urgent, clean layout. Use light gray background for table headers (`<th style="background-color: #f2f2f2; padding: 8px;">`).
"""

# --- 4. AI SETUP ---
@st.cache_resource
def initialize_llm():
    try:
        # Check if library is available
        if AzureChatOpenAI is None:
            return None
            
        endpoint = get_config("AZURE_ENDPOINT")
        api_key = get_config("OPENAI_API_KEY")
        deployment = get_config("AZURE_DEPLOYMENT")
        
        if not endpoint or not api_key:
            return None
            
        return AzureChatOpenAI(
            azure_endpoint=endpoint.rstrip("/"),
            openai_api_key=api_key,
            azure_deployment=deployment,
            api_version=get_config("AZURE_API_VERSION", "2024-02-15-preview"),
            temperature=0.2
        )
    except Exception as e:
        # Print error to logs but don't crash app
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
    """
    Smart Data Loader:
    1. Checks for Environment Variable (URL) first.
    2. Falls back to local file.
    3. Handles Blob Storage URLs gracefully.
    """
    
    # 1. GET CONFIG (Use the filename from your screenshot as default)
    # The default filenames are fallbacks if the Env Vars are missing
    path_tele = get_config("TELEMATICS_URL", "telematics_new_data_2207.csv")
    path_head = get_config("HEADCOUNTS_URL", "depot_headcounts.csv")
    path_excl = get_config("EXCLUSIONS_URL", "vehicle_exclusions.csv")

    # Helper: Read CSV from URL or Local
    def smart_read(path_val, is_required=False):
        if not path_val:
            return pd.DataFrame()
        
        path_str = str(path_val).strip()
        
        # A) Is it a URL? (Azure Blob with SAS Token)
        if path_str.lower().startswith("http"):
            try:
                # Pandas reads URLs directly
                return pd.read_csv(path_str)
            except Exception as e:
                if is_required:
                    st.error(f"❌ **Azure Connection Error**: Failed to download data from URL.\nError: {e}")
                return pd.DataFrame()
        
        # B) Is it a Local File?
        else:
            if os.path.exists(path_str):
                return pd.read_csv(path_str)
            else:
                if is_required:
                    # Specific error to help user debug
                    st.error(f"""
                    ❌ **Data Load Error**
                    
                    The app looked for: `{path_str}`
                    
                    1. **If using Azure Blob Storage:** The `TELEMATICS_URL` environment variable is NOT being read, so the app fell back to looking for a local file. Please check your App Service Configuration.
                    2. **If using Local Files:** The file is missing from the deployment zip.
                    """)
                return pd.DataFrame()

    # 2. LOAD
    df = smart_read(path_tele, is_required=True)
    headcounts = smart_read(path_head)
    exclusions = smart_read(path_excl)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 3. PRE-PROCESS
    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Clean string columns
    for c in ['bus_no', 'driver_id', 'alarm_type', 'depot_id', 'svc_no']:
        if c in df.columns: 
            df[c] = df[c].astype(str).str.strip().str.upper()
            df[c] = df[c].replace(['NAN', 'NULL', 'NONE', ''], None)
    
    # Process headcounts
    if not headcounts.empty and 'depot_id' in headcounts.columns:
        headcounts['depot_id'] = headcounts['depot_id'].astype(str).str.strip().str.upper()
    else:
        # Fallback empty df with correct columns
        headcounts = pd.DataFrame(columns=['depot_id', 'headcount'])

    # Process exclusions
    if not exclusions.empty and 'bus_no' in exclusions.columns:
        excl_list = set(exclusions['bus_no'].astype(str).str.strip().str.upper())
        df = df[~df['bus_no'].isin(excl_list)]

    # Parse Dates
    df['date'] = pd.to_datetime(df.get('alarm_calendar_date'), dayfirst=True, errors='coerce')
    df['year'] = df['date'].dt.isocalendar().year
    df['week'] = df['date'].dt.isocalendar().week
    
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
    
    # --- PAYLOADS ---
    latest_wk = weekly.iloc[-1]['week'] if not weekly.empty else 0
    df_curr = df_filtered[df_filtered['week'] == latest_wk]
    
    weekly_payload = {
        "week_number": int(latest_wk),
        "total_alarms": int(len(df_curr)),
        "depot_breakdown": df_curr['depot_id'].value_counts().to_dict(),
        "top_20_drivers": df_curr['driver_id'].value_counts().head(20).to_dict(),
        "top_20_buses": df_curr['bus_no'].value_counts().head(20).to_dict(),
        "top_20_services": df_curr['svc_no'].value_counts().head(20).to_dict(),
        "top_15_toxic_combinations": df_curr.groupby(['depot_id', 'svc_no', 'bus_no', 'driver_id']).size().sort_values(ascending=False).head(15).reset_index(name='count').to_dict(orient='records')
    }
    
    df_4wk = df_filtered[df_filtered['week'] >= (latest_wk - 3)]
    wk_4_payload = {
        "drivers": df_4wk.groupby(['driver_id', 'week']).size().unstack(fill_value=0).sum(axis=1).sort_values(ascending=False).head(20).to_dict(),
        "services": df_4wk.groupby(['svc_no', 'week']).size().unstack(fill_value=0).sum(axis=1).sort_values(ascending=False).head(20).to_dict(),
        "buses": df_4wk.groupby(['bus_no', 'week']).size().unstack(fill_value=0).sum(axis=1).sort_values(ascending=False).head(20).to_dict()
    }
    
    return df_filtered, weekly, weekly_payload, wk_4_payload, latest_wk

# --- 6. VISUALIZATIONS ---
def plot_trend_old_style(weekly_df, alarm_name):
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

def plot_prediction_bar(current, projected):
    fig = go.Figure()
    p_color = "#E74C3C" if projected > 5 else ("#F1C40F" if projected > 3 else "#2ECC71")
    fig.add_trace(go.Bar(
        x=["Current Status", "End-of-Week Projection"], y=[current, projected],
        marker_color=["#95A5A6", p_color], text=[f"{current:.2f}", f"{projected:.2f}"],
        textposition='auto', width=0.5
    ))
    fig.add_shape(type="line", x0=-0.5, x1=1.5, y0=5, y1=5, line=dict(color="red", width=2, dash="dash"))
    fig.update_layout(title="Risk Forecast", margin=dict(l=20, r=20, t=40, b=20), height=250, yaxis_range=[0, max(6, projected*1.2)])
    return fig

# --- 7. AUTOMATION ---
def trigger_power_automate(html_body, prediction):
    try:
        # Use flat environment variable
        url = get_config("POWER_AUTOMATE_FLOW_URL")
        
        if not url:
            st.error("Power Automate URL is missing in Environment Variables.")
            return False
            
        payload = {
            "subject": f"⚠️ ALERT: Rate Projected to {prediction:.2f} (Red Zone)", 
            "body": html_body, 
            "recipient": "devi02@smrt.com.sg"
        }
        res = requests.post(url, json=payload, timeout=6)
        return res.status_code == 202
    except Exception as e:
        st.error(f"Automation Error: {e}")
        return False

# --- 8. MAIN APP ---
def main():
    llm = initialize_llm()
    
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        alarm = st.selectbox("Alarm Type", list(ALARM_MAP.keys()))
        
        # Load Data with Smart Loader
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
    if weekly.empty: st.warning("No data found for these filters."); st.stop()
    
    latest_row = weekly.iloc[-1]
    # Simple projection logic
    projected_val = (latest_row['per_bc'] / (df_filtered['date'].max().weekday() + 1)) * 7

    # =========================================================
    # SECTION 1: 12-WEEK TREND
    # =========================================================
    st.markdown(f"## {ALARM_MAP[alarm]['long']} Intelligence Hub")
    
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col_chart, col_text = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(plot_trend_old_style(weekly.tail(12), alarm), use_container_width=True)
        with col_text:
            st.markdown("### 📋 Executive Brief")
            if llm:
                if "exec_brief" not in st.session_state:
                    with st.spinner("Analyzing high-level trend..."):
                        trend_list = weekly.tail(12)['per_bc'].tolist()
                        last_full_val = trend_list[-2] if len(trend_list) > 1 else trend_list[-1]
                        
                        prompt = PROMPT_EXEC_BRIEF.format(
                            trend_vals=trend_list,
                            last_full_val=last_full_val,
                            depot_stats=wk_payload['depot_breakdown']
                        )
                        st.session_state.exec_brief = llm.predict(prompt)
                st.write(st.session_state.exec_brief)
            else:
                st.info("AI Not Connected. Check Azure Keys.")
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # SECTION 2: OPERATIONS COMMAND CENTER
    # =========================================================
    st.markdown("### Operations Center")
    t1, t2, t3 = st.tabs(["📊 Latest Week Deep Dive", "🧠 4-Week Pattern Scan", "🔮 Risk Forecast & Alert"])
    
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
                    with st.spinner("Analyzing full week dataset..."):
                        st.markdown(llm.predict(prompt))
            else:
                st.write("AI needed.")
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_4wk = weekly.tail(4)['per_bc'].mean()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("4-Week Baseline", f"{avg_4wk:.2f}")
            st.caption("Average rate over last month")
        with c2:
            st.markdown("#### 🧠 Systemic Trend Analysis")
            if llm:
                json_str = json.dumps(wk4_payload, indent=2, cls=NpEncoder)
                context_str = f"Current Rate: {latest_row['per_bc']}. 4-Week Avg: {avg_4wk}."
                prompt = PROMPT_4WEEK_DEEP.format(alarm_code=alarm, context_str=context_str, data_json=json_str)
                if st.button("Run Systemic Scan"):
                    with st.spinner("Scanning 4-week trends..."):
                        st.markdown(llm.predict(prompt))
        st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        c_pred_graph, c_pred_action = st.columns(2)
        with c_pred_graph:
            st.plotly_chart(plot_prediction_bar(latest_row['per_bc'], projected_val), use_container_width=True)
        with c_pred_action:
            st.markdown("#### ⚠️ Operational Risk Status")
            if projected_val > 4.0: 
                st.markdown(f"""<div class="alert-banner">🚨 <b>CRITICAL RISK</b><br>Projection: <b>{projected_val:.2f}</b> (Red Zone).</div>""", unsafe_allow_html=True)
                if llm:
                    avg_4 = weekly.tail(4)['per_bc'].mean()
                    trend_txt = f"This projection is {((projected_val - avg_4)/avg_4)*100:.1f}% higher than the 4-week average ({avg_4:.2f})"
                    offender_list = wk_payload['top_15_toxic_combinations'][:5]
                    offender_str = "\n".join([f"- {i['depot_id']} | Svc {i['svc_no']} | Bus {i['bus_no']} | Driver {i['driver_id']} : {i['count']} alarms" for i in offender_list])
                    
                    if "email_html_draft" not in st.session_state:
                        st.session_state.email_html_draft = llm.predict(PROMPT_ALERT_EMAIL.format(
                            alarm=alarm, projection=f"{projected_val:.2f}", trend_context=trend_txt, offender_data=offender_str))
                    
                    st.write("**Draft Email:**")
                    st.components.v1.html(st.session_state.email_html_draft, height=400, scrolling=True)
                    if st.button("📨 Send Email", type="primary"):
                        with st.spinner("Sending..."):
                            success = trigger_power_automate(st.session_state.email_html_draft, projected_val)
                            if success: st.success("Email Sent!")
                            else: st.error("Failed to send.")
            else:
                st.success(f"Projected Rate: {projected_val:.2f}. Status: Safe.")
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
