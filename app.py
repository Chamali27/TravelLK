"""
app.py  —  TravelLK · AI Travel Planning Agent for Sri Lanka
KDU BSc Applied Data Science Communication · LB3114 Assignment II
"""

import re
import datetime
import streamlit as st
import base64
from pathlib import Path

from agent import (
    plan_trip, refine_trip, chat_with_agent,
    get_weather, extract_place_names, get_place_locations,
    decide_travel_style, check_goal_achievement,
)
from memory import (
    save_trip, get_recent_trips, get_total_trips,
    get_memory_context,
)

# PAGE CONFIG
st.set_page_config(
    page_title="TravelLK · AI Travel Agent",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# STYLES
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:#0d1f17; --bg2:#122a1e; --panel:#1a3528; --panel2:#1f3d2f;
  --border:rgba(255,255,255,0.08); --border2:rgba(255,255,255,0.14);
  --green:#3dba7e; --gold:#e8b84b;
  --text:#e8f0ec; --text2:#a8bfb3; --muted:#5a7a6a;
}

html,body,[class*="css"]{ font-family:'DM Sans',sans-serif!important; background:var(--bg)!important; color:var(--text)!important; }
.stApp{ background:var(--bg)!important; }
#MainMenu,footer{ visibility:hidden; }
header{ visibility:visible!important; background:transparent!important; }
.block-container{ padding:0 1.6rem 2rem!important; max-width:100%!important; }

/* HERO */
.hero-wrap{ width:100%; height:220px; position:relative; overflow:hidden; border-radius:0 0 24px 24px; margin-bottom:20px; }
.hero-img{ width:100%; height:100%; object-fit:cover; display:block; }
.hero-overlay{ position:absolute; inset:0; background:linear-gradient(105deg,rgba(5,18,11,0.92) 0%,rgba(5,18,11,0.6) 55%,rgba(5,18,11,0.15) 100%); }
.hero-content{ position:absolute; inset:0; display:flex; align-items:center; padding:0 48px; gap:20px; }
.hero-badge{ font-size:3rem; filter:drop-shadow(0 2px 8px rgba(0,0,0,0.5)); }
.hero-text h1{ font-family:'Cormorant Garamond',serif; font-size:2.8rem; font-weight:600; color:#fff; margin:0; text-shadow:0 2px 12px rgba(0,0,0,0.4); }
.hero-text p{ font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:2.5px; text-transform:uppercase; color:rgba(255,255,255,0.6); margin:10px 0 0; }
.hero-stats{ margin-left:auto; display:flex; gap:16px; }
.hero-stat{ text-align:center; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14); border-radius:9px; padding:10px 18px; }
.hero-stat-num{ font-family:'Cormorant Garamond',serif; font-size:1.6rem; font-weight:600; color:#3dba7e; line-height:1; }
.hero-stat-lbl{ font-family:'DM Mono',monospace; font-size:0.55rem; letter-spacing:1.2px; text-transform:uppercase; color:rgba(255,255,255,0.5); margin-top:4px; }

/* SIDEBAR */
[data-testid="stSidebar"]{ background:#0a1912!important; border-right:1px solid rgba(255,255,255,0.08)!important; }
[data-testid="stSidebar"] .block-container{ padding:1.2rem 1rem 2rem!important; }
[data-testid="collapsedControl"]{ display:flex!important; visibility:visible!important; opacity:1!important;
  background:#1a3528!important; border:1px solid rgba(61,186,126,0.3)!important; border-left:none!important;
  border-radius:0 8px 8px 0!important; position:fixed!important; top:50%!important; left:0!important;
  z-index:9999!important; width:24px!important; height:48px!important; cursor:pointer!important; }
[data-testid="collapsedControl"] svg{ fill:#3dba7e!important; stroke:#3dba7e!important; }

/* TEXT */
label,p,span,div,li,.stMarkdown p,.stMarkdown span,
div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] span { color:var(--text)!important; }
div[data-testid="stSlider"] p{ color:var(--text2)!important; }

/* INPUTS */
.stTextInput input,.stTextArea textarea{
  background:#1f3d2f!important; border:1.5px solid rgba(255,255,255,0.14)!important;
  color:#e8f0ec!important; border-radius:9px!important; font-size:0.87rem!important; }
.stTextInput input::placeholder,.stTextArea textarea::placeholder{ color:#5a7a6a!important; }
.stTextInput input:focus,.stTextArea textarea:focus{ border-color:#3dba7e!important; box-shadow:0 0 0 3px rgba(61,186,126,0.15)!important; }

/* RADIO */
.stRadio>div{ display:flex; flex-direction:column; gap:6px; }
.stRadio div[role="radiogroup"]>label{
  background:#1a3528!important; border:1.5px solid rgba(255,255,255,0.12)!important;
  border-radius:7px!important; padding:8px 14px!important; color:#a8bfb3!important;
  font-size:0.84rem!important; cursor:pointer!important; }
.stRadio div[role="radiogroup"]>label:has(input:checked){
  border-color:#e8b84b!important; background:rgba(232,184,75,0.1)!important; color:#e8b84b!important; font-weight:600!important; }

/* MAIN BUTTON */
.stButton>button{
  background:linear-gradient(135deg,#3dba7e,#2a9062)!important; color:#fff!important;
  border:none!important; border-radius:14px!important; font-weight:600!important;
  font-size:0.9rem!important; padding:12px 24px!important; width:100%;
  box-shadow:0 4px 18px rgba(61,186,126,0.28)!important; transition:all 0.2s!important; }
.stButton>button:hover{
  background:linear-gradient(135deg,#48cc8c,#33a870)!important;
  box-shadow:0 6px 26px rgba(61,186,126,0.4)!important; transform:translateY(-1px)!important; }

/* SIDEBAR BUTTON */
[data-testid="stSidebar"] .stButton>button{
  background:rgba(61,186,126,0.1)!important; color:#3dba7e!important;
  border:1px solid rgba(61,186,126,0.25)!important; box-shadow:none!important;
  font-size:0.78rem!important; padding:6px 10px!important; }
[data-testid="stSidebar"] .stButton>button:hover{ background:rgba(61,186,126,0.18)!important; transform:none!important; }

/* DOWNLOAD BUTTON */
[data-testid="stDownloadButton"]>button{
  background:rgba(61,186,126,0.12)!important; color:#3dba7e!important;
  border:1.5px solid rgba(61,186,126,0.35)!important; border-radius:10px!important;
  font-weight:600!important; font-size:0.86rem!important; padding:10px 20px!important; width:100%; box-shadow:none!important; }
[data-testid="stDownloadButton"]>button:hover{
  background:rgba(61,186,126,0.22)!important; border-color:#3dba7e!important; transform:translateY(-1px)!important; }

/* EXPANDER */
.streamlit-expanderHeader{ background:#1a3528!important; border:1px solid rgba(255,255,255,0.1)!important; border-radius:9px!important; color:#a8bfb3!important; }
details[open] .streamlit-expanderHeader{ border-radius:9px 9px 0 0!important; }
.streamlit-expanderContent{ background:#1f3d2f!important; border:1px solid rgba(255,255,255,0.08)!important; border-top:none!important; border-radius:0 0 9px 9px!important; }
[data-testid="stExpander"]{ background:#1a3528!important; border:1px solid rgba(255,255,255,0.1)!important; border-radius:9px!important; }
[data-testid="stExpander"] summary{ background:#1a3528!important; color:#a8bfb3!important; }
[data-testid="stExpander"] details{ background:#1f3d2f!important; }

/* SELECTBOX */
.stSelectbox>div>div{ background:#1f3d2f!important; border:1.5px solid rgba(255,255,255,0.14)!important; color:#e8f0ec!important; border-radius:9px!important; }
[data-baseweb="popover"],[data-baseweb="menu"]{ background:#1a3528!important; }
[data-baseweb="option"]{ color:#e8f0ec!important; background:#1a3528!important; }
[data-baseweb="option"]:hover{ background:#1f3d2f!important; }

/* ARRIVAL CARD BUTTONS */
.arr-btn-wrap > div[data-testid="stButton"] > button {
  background: #1a3528 !important;
  border: 1.5px solid rgba(255,255,255,0.1) !important;
  border-radius: 12px !important;
  padding: 10px 4px 8px !important;
  width: 100% !important;
  height: 86px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 3px !important;
  color: #a8bfb3 !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  box-shadow: none !important;
  transform: none !important;
  transition: border-color 0.15s, background 0.15s !important;
  white-space: pre-line !important;
  line-height: 1.4 !important;
  text-align: center !important;
}
.arr-btn-wrap > div[data-testid="stButton"] > button:hover {
  border-color: rgba(61,186,126,0.45) !important;
  background: #1f3d2f !important;
  transform: none !important;
  box-shadow: none !important;
}
.arr-btn-wrap.active > div[data-testid="stButton"] > button {
  border-color: #3dba7e !important;
  background: rgba(61,186,126,0.15) !important;
  box-shadow: 0 0 0 3px rgba(61,186,126,0.25) !important;
  color: #3dba7e !important;
  font-weight: 700 !important;
}

/* INTEREST IMAGE CARD BUTTONS */
.img-btn-wrap > div[data-testid="stButton"] > button {
  background: transparent !important;
  border: 2px solid transparent !important;
  border-radius: 12px !important;

  width: 20px !important;   /* reduce width */
  height: 20px !important;  /* reduce height */

  padding: 0 !important;
  overflow: hidden !important;

  box-shadow: none !important;
  transition: border-color 0.18s, box-shadow 0.18s !important;

  display: block !important;
  cursor: pointer !important;
}
.img-btn-wrap > div[data-testid="stButton"] > button:hover {
  transform: none !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
  border-color: rgba(61,186,126,0.4) !important;
}
.img-btn-wrap.sel > div[data-testid="stButton"] > button {
  border-color: #3dba7e !important;
  box-shadow: 0 0 0 2px rgba(61,186,126,0.4) !important;
}

/* CHIPS */
.chip{ font-family:'DM Mono',monospace; font-size:0.62rem; padding:3px 10px; border-radius:20px; display:inline-block; }
.chip-g{ background:rgba(61,186,126,0.12); color:#3dba7e; border:1px solid rgba(61,186,126,0.25); }
.chip-y{ background:rgba(232,184,75,0.12); color:#e8b84b; border:1px solid rgba(232,184,75,0.25); }
.chip-m{ background:rgba(255,255,255,0.05); color:#a8bfb3; border:1px solid rgba(255,255,255,0.1); }
.chip-row{ display:flex; gap:5px; flex-wrap:wrap; margin-top:10px; }
.goal-pass{ background:rgba(61,186,126,0.12); color:#3dba7e; border:1px solid rgba(61,186,126,0.25); font-family:'DM Mono',monospace; font-size:0.62rem; padding:3px 10px; border-radius:20px; display:inline-block; }
.goal-fail{ background:rgba(220,80,60,0.12); color:#e87060; border:1px solid rgba(220,80,60,0.25); font-family:'DM Mono',monospace; font-size:0.62rem; padding:3px 10px; border-radius:20px; display:inline-block; }

/* MISC */
.mono-label{ font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:1.5px; text-transform:uppercase; color:#3dba7e; margin-bottom:8px; display:block; }
.sec-title{ font-family:'Cormorant Garamond',serif; font-size:1.4rem; font-weight:600; color:#e8f0ec; margin:0 0 2px; }
.sec-sub{ font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:1.5px; text-transform:uppercase; color:#5a7a6a; margin-bottom:16px; display:block; }
.divider{ border:none; border-top:1px solid rgba(255,255,255,0.08); margin:18px 0; }
.ai-box{ background:rgba(61,186,126,0.07); border:1px solid rgba(61,186,126,0.2); border-left:3px solid #3dba7e; border-radius:9px; padding:11px 15px; font-size:0.82rem; color:#a8bfb3; margin-top:10px; }
.ai-box strong{ color:#3dba7e; }
.hist-card{ background:#1f3d2f; border:1px solid rgba(255,255,255,0.07); border-radius:9px; padding:10px 14px; margin-bottom:8px; font-size:0.77rem; line-height:1.65; color:#a8bfb3; }
.hist-card b{ color:#3dba7e; }
.status-item{ background:#1a3528; border:1px solid rgba(255,255,255,0.08); border-radius:9px; padding:8px 12px; text-align:center; font-family:'DM Mono',monospace; font-size:0.64rem; }
.status-on{ color:#3dba7e; } .status-off{ color:#5a7a6a; }

/* ARRIVAL ADVICE */
.arrival-advice{ border-radius:8px; padding:10px 13px; font-size:0.76rem; line-height:1.6; border-left:3px solid; margin-top:10px; }
.advice-morning{ background:rgba(232,184,75,0.08); border-color:#e8b84b; color:#c9a03e; }
.advice-afternoon{ background:rgba(61,186,126,0.08); border-color:#3dba7e; color:#3dba7e; }
.advice-evening{ background:rgba(100,160,220,0.08); border-color:#6490cc; color:#7aaae8; }
.advice-night{ background:rgba(160,100,220,0.08); border-color:#9b6fd4; color:#b08ae0; }

/* ITINERARY INNER */
#itin-inner h1,#itin-inner h2{ font-family:'Cormorant Garamond',serif; color:#3dba7e; border-bottom:1px solid rgba(61,186,126,0.2); padding-bottom:4px; margin:16px 0 6px; }
#itin-inner h1{ font-size:1.4rem; } #itin-inner h2{ font-size:1.2rem; }
#itin-inner h3{ color:#e8b84b; font-size:1rem; margin:10px 0 3px; }
#itin-inner p,#itin-inner li{ font-size:0.94rem; line-height:2.0; color:#e8f0ec; }
#itin-inner strong{ color:#e8b84b; }
#itin-inner hr{ border:none; border-top:1px solid rgba(255,255,255,0.08); margin:12px 0; }
#itin-inner ul{ padding-left:1.2rem; margin:4px 0; }

/* CHAT BUBBLES */
.chat-header-bar{ background:linear-gradient(135deg,#1a3528,#162d22); border:1px solid rgba(61,186,126,0.2); border-radius:14px 14px 0 0; padding:14px 20px; display:flex; align-items:center; gap:10px; }
.chat-hicon{ width:34px; height:34px; background:linear-gradient(135deg,#3dba7e,#2a9062); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
.chat-htitle{ font-family:'Cormorant Garamond',serif; font-size:1.05rem; font-weight:600; color:#e8f0ec; }
.chat-hsub{ font-family:'DM Mono',monospace; font-size:0.54rem; letter-spacing:1px; text-transform:uppercase; color:#5a7a6a; }
.chat-online{ width:7px; height:7px; background:#3dba7e; border-radius:50%; box-shadow:0 0 6px #3dba7e; margin-left:auto; }

.bubble-user-row{ display:flex; justify-content:flex-end; margin-bottom:12px; }
.bubble-user{ background:linear-gradient(135deg,rgba(61,186,126,0.2),rgba(42,144,98,0.14)); border:1px solid rgba(61,186,126,0.3); border-radius:14px 14px 4px 14px; padding:10px 14px; max-width:72%; color:#e8f0ec; font-size:0.86rem; line-height:1.7; }
.bubble-user-meta{ font-family:'DM Mono',monospace; font-size:0.52rem; color:#5a7a6a; text-align:right; margin-top:3px; }

.bubble-agent-row{ display:flex; justify-content:flex-start; align-items:flex-start; gap:8px; margin-bottom:12px; }
.chat-avatar{ width:28px; height:28px; background:linear-gradient(135deg,#3dba7e,#2a9062); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.8rem; flex-shrink:0; margin-top:2px; }
.bubble-agent{ background:#1f3d2f; border:1px solid rgba(255,255,255,0.09); border-radius:4px 14px 14px 14px; padding:10px 14px; max-width:80%; color:#e8f0ec; font-size:0.86rem; line-height:1.8; }
.bubble-agent-meta{ font-family:'DM Mono',monospace; font-size:0.52rem; color:#5a7a6a; margin-top:3px; }

.sug-row{ display:flex; gap:6px; flex-wrap:wrap; padding:10px 18px 8px; background:#0f1e15; border:1px solid rgba(61,186,126,0.15); border-top:none; }
.sug-chip{ font-size:0.68rem; background:rgba(61,186,126,0.08); border:1px solid rgba(61,186,126,0.22); border-radius:20px; padding:4px 11px; color:#3dba7e; white-space:nowrap; }
.sug-label{ font-family:'DM Mono',monospace; font-size:0.54rem; letter-spacing:1px; text-transform:uppercase; color:#5a7a6a; width:100%; }

.chat-empty{ background:#111f18; border:1px solid rgba(61,186,126,0.15); border-top:none; padding:28px; text-align:center; border-radius:0 0 14px 14px; }

::-webkit-scrollbar{ width:5px; }
::-webkit-scrollbar-track{ background:#0d1f17; }
::-webkit-scrollbar-thumb{ background:#1f3d2f; border-radius:4px; }
        
/* BUDGET BUTTONS — full green when active */
.budget-btn-wrap.active > div[data-testid="stButton"] > button {
  border-color: #3dba7e !important;
  background: #3dba7e !important;
  box-shadow: 0 0 0 3px rgba(61,186,126,0.3) !important;
  color: #fff !important;
  font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# SESSION STATE
for k, v in {
    "itinerary":"", "chat_messages":[], "chat_history":[],
    "weather_city":"Colombo", "weather_data":None,
    "goal_eval":None, "place_names":[], "generated":False,
    "interests_set":set(), "arrival_time":"morning",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# INTEREST DATA
def img_to_b64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

INTEREST_PHOTOS = {
    "Beaches":            img_to_b64("images/beach.jpg"),
    "Hiking":             img_to_b64("images/hiking.jpg"),
    "Nature":             img_to_b64("images/nature.jpg"),
    "Photography":        img_to_b64("images/photography.jpg"),
    "History & Culture":  img_to_b64("images/history.jpg"),
    "Wildlife":           img_to_b64("images/wildlife.jpg"),
    "Food & Cuisine":     img_to_b64("images/food.jpg"),
    "Relaxation":         img_to_b64("images/relaxation.jpg"),
}
INTEREST_EMOJIS = {
    "Beaches":"🏖️","Hiking":"🥾","Nature":"🦁",
    "Photography":"📷","History & Culture":"🏛️","Wildlife":"🐘",
    "Food & Cuisine":"🍛","Relaxation":"🧘",
}

# ARRIVAL TIME CONFIG
ARRIVAL_OPTIONS = {
    "morning": {
        "icon":   "🌅",
        "label":  "Morning",
        "time":   "Before 12 pm",
        "advice": "✈️ Great timing! Day 1 can head straight to any destination — Mirissa, Sigiriya, Ella. No need to stop near the airport.",
        "cls":    "advice-morning",
    },
    "afternoon": {
        "icon":   "☀️",
        "label":  "Afternoon",
        "time":   "12 pm – 6 pm",
        "advice": "🏨 Not enough time to travel far. Day 1 will be a relaxed arrival in Negombo (15 min from airport). Journey begins Day 2.",
        "cls":    "advice-afternoon",
    },
    "evening": {
        "icon":   "🌆",
        "label":  "Evening",
        "time":   "6 pm – 10 pm",
        "advice": "🌙 Too late for long travel. Day 1 is a short transfer to Negombo — check in and rest. Exploring starts Day 2.",
        "cls":    "advice-evening",
    },
    "night": {
        "icon":   "🌙",
        "label":  "Night",
        "time":   "After 10 pm",
        "advice": "😴 Straight to bed! Day 1 is a pure rest night in Negombo or Katunayake. The adventure begins properly on Day 2.",
        "cls":    "advice-night",
    },
}

# HERO
st.markdown("""
<div class="hero-wrap">
  <img class="hero-img" src="https://images.unsplash.com/photo-1564501049412-61c2a3083791?w=1800&auto=format&fit=crop&q=85" alt="Sri Lanka">
  <div class="hero-overlay"></div>
  <div class="hero-content">
    <div class="hero-badge">🌴</div>
    <div class="hero-text"><h1>TravelLK</h1><p>AI-Powered Travel Planning Agent &nbsp;·&nbsp; Sri Lanka</p></div>
    <div class="hero-stats">
      <div class="hero-stat"><div class="hero-stat-num">330+</div><div class="hero-stat-lbl">destinations</div></div>
      <div class="hero-stat"><div class="hero-stat-num">AI</div><div class="hero-stat-lbl">powered</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:8px 0 16px;">
      <div style="font-family:'Cormorant Garamond',serif;font-size:1.6rem;color:#3dba7e;font-weight:600;">🌴 TravelLK</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#5a7a6a;margin-top:4px;">{get_total_trips()} trips planned</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<span class="mono-label">🌤 Live Weather</span>', unsafe_allow_html=True)
    weather_input = st.text_input("city_search", value=st.session_state.weather_city,
        placeholder="Kandy, Galle, Ella...", label_visibility="collapsed")
    if st.button("Search Weather", key="btn_weather"):
        if weather_input.strip():
            st.session_state.weather_city = weather_input.strip()
            st.session_state.weather_data = get_weather(weather_input.strip())

    if st.session_state.weather_data is None:
        st.session_state.weather_data = get_weather(st.session_state.weather_city)

    wd = st.session_state.weather_data
    if wd and wd.get("success"):
        icon_map = {"Clear":"☀️","Clouds":"⛅","Rain":"🌧️","Drizzle":"🌦️","Thunderstorm":"⛈️","Snow":"❄️","Mist":"🌫️","Haze":"🌫️","Fog":"🌫️"}
        icon = icon_map.get(wd.get("icon",""), "🌡️")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1e4d35,#0f2a1e);border:1px solid rgba(61,186,126,0.2);border-radius:14px;padding:16px 20px;margin-bottom:14px;">
          <p style="font-family:'Cormorant Garamond',serif;font-size:1.1rem;font-weight:600;color:#e8f0ec;margin:0;">{icon} {wd['city']}, {wd['country']}</p>
          <p style="font-family:'Cormorant Garamond',serif;font-size:2.4rem;font-weight:400;color:#3dba7e;line-height:1;margin:4px 0;">{wd['temp']}°C</p>
          <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#a8bfb3;margin:0;">{wd['description']} · feels {wd['feels_like']}°C</p>
          <div style="display:flex;gap:14px;margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);">
            <div style="font-size:0.72rem;color:#5a7a6a;">Humidity<br><span style="color:#e8f0ec;font-weight:500;">{wd['humidity']}%</span></div>
            <div style="font-size:0.72rem;color:#5a7a6a;">Wind<br><span style="color:#e8f0ec;font-weight:500;">{wd['wind']} m/s</span></div>
          </div>
        </div>""", unsafe_allow_html=True)
    elif wd and not wd.get("success"):
        st.warning(f"⚠️ {wd.get('error','City not found')}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<span class="mono-label">🕐 Past Trips</span>', unsafe_allow_html=True)
    recent = get_recent_trips(15)
    if not recent:
        st.caption("No trips yet — plan your first one!")
    else:
        for i, (days_r, interests_r, budget_r, timestamp_r, itinerary_r) in enumerate(recent):
            date_str = timestamp_r[:10] if timestamp_r else ""
            st.markdown(f"""<div class="hist-card"><b>{days_r} days · {budget_r.split()[0]}</b><br>
              {interests_r[:55]}{"..." if len(interests_r)>55 else ""}<br>
              <span style="font-size:0.67rem;color:#5a7a6a;">{date_str}</span></div>""", unsafe_allow_html=True)
            if st.button(f"↩ Load trip {i+1}", key=f"reload_{i}"):
                st.session_state.itinerary   = itinerary_r
                st.session_state.generated   = True
                st.session_state.place_names = extract_place_names(itinerary_r)
                st.session_state.goal_eval   = check_goal_achievement(itinerary_r)
                st.session_state.chat_messages = []
                st.session_state.chat_history  = []
                st.rerun()

# FORM — Left + Right columns
form_left, form_right = st.columns([5, 7], gap="large")

# LEFT COLUMN
with form_left:
    st.markdown('<div class="sec-title">Plan Your Trip</div>', unsafe_allow_html=True)
    st.markdown('<span class="sec-sub">Tell the AI agent about your journey</span>', unsafe_allow_html=True)

    # ── Days slider ────────────────────────────
    st.markdown('<span style="font-size:0.82rem;color:#a8bfb3;font-weight:500;">📅 How many days?</span>', unsafe_allow_html=True)
    days = st.slider("days", 1, 21, 7, label_visibility="collapsed")

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # ── Budget card ────────────────────────────
    st.markdown("""
    <div style="margin-bottom:8px;margin-top:6px;">
      <span style="font-size:1rem;color:#e8f0ec;font-weight:600;">💰 Your Budget?</span>
    </div>
    """, unsafe_allow_html=True)

    BUDGET_OPTIONS = {
        "Budget":   {"icon": "🎒", "sub": "Under USD 50/day",  "key": "Budget (Under USD 50/day)"},
        "Mid-range":{"icon": "✈️", "sub": "USD 50–150/day",    "key": "Mid-range (USD 50-150/day)"},
        "Luxury":   {"icon": "💎", "sub": "USD 150+/day",      "key": "Luxury (USD 150+/day)"},
    }

    if "budget_choice" not in st.session_state:
        st.session_state.budget_choice = "Budget (Under USD 50/day)"

    budget_cols = st.columns(3, gap="small")
    for col, (label, opt) in zip(budget_cols, BUDGET_OPTIONS.items()):
        is_active = st.session_state.budget_choice == opt["key"]
        active_cls = "active" if is_active else ""
        with col:
            is_active = st.session_state.budget_choice == opt["key"]
            bg = "#3dba7e" if is_active else "#1a3528"
            color = "#fff" if is_active else "#a8bfb3"
            border = "2px solid #3dba7e" if is_active else "1px solid rgba(255,255,255,0.1)"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:12px;padding:14px 6px;
                        text-align:center;height:86px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;gap:3px;margin-bottom:4px;">
              <div style="font-size:1.8rem;">{opt['icon']}</div>
              <div style="font-size:1rem;color:{color};opacity:0.85;">{opt['sub']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"{label}", key=f"budget_{label}", use_container_width=True):
                st.session_state.budget_choice = opt["key"]
                st.rerun()

    budget = st.session_state.budget_choice
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

# ── Arrival Time ──────────────────────
    st.markdown("""
    <div style="margin-bottom:10px;margin-top:6px;">
      <span style="font-size:0.88rem;color:#e8f0ec;font-weight:600;">✈️ When do you arrive?</span><br>
      <span style="font-size:0.71rem;color:#5a7a6a;">Your arrival time affects Day 1 planning</span>
    </div>
    """, unsafe_allow_html=True)
    selected_arrival = st.session_state.arrival_time
    arr_keys = list(ARRIVAL_OPTIONS.keys())

# Row 1: Morning, Afternoon
    row1_cols = st.columns(2, gap="small")
    for col, key in zip(row1_cols, arr_keys[:2]):
        opt = ARRIVAL_OPTIONS[key]
        is_active = selected_arrival == key
        active_cls = "active" if is_active else ""
        with col:
            st.markdown(f'<div class="arr-btn-wrap {active_cls}">', unsafe_allow_html=True)
            # ✅ tick shows when selected
            tick = "✅" if is_active else opt['icon']
            btn_label = f"{tick}\n{opt['label']}\n{opt['time']}"
            if st.button(btn_label, key=f"arr_{key}", use_container_width=True):
                st.session_state.arrival_time = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


    # Row 2: Evening, Night
    row2_cols = st.columns(2, gap="small")
    for col, key in zip(row2_cols, arr_keys[2:]):
        opt = ARRIVAL_OPTIONS[key]
        is_active = selected_arrival == key
        active_cls = "active" if is_active else ""
        with col:
            st.markdown(f'<div class="arr-btn-wrap {active_cls}">', unsafe_allow_html=True)
            # ✅ tick shows when selected
            tick = "✅" if is_active else opt['icon']
            btn_label = f"{tick}\n{opt['label']}\n{opt['time']}"
            if st.button(btn_label, key=f"arr_{key}", use_container_width=True):
                st.session_state.arrival_time = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Advice box
    opt = ARRIVAL_OPTIONS[selected_arrival]
    st.markdown(f"""
      <div class="arrival-advice {opt['cls']}">
        {opt['advice']}
      </div>
    """, unsafe_allow_html=True)

    # ── Extra info ─────────────────────────────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<span style="font-size:0.82rem;color:#a8bfb3;font-weight:500;">💬 Anything extra?</span>', unsafe_allow_html=True)
    extra_info = st.text_area("extra",
        placeholder="e.g. travelling with family, honeymoon, starting from Colombo...",
        height=90, label_visibility="collapsed")


st.markdown("""
<style>
/* Hide default button styling */
div.stButton > button {
    background: transparent !important;
    border: none !important;
    color: #e8f0ec !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    text-align: center;
    padding: 6px 0px;
}

/* Remove hover green effect */
div.stButton > button:hover {
    background: transparent !important;
    color: #ffffff !important;
}

/* Remove focus outline */
div.stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# RIGHT COLUMN
with form_right:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
      <span style="font-size:1.1rem;">🎯</span>
      <span style="font-family:'Cormorant Garamond',serif;font-size:1.4rem;font-weight:600;color:#e8f0ec;">What are your interests?</span>
    </div>
    <span style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:1.5px;
                 text-transform:uppercase;color:#5a7a6a;display:block;margin-bottom:14px;">
      Click a card to select · click again to deselect
    </span>
    """, unsafe_allow_html=True)

    interest_labels = list(INTEREST_PHOTOS.keys())
    rows = [interest_labels[i:i+4] for i in range(0, len(interest_labels), 4)]

    for row in rows:
        cols = st.columns(4, gap="small")
        for col, label in zip(cols, row):
            with col:
                selected  = label in st.session_state.interests_set
                emoji     = INTEREST_EMOJIS[label]
                img_url   = INTEREST_PHOTOS[label]
                border_c  = "#cad1ce" if selected else "rgba(255,255,255,0.1)"
                bg_c      = "rgba(61,186,126,0.12)" if selected else "#1a3528"
                tick      = "✓ " if selected else ""

               
                st.markdown(f"""
                <div style="border:2px solid {border_c};border-radius:10px;
                            overflow:hidden;background:{bg_c};
                            transition:border-color 0.15s;">
                  <img src="{img_url}"
                       style="width:100%;height:200px;object-fit:cover;display:block;
                              opacity:{'0.95' if selected else '0.7'};" />
                </div>
                """, unsafe_allow_html=True)

                btn_text = f"{tick}{emoji} {label}"
                if st.button(btn_text, key=f"int_{label}", use_container_width=True):
                    if label in st.session_state.interests_set:
                        st.session_state.interests_set.discard(label)
                    else:
                        st.session_state.interests_set.add(label)
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    # Selected chips + AI style decision
    interests_selected = list(st.session_state.interests_set)
    if interests_selected:
        chips = " ".join(
            f'<span class="chip chip-g">{INTEREST_EMOJIS[i]} {i}</span>'
            for i in interests_selected
        )
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)
        style_dec = decide_travel_style(interests_selected, budget)
        st.markdown(f"""
        <div class="ai-box">
          <strong>🤖 AI Decision:</strong> {style_dec['stay']} · {style_dec['pace']} pace
          · {style_dec['cost_level']} cost · {style_dec['focus']}
        </div>""", unsafe_allow_html=True)


# GENERATE BUTTON
interests_selected = list(st.session_state.interests_set)

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
if st.button("✈️  Generate My Itinerary", use_container_width=True, key="btn_generate"):
    if not interests_selected:
        st.warning("Please select at least one interest.")
    else:
        memory_ctx    = get_memory_context(3)
        interests_str = ", ".join(interests_selected)
        with st.spinner("🌴 Planning your perfect Sri Lanka trip..."):
            itinerary, goal_eval = plan_trip(
                days, interests_selected, budget,
                arrival_time=st.session_state.arrival_time,
                extra_info=extra_info,
                memory_context=memory_ctx,
            )
        st.session_state.itinerary     = itinerary
        st.session_state.goal_eval     = goal_eval
        st.session_state.generated     = True
        st.session_state.place_names   = extract_place_names(itinerary)
        st.session_state.chat_messages = []
        st.session_state.chat_history  = []
        save_trip(days, interests_str, budget, itinerary)
        st.rerun()

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# RESULTS
if not st.session_state.generated:
    st.markdown("""
    <div style="text-align:center;padding:70px 20px 0;color:#5a7a6a;">
      <div style="font-size:4rem;margin-bottom:18px;">🗺️</div>
      <div style="font-family:'Cormorant Garamond',serif;font-size:1.7rem;color:#a8bfb3;margin-bottom:10px;">Your itinerary will appear here</div>
      <div style="font-size:0.84rem;line-height:2;color:#5a7a6a;">Select interests above and hit Generate to begin.</div>
    </div>""", unsafe_allow_html=True)

else:
    itinerary = st.session_state.itinerary

    # Goal eval banner
    goal = st.session_state.goal_eval or {}
    if goal:
        sc = {"complete":"#3dba7e","partial":"#e8b84b","incomplete":"#e87060"}.get(goal.get("status",""),"#5a7a6a")
        chips_html = " ".join(
            f'<span class="{"goal-pass" if ok else "goal-fail"}">{"✓" if ok else "✗"} {c}</span>'
            for c, ok in goal.get("checks",{}).items())
        st.markdown(f"""
        <div style="background:#1a3528;border:1px solid rgba(255,255,255,0.1);border-left:4px solid {sc};
                    border-radius:9px;padding:12px 16px;margin-bottom:14px;">
          <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:{sc};margin-bottom:6px;">Agent Self-Evaluation</div>
          <div style="font-size:0.8rem;color:#a8bfb3;margin-bottom:8px;">{goal.get('label','')}</div>
          <div style="display:flex;gap:5px;flex-wrap:wrap;">{chips_html}</div>
        </div>""", unsafe_allow_html=True)

    # Itinerary + Map
    res_left, res_right = st.columns([3, 2], gap="large")

    with res_left:
        st.markdown('<span class="mono-label">📋 Your Itinerary</span>', unsafe_allow_html=True)

        def md_to_html(text):
            text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
            text = re.sub(r'^## (.+)$',  r'<h2>\1</h2>', text, flags=re.MULTILINE)
            text = re.sub(r'^# (.+)$',   r'<h1>\1</h1>', text, flags=re.MULTILINE)
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            text = re.sub(r'^---+$', r'<hr>', text, flags=re.MULTILINE)
            text = re.sub(r'^[-*] (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
            text = re.sub(r'(<li>.*?</li>\n?)+', lambda m: '<ul>' + m.group(0) + '</ul>', text)
            lines = text.split('\n')
            out = []
            for line in lines:
                s = line.strip()
                if not s: out.append('<br>')
                elif s.startswith('<'): out.append(s)
                else: out.append(s + '<br>')
            return '\n'.join(out)

        st.markdown(
            f'<div style="background:#1a3528;border:1px solid rgba(255,255,255,0.1);border-left:4px solid #3dba7e;'
            f'border-radius:14px;padding:24px 26px;max-height:600px;overflow-y:auto;">'
            f'<div id="itin-inner">{md_to_html(itinerary)}</div></div>',
            unsafe_allow_html=True)

        arr_opt = ARRIVAL_OPTIONS.get(st.session_state.arrival_time, ARRIVAL_OPTIONS["morning"])
        st.markdown(f"""<div class="chip-row" style="margin-top:10px;">
          <span class="chip chip-g">{len(itinerary.split())} words</span>
          <span class="chip chip-y">llama-3.3-70b</span>
          <span class="chip chip-m">memory-aware</span>
          <span class="chip chip-m">{arr_opt['icon']} {arr_opt['label']} arrival</span>
        </div>""", unsafe_allow_html=True)

        st.download_button("⬇️  Download Itinerary (.txt)", data=itinerary,
            file_name="srilanka_itinerary.txt", mime="text/plain", use_container_width=True)

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        with st.expander("🔄 Refine this itinerary"):
            refine_input = st.text_area("refine",
                placeholder="e.g. Add more beach time on day 3, swap Kandy for Ella...",
                height=60, label_visibility="collapsed")
            if st.button("Apply Changes", key="btn_refine"):
                if refine_input.strip():
                    with st.spinner("Refining..."):
                        new_it, ge = refine_trip(st.session_state.itinerary, refine_input)
                    st.session_state.itinerary   = new_it
                    st.session_state.goal_eval   = ge
                    st.session_state.place_names = extract_place_names(new_it)
                    st.session_state.chat_messages = []
                    st.session_state.chat_history  = []
                    save_trip(days, ", ".join(interests_selected), budget, new_it)
                    st.rerun()
                else:
                    st.warning("Describe what to change first.")

    with res_right:
        locations = get_place_locations(st.session_state.place_names)
        st.markdown(f'<span class="mono-label">🗺️ Map · {len(locations)} places</span>', unsafe_allow_html=True)
        if not locations:
            st.info("Locations will appear after generation.")
        else:
            try:
                import folium
                from streamlit_folium import st_folium
                m = folium.Map(location=[7.8731, 80.7718], zoom_start=7, tiles="CartoDB dark_matter")
                for loc in locations:
                    folium.CircleMarker(
                        location=[loc["latitude"], loc["longitude"]],
                        radius=9, color="#3dba7e", fill=True, fill_color="#3dba7e", fill_opacity=0.85,
                        tooltip=folium.Tooltip(loc["name"],
                            style="font-family:DM Sans;font-size:12px;background:#1a3528;color:#e8f0ec;border:none;padding:4px 8px;border-radius:6px;"),
                        popup=folium.Popup(loc["name"], max_width=120),
                    ).add_to(m)
                    folium.Marker(
                        location=[loc["latitude"], loc["longitude"]],
                        icon=folium.DivIcon(
                            html=f'<div style="font-family:DM Sans,sans-serif;font-size:10px;font-weight:600;color:#e8f0ec;white-space:nowrap;margin-top:12px;text-shadow:0 1px 4px rgba(0,0,0,0.9);">{loc["name"]}</div>',
                            icon_size=(100,20), icon_anchor=(0,0)),
                    ).add_to(m)
                st_folium(m, width=None, height=580, returned_objects=[])
            except ImportError:
                for loc in locations:
                    st.markdown(f'<div class="hist-card">📍 <b style="color:#3dba7e;">{loc["name"]}</b></div>', unsafe_allow_html=True)
                st.caption("pip install folium streamlit-folium")

    # CHAT
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    msg_count = len([m for m in st.session_state.chat_history if m["role"] == "user"])

    st.markdown(f"""
    <div class="chat-header-bar">
      <div class="chat-hicon">🤖</div>
      <div style="flex:1;">
        <div class="chat-htitle">TravelLK AI Agent</div>
        <div class="chat-hsub">Ask anything about your itinerary · {msg_count} messages</div>
      </div>
      <div class="chat-online"></div>
    </div>""", unsafe_allow_html=True)

    # Chat messages area
    chat_container_style = """
    background:#0f1e15;
    border:1px solid rgba(61,186,126,0.15);
    border-top:none;
    padding:16px 18px;
    min-height:180px;
    max-height:400px;
    overflow-y:auto;
    """

    if not st.session_state.chat_history:
        st.markdown(f"""
        <div style="{chat_container_style}text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:180px;">
          <div style="font-size:2rem;margin-bottom:8px;">💬</div>
          <div style="font-size:0.82rem;color:#5a7a6a;line-height:1.9;">
            Ask me anything about your trip!<br>
            <span style="color:#3dba7e;">Add places · Get tips · Plan transport · Budget breakdown</span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        def format_agent_text(text: str) -> str:
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e8b84b;">\1</strong>', text)
            lines = text.split('\n')
            out, in_list = [], False
            for line in lines:
                s = line.strip()
                if s.startswith('- ') or s.startswith('* '):
                    if not in_list:
                        out.append('<ul style="padding-left:1.1rem;margin:6px 0;">')
                        in_list = True
                    out.append(f'<li style="margin-bottom:5px;color:#e8f0ec;">{s[2:]}</li>')
                else:
                    if in_list:
                        out.append('</ul>')
                        in_list = False
                    if s:
                        out.append(f'<p style="margin:4px 0;color:#e8f0ec;">{s}</p>')
            if in_list:
                out.append('</ul>')
            return '\n'.join(out)

        t = datetime.datetime.now().strftime("%H:%M")
        msgs_html = ""
        for msg in st.session_state.chat_history:
            is_user = msg["role"] == "user"
            if is_user:
                safe = msg["content"].replace("<","&lt;").replace(">","&gt;")
                msgs_html += f"""
                <div class="bubble-user-row">
                  <div>
                    <div class="bubble-user">{safe}</div>
                    <div class="bubble-user-meta">You · {t}</div>
                  </div>
                </div>"""
            else:
                formatted = format_agent_text(msg["content"])
                msgs_html += f"""
                <div class="bubble-agent-row">
                  <div class="chat-avatar">🤖</div>
                  <div>
                    <div class="bubble-agent">{formatted}</div>
                    <div class="bubble-agent-meta">Agent · {t}</div>
                  </div>
                </div>"""

        st.markdown(f'<div style="{chat_container_style}">{msgs_html}</div>', unsafe_allow_html=True)

    # Quick suggestions row
    suggestions = ["🚂 Best transport?","🏨 Hotel tips?","🍛 Must-try foods?","📅 Add Kandy","💰 Budget breakdown?","🌧️ Best time to visit?"]
    sug_html = " ".join(f'<span class="sug-chip" style="cursor:pointer;">{s}</span>' for s in suggestions)
    st.markdown(f"""
    <div class="sug-row">
      <span class="sug-label">Quick questions</span>
      {sug_html}
    </div>""", unsafe_allow_html=True)

    # ── Input box + Send button INSIDE the chat frame ──
    st.markdown("""
    <div style="background:#0f1e15;border:1px solid rgba(61,186,126,0.15);
                border-top:1px solid rgba(61,186,126,0.1);
                border-radius:0 0 14px 14px;padding:10px 14px;">
    </div>""", unsafe_allow_html=True)

    input_col, btn_col = st.columns([9, 1], gap="small")
    with input_col:
        st.markdown("""
        <style>
        div[data-testid="stTextInput"] > div > div > input {
            background: #0f1e15 !important;
            border: 1.5px solid rgba(61,186,126,0.25) !important;
            border-radius: 10px !important;
            color: #e8f0ec !important;
            font-size: 0.86rem !important;
            padding: 10px 14px !important;
        }
        div[data-testid="stTextInput"] > div > div > input:focus {
            border-color: #3dba7e !important;
            box-shadow: 0 0 0 3px rgba(61,186,126,0.15) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.session_state.get("clear_chat_input"):
            st.session_state["clear_chat_input"] = False
            st.session_state["chat_input"] = ""
        user_q = st.text_input(
            "chat_input_field",
            placeholder="Ask anything... e.g. Best restaurants in Kandy · Train routes · Budget tips",
            label_visibility="collapsed",
            key="chat_input"
        )
    with btn_col:
        st.markdown("""
        <style>
        .send-btn-wrap > div[data-testid="stButton"] > button {
            background: linear-gradient(135deg,#3dba7e,#2a9062) !important;
            border: none !important;
            border-radius: 10px !important;
            color: #fff !important;
            font-size: 1.1rem !important;
            padding: 9px 10px !important;
            width: 100% !important;
            box-shadow: 0 4px 14px rgba(61,186,126,0.3) !important;
            margin-top: -12px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="send-btn-wrap">', unsafe_allow_html=True)
        send_clicked = st.button("➤", key="btn_send_chat", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Handle send
    if send_clicked and user_q and user_q.strip():
        with st.spinner("🤖 Agent thinking..."):
            reply, updated = chat_with_agent(
                st.session_state.chat_messages, user_q.strip(), itinerary)
        st.session_state.chat_messages = updated
        st.session_state.chat_history.append({"role": "user",      "content": user_q.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state["clear_chat_input"] = True
        st.rerun()
    # Clear button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history", key="btn_clr", use_container_width=False):
            st.session_state.chat_messages = []
            st.session_state.chat_history  = []
            st.rerun()