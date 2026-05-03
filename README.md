# 🌴 TravelLK - AI Travel Planning Agent for Sri Lanka

---

## Overview

**TravelLK** is an autonomous AI travel planning agent that generates personalised, day-by-day Sri Lanka itineraries. Given a trip duration, interests, and budget, the agent reasons about destinations, logistics, food, costs, and live weather then outputs a fully structured travel plan.

The agent integrates:
- **LLM reasoning** (Groq API · LLaMA 3.3 70B) for itinerary generation and travel-style decisions
- **Live weather** via OpenWeatherMap API
- **Persistent memory** via SQLite learns from past trips to personalise future plans
- **Interactive map** with extracted place names and coordinates
- **Conversational chat** to refine or ask questions about any itinerary

---

## Demo

**[Watch the YouTube Short](https://youtube.com/shorts/XO8byYigMPQ?feature=share)**

---

## Project Structure

```
travellk/
│
├── app.py            # Streamlit UI all pages and user interaction
├── agent.py          # Core AI agent LLM calls, tools, reasoning, planning
├── memory.py         # SQLite memory saves/retrieves past trips & preferences
├── travellk.db       # Auto-created SQLite database (generated at runtime)
├── requirements.txt
└── README.md
```

---

## Features

### Agent Capabilities
| Feature | Description |
|---|---|
| **Trip Planning** | Generates structured day-by-day itineraries with morning / afternoon / evening breakdowns |
| **Travel Style Detection** | Automatically decides travel pace (relaxed, moderate, packed) from user preferences |
| **Goal Checking** | Verifies the itinerary meets the stated user goals before presenting it |
| **Trip Refinement** | Refines an existing plan based on follow-up user feedback |
| **Chat Mode** | Freeform Q&A about any aspect of the itinerary or Sri Lanka travel |

### Memory System
| Feature | Description |
|---|---|
| **Trip History** | Saves every generated itinerary to SQLite with timestamp |
| **Preference Learning** | Analyses past trips to find preferred budget, avg trip length, and top interests |
| **Smart Context Injection** | Injects memory summary into every new LLM prompt for personalised results |
| **Past Itinerary Reload** | Load and view any previously generated trip |

### Live Weather
- Fetches real-time weather for extracted destination cities using OpenWeatherMap
- Displays temperature, conditions, humidity, and wind speed alongside the itinerary

### Interactive Map
- Extracts place names from the generated itinerary
- Plots them on an interactive map within the Streamlit UI

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/travellk.git
cd travellk
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
WEATHER_API_KEY = "your_openweathermap_api_key_here"
```

Or export as environment variables:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
export WEATHER_API_KEY="your_openweathermap_api_key_here"
```

- Get a free Groq API key at [console.groq.com](https://console.groq.com)
- Get a free OpenWeatherMap key at [openweathermap.org](https://openweathermap.org/api)

### 4. Run the App

```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
groq
requests
```

> SQLite3 is included in Python's standard library - no installation needed.

---

## Architecture

```
User Input (Streamlit UI)
        │
        ▼
   agent.py
   ├── decide_travel_style()    → LLM decides pace
   ├── get_memory_context()     → Injects past trip history
   ├── plan_trip()              → Main LLM call → itinerary text
   ├── check_goal_achievement() → LLM verifies plan meets goals
   ├── get_weather()            → OpenWeatherMap API
   └── extract_place_names()   → Extracts destinations from itinerary
        │
        ▼
   memory.py
   └── save_trip()             → Persists to SQLite (travellk.db)
        │
        ▼
   Streamlit UI renders itinerary, map, weather, and history
```
