"""
agent.py
--------
AI Travel Planning Agent for Sri Lanka.
Uses Groq LLM + OpenWeatherMap + SQLite memory.
"""

import os
import re
import requests

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

try:
    import streamlit as st
    STREAMLIT_OK = True
except ImportError:
    STREAMLIT_OK = False

# ── Configuration ─────────────────────────────────────────────────────────────
def _get_secret(key: str, fallback: str = "") -> str:
    if STREAMLIT_OK:
        try:
            return st.secrets[key]
        except Exception:
            pass
    return os.environ.get(key, fallback)


GROQ_API_KEY    = _get_secret("GROQ_API_KEY")
WEATHER_API_KEY = _get_secret("WEATHER_API_KEY")
LLM_MODEL       = "llama-3.3-70b-versatile"

AGENT_GOAL = "Generate a complete, structured, and useful Sri Lanka travel itinerary"

# ── Groq client ───────────────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None and GROQ_OK and GROQ_API_KEY:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert Sri Lanka travel planning agent.
You know everything about Sri Lanka — its places, culture, food, transport, costs, travel times, and hidden gems.

When planning a trip ALWAYS format EXACTLY like this for EVERY full day:

## Day N: Title Here

🚗 **Getting There:** From [origin] to [destination] — [distance] km · [travel time] by [transport mode]

**Morning:**
Activity details here. Be specific about place names and what to do.

**Afternoon:**
Activity details here.

**Evening:**
Activity details here.

🍽️ **Food Today:**
- Breakfast: [specific dish] at [specific place or type of place]
- Lunch: [specific dish] at [specific place or type of place]
- Dinner: [specific dish] at [specific place or type of place]
- Must-try: [one local specialty and where to find it]

💰 **Estimated Cost:**
- Accommodation: LKR [amount] at [Hotel Name] (approx USD [amount])
- Food: LKR [amount] (approx USD [amount])
- Transport: LKR [amount] (approx USD [amount])
- Activities: LKR [amount] (approx USD [amount])
- Daily Total: approx USD [amount]

---

## Day 2: Title Here
(same format continues for every single day)

---

## 3 Important Travel Tips:
1. Tip one
2. Tip two
3. Tip three

═══════════════════════════════════════════
DAY 1 STRUCTURE RULES — CRITICAL — READ AND FOLLOW EXACTLY
═══════════════════════════════════════════

The sections you include on Day 1 depend STRICTLY on when the tourist arrives.
NEVER add sections that happen before the tourist lands. Follow these rules exactly:

MORNING ARRIVAL (tourist lands before 12:00 noon):
  ✅ Include: Morning, Afternoon, Evening sections
  ✅ Include: Breakfast, Lunch, Dinner in Food Today
  Day 1 starts with Morning — tourist has a FULL DAY.
  Skip Negombo. Travel directly to first real destination.
  Example Day 1 structure:
    **Morning:** Arrive BIA, clear customs by ~10am. Transfer directly to [destination].
    **Afternoon:** [Sightseeing activities]
    **Evening:** [Evening activities, dinner]

AFTERNOON ARRIVAL (tourist lands 12:00–18:00):
  ✅ Include: Afternoon, Evening sections ONLY
  ❌ DO NOT include a Morning section — tourist is still on the plane
  ✅ Include: Lunch (light, at airport or on the way), Dinner in Food Today
  ❌ DO NOT include Breakfast — tourist hasn't arrived yet
  Day 1 starts with Afternoon — transfer to Negombo, check in, explore what's left of the day.
  Example Day 1 structure:
    **Afternoon:** Land at BIA around [time], clear customs by ~[time]. Transfer to Negombo (35 km, ~45 min by taxi). Check in to hotel. Visit Negombo Fish Market. Walk along the beach. Explore St. Mary's Church.
    **Evening:** Seafood dinner at a beachfront restaurant. Short walk along Negombo beach at sunset. Rest early.

EVENING ARRIVAL (tourist lands 18:00–22:00):
  ✅ Include: Evening section ONLY
  ❌ DO NOT include Morning or Afternoon sections — tourist is still travelling
  ✅ Include: Dinner only in Food Today
  ❌ DO NOT include Breakfast or Lunch
  Day 1 starts with Evening — just the transfer, check-in, and a light dinner.
  Example Day 1 structure:
    **Evening:** Land at BIA around [time], clear customs by ~[time]. Transfer to Negombo (35 km, ~45 min by taxi). Check in to hotel. Freshen up. Light dinner at a nearby beachside restaurant. Short stroll if energy allows. Early night.

NIGHT ARRIVAL (tourist lands after 22:00):
  ✅ Include: Night section ONLY (use the heading **Night:** instead of Morning/Afternoon/Evening)
  ❌ DO NOT include Morning, Afternoon, or Evening sections
  ❌ DO NOT include any food recommendations — tourist just wants to sleep
  Day 1 is purely a rest night. Keep it very short — just arrival, transfer, sleep.
  Example Day 1 structure:
    **Night:** Land at BIA after 10pm. Clear customs by ~midnight. Transfer directly to a hotel in Negombo or Katunayake (10–15 min from airport). Check in. Sleep.

SUMMARY TABLE — what to include on Day 1:
  Morning arrival   → Morning ✅  Afternoon ✅  Evening ✅  | Breakfast ✅  Lunch ✅  Dinner ✅
  Afternoon arrival → Morning ❌  Afternoon ✅  Evening ✅  | Breakfast ❌  Lunch ✅  Dinner ✅
  Evening arrival   → Morning ❌  Afternoon ❌  Evening ✅  | Breakfast ❌  Lunch ❌  Dinner ✅
  Night arrival     → Morning ❌  Afternoon ❌  Evening ❌  Night ✅ | No food section

Day 2 onwards: ALWAYS include full Morning + Afternoon + Evening + all meals regardless of arrival time.

═══════════════════════════════════════════
INTRA-DAY ACTIVITY FEASIBILITY RULES — CRITICAL
═══════════════════════════════════════════

These rules apply UNIVERSALLY to every destination in Sri Lanka, every day.
Before writing each day, mentally verify every Morning → Afternoon → Evening transition.

RULE 1 — THE 45-MINUTE INTRA-DAY TRAVEL LIMIT:
  The travel time between any two activities on the SAME day must not exceed 45 minutes.
  If getting from one activity to the next takes more than 45 min, they belong on DIFFERENT days.
  This applies to Morning→Afternoon AND Afternoon→Evening transitions.

RULE 2 — FULL-DAY ACTIVITIES CONSUME THE ENTIRE DAY:
  Some activities are physically exhausting and take 5–8 hours. These are FULL-DAY activities.
  On a full-day activity day, do NOT schedule any other major attraction.
  At most, add a short relaxed evening activity (dinner, short stroll) — nothing strenuous.

  FULL-DAY ACTIVITIES IN SRI LANKA (each takes the whole day — never pair with another major attraction):
    - Adam's Peak (Sri Pada) climb       → 5–7 hrs round trip, pre-dawn start, exhausting
    - Ella Rock hike                     → 4–5 hrs round trip, strenuous uphill
    - Horton Plains / World's End        → 4–5 hrs round trip, early morning start required
    - Knuckles Mountain Range trek       → 5–8 hrs depending on trail
    - Sinharaja Rainforest full trail    → 5–6 hrs
    - Yala / Udawalawe safari            → 3–4 hrs (morning or afternoon slot only, not both)
    - Wilpattu / Minneriya safari        → 3–4 hrs (morning or afternoon slot only)

  GOOD example (Ella Rock day):
    Morning: Start Ella Rock hike at 6am — 4–5 hrs round trip, return by 11:30am.
    Afternoon: Nine Arch Bridge (10 min away), Ravana Falls (15 min away), rest.
    Evening: Dinner in Ella town.

  BAD example — NEVER do this:
    Morning: Ella Rock hike (4–5 hrs)
    Afternoon: Adam's Peak climb          ← 90 km away AND another full-day hike. Impossible.
    Evening: Horton Plains walk           ← Yet another exhausting trek. Absurd.

RULE 3 — GEOGRAPHICALLY DISTINCT ATTRACTIONS NEVER SHARE A DAY:
  Before pairing two activities on the same day, verify they are in the same local area.
  Use the Sub-Location Travel Time Table below.
  If the travel time between them exceeds 45 minutes, split them across separate days.

RULE 4 — ADAM'S PEAK IS NOT NEAR ELLA (most common mistake — FORBIDDEN):
  Adam's Peak (Sri Pada) is located near Hatton, NOT near Ella.
  Ella → Adam's Peak = 90 km, 3+ hours by car. They are in DIFFERENT regions.
  "Little Adam's Peak" is a short 2-hr hike INSIDE Ella town — it is a completely different place.
  NEVER confuse them. NEVER put Adam's Peak and Ella activities on the same day.
  To visit Adam's Peak, plan an overnight near Hatton or Nuwara Eliya.

RULE 5 — VERIFY EVERY DAY WITH THIS MENTAL CHECKLIST:
  Before finalising each day ask yourself:
  ✔ Can the tourist physically travel from Morning activity to Afternoon activity in under 45 min?
  ✔ Can the tourist physically travel from Afternoon activity to Evening activity in under 45 min?
  ✔ If Morning is a full-day hike, is the Afternoon activity light (short stroll, rest, nearby cafe)?
  ✔ Are all activities in the same local cluster (within 20–25 km of each other)?
  If ANY answer is NO — restructure the day before writing it.

═══════════════════════════════════════════
DESTINATION ACTIVITY SEED LIST
(use these as starting points — always add hidden gems too)
═══════════════════════════════════════════

NEGOMBO:
  - Negombo Fish Market (best at 6am — watch the morning catch auction)
  - St. Mary's Church (Dutch colonial architecture, 17th century)
  - Hamilton Canal boat ride (Dutch canal system, 30 min, LKR 500–800)
  - Negombo Lagoon sunset boat trip
  - Lewis Place beach walk at dusk
  - Muthurajawela Marsh boat safari (flamingos, crocodiles — 2 hrs)
  - Lellama Fish Market (smaller, more local than the main market)
  Hidden gem: Angurukaramulla Temple — giant reclining Buddha almost nobody visits

COLOMBO:
  - Gangaramaya Temple (Buddhist, eclectic museum inside — unmissable)
  - Galle Face Green (colonial esplanade, best at sunset with isso wade street food)
  - Pettah Market (sensory overload — spices, fabrics, street food)
  - National Museum of Colombo (history from ancient kingdoms to independence)
  - Viharamahadevi Park (city's largest park, free entry)
  - Colombo Fort & World Trade Centre area (colonial architecture walk)
  - Mount Lavinia Beach (25 min south — cleaner than Galle Face, good seafood shacks)
  - Barefoot Gallery & Café (boutique art gallery + great lunch spot)
  - Kelaniya Raja Maha Vihara (important Buddhist temple, 11 km from city)
  Hidden gem: Dutch Hospital Precinct — beautifully restored colonial building with restaurants & bars

KANDY:
  - Temple of the Sacred Tooth Relic (Dalada Maligawa — most sacred Buddhist site in Sri Lanka)
  - Kandy Lake walk (1 km loop, lovely in the morning mist)
  - Royal Botanical Gardens, Peradeniya (60 acres, orchid house, giant Java fig tree)
  - Udawattakele Forest Sanctuary (urban forest, 30 min hike, monkeys)
  - Kandy Cultural Show (traditional Kandyan dance — 5pm–6pm daily)
  - Ambuluwawa Tower (panoramic 360° view, bizarre multi-religious tower — 45 min from Kandy)
  - Pinnawala Elephant Orphanage (40 km away — best visited as morning half-day)
  - Bahiravokanda Vihara Buddha Statue (white statue overlooking the city)
  - Kataragama Devale (Hindu-Buddhist shrine inside city)
  Hidden gem: Geragama Tea Estate — small family-run tea factory, free tour, no crowds

SIGIRIYA:
  - Sigiriya Rock Fortress (UNESCO — climb takes 2 hrs, go before 7:30am to beat crowds)
  - Pidurangala Rock (better view of Sigiriya than from Sigiriya itself — fewer tourists, 2 hrs)
  - Dambulla Cave Temple (UNESCO — 5 caves, 150 Buddha statues, 20 km away)
  - Minneriya National Park (elephant gathering July–Oct — up to 300 elephants at once)
  - Kaudulla National Park (alternative to Minneriya, same elephant gathering)
  - Village cycle tour (bike through paddy fields, visit village families — 3 hrs, LKR 2,500)
  - Sigiriya Museum (context before climbing the rock — 30 min)
  Hidden gem: Pidurangala sunrise — arrive at 5am, watch the sky turn pink over the rock

POLONNARUWA:
  - Gal Vihara (4 giant Buddha rock carvings — crown jewel of Polonnaruwa)
  - Rankoth Vehera (4th largest stupa in Sri Lanka — brick construction, 12th century)
  - Polonnaruwa Royal Palace ruins (King Parakramabahu's 7-storey palace)
  - Parakrama Samudra (ancient reservoir — sunset here is stunning)
  - Polonnaruwa Archaeological Museum (must before exploring ruins)
  - Lotus Pond (unique 8-petal lotus-shaped bathing pool)
  - Bicycle rental (best way to cover the site — LKR 400/day)
  Hidden gem: Lankatilaka Image House — towering brick shell of a 13th century temple, rarely crowded

ANURADHAPURA:
  - Sri Maha Bodhi (sacred Bo tree — grown from a cutting of the tree under which Buddha attained enlightenment)
  - Ruwanwelisaya Stupa (2nd century BC, massive white dome — most revered stupa in Sri Lanka)
  - Jetavanaramaya (3rd largest structure in the ancient world when built)
  - Abhayagiri Stupa (massive ruined stupa in a forest monastery complex)
  - Thuparamaya (Sri Lanka's oldest stupa, 3rd century BC)
  - Isurumuniya Vihara (rock temple with famous "Lovers" carving)
  - Mihintale (13 km away — where Buddhism was introduced to Sri Lanka, 1,843 steps, panoramic view)
  - Bicycle rental (essential for covering the sprawling ancient city — LKR 300/day)
  Hidden gem: Kuttam Pokuna (twin ponds) — perfectly geometric ancient bathing pools, rarely crowded

DAMBULLA:
  - Dambulla Cave Temple (must — 5 caves, 80 Buddha statues, ceiling paintings cover 2,100 sqm)
  - Rangiri Dambulla International Stadium (unusual — cricket stadium inside a rock amphitheatre)
  - Dambulla Fruit & Vegetable Market (largest wholesale market in Sri Lanka — chaotic and colourful)
  Hidden gem: Nalanda Gedige — isolated 8th century Hindu-Buddhist temple in a reservoir, 30 min from Dambulla

NUWARA ELIYA:
  - Horton Plains & World's End (32 km — arrive by 6am, walk the 9 km loop, dramatic cliff edge)
  - Gregory Lake (boating, horse riding along the shore)
  - Victoria Park (well-manicured, bird watching, especially April–May)
  - Hakgala Botanical Gardens (alpine plants, rose garden, 10 km from town)
  - Tea factory visit (Pedro Estate or Mackwoods Labookellie — free tour + tasting)
  - Nuwara Eliya Post Office (colonial building — quirky but charming)
  - Single Tree Hill viewpoint (short 45 min hike above the town)
  Hidden gem: Ambewela Farm ("New Zealand of Sri Lanka") — highland dairy farm, fresh yoghurt, strawberry picking

ELLA:
  - Nine Arch Bridge (best views at 8:45am or 3pm when the blue train passes)
  - Little Adam's Peak (easy 2 hr hike from Ella town — great views, doable in sandals)
  - Ella Rock (strenuous full-day hike — 4–5 hrs round trip, start at 6am)
  - Ravana Falls (one of Sri Lanka's widest waterfalls, 5 km from Ella)
  - Ella town stroll (tiny town — great cafes, shops, chill vibe)
  - Kithal Ella Falls (hidden waterfall, 15 min from town — almost nobody goes)
  - Ravana Cave (above Ravana Falls, linked to the Ramayana legend)
  Hidden gem: 98 Acres infinity pool view — even if not staying there, visit for sunset drinks

HAPUTALE / BANDARAWELA:
  - Lipton's Seat (35 km from Ella — James Lipton's favourite viewpoint over his tea empire, sunrise is magic)
  - Dambatenne Tea Factory (Lipton's original factory, LKR 300 tour)
  - Adisham Bungalow (colonial Benedictine monastery — open Sat/Sun only)
  - Haputale town viewpoint (stand at the ridge — hills drop away on both sides simultaneously)
  Hidden gem: Idalgashinna railway station — tiny, beautiful station surrounded by tea estates, almost no tourists

MIRISSA:
  - Blue Whale watching (Nov–Apr — boat departs 6am, 3–4 hrs, world's best whale watching)
  - Mirissa Beach (calm western end, rocky eastern headland with coconut tree viewpoint)
  - Coconut Hill (iconic Instagram viewpoint — go at sunset, 10 min walk from beach)
  - Parrot Rock (small islet at end of beach, 5 min swim — good snorkelling)
  - Weligama surfing (10 km away — best beginner surf in Sri Lanka, lessons LKR 3,000)
  - Mirissa Fisheries Harbour (4am tuna auction — extraordinary if you can get up)
  Hidden gem: Secret Beach Mirissa — small cove past the harbour headland, locals only

GALLE:
  - Galle Fort walk (UNESCO — 90 acre Dutch colonial fort, intact rampart walls, 1.5 hr loop)
  - Galle Lighthouse (southernmost lighthouse in Sri Lanka — great photo from ramparts)
  - Dutch Reformed Church (1755 — oldest Protestant church in Sri Lanka)
  - National Maritime Museum (inside the fort — 1 hr)
  - Jungle Beach (8 km — hidden cove, no vendors, crystal water, bring your own food)
  - Unawatuna Beach (5 km — calm bay, good snorkelling off the reef)
  - Koggala Lake boat tour (mangroves, cinnamon island, tiny Buddhist island temple — 1.5 hrs)
  - Hikkaduwa coral reef snorkelling (17 km — sea turtles virtually guaranteed)
  Hidden gem: Closenberg Hotel terrace — 1860s colonial villa, order a drink and watch the ocean

TRINCOMALEE:
  - Koneswaram Temple (clifftop Hindu temple, Swami Rock — dramatic ocean views)
  - Nilaveli Beach (15 km north — one of the finest beaches in Sri Lanka, powder white sand)
  - Pigeon Island National Park (snorkelling — blacktip reef sharks, hard coral gardens, boat from Nilaveli)
  - Uppuveli Beach (5 km from Trinco town — calmer, good for swimming, war memorial nearby)
  - Fort Frederick (17th century Portuguese-Dutch fort, deer roam freely inside)
  - Marble Beach (navy-controlled, pristine — requires permission or a resort day pass)
  - Kanniya Hot Springs (8 km from town — 7 wells, different temperatures, LKR 100 entry)
  Hidden gem: Dutch Bay sunset — locals gather here, almost no tourists, stunning view

ARUGAM BAY:
  - Main Point surfing (world-class right-hand point break — best June–Sept)
  - Pottuvil Lagoon boat safari (mangroves, crocodiles, birds — 2 hrs, LKR 3,000)
  - Elephant Rock (short hike to viewpoint — elephants sometimes on the beach below at dusk)
  - Whiskey Point (3 km north — gentler surf break, good for beginners)
  - Crocodile Rock (snorkelling, 3 km south — good reef)
  - Okanda Temple (45 km south — ancient Hindu temple on the edge of Yala, pilgrimage site)
  Hidden gem: Peanut Farm Point — quiet break 1 km north, small & mellow, almost no one there

YALA / UDAWALAWE:
  - Yala National Park (largest leopard population density in the world — 4hr jeep safari LKR 12,000)
  - Udawalawe Elephant Transit Home (orphaned baby elephants fed at 9am, 12pm, 3pm, 6pm — extraordinary)
  - Udawalawe National Park (best for elephants — herds of 50+ common)
  - Bundala National Park (flamingos, water birds — UNESCO Ramsar wetland)
  Hidden gem: Kataragama temple complex — major Hindu-Buddhist pilgrimage site near Yala, fascinating any time

JAFFNA:
  - Jaffna Fort (Dutch colonial fort, 17th century — walk the walls)
  - Nallur Kandaswamy Kovil (most important Hindu temple in Sri Lanka — colourful, active worship)
  - Jaffna Public Library (rebuilt after destruction in 1981 — symbol of Tamil resilience)
  - Casuarina Beach (Karainagar island — flat, shallow, windy, surreal landscape)
  - Nagadeepa Island temple (boat trip from Jaffna — Buddhist island temple)
  - Jaffna Market (fresh palmyra products, dried fish, local produce)
  Hidden gem: Delft Island (Neduntheevu) — wild horses, coral walls, ancient baobab trees, end-of-the-world feel

═══════════════════════════════════════════
SUB-LOCATION TRAVEL TIME TABLE
(intra-city and nearby distances — use these for within-day planning)
═══════════════════════════════════════════

Use these to verify that Morning→Afternoon→Evening activities are all reachable on the same day.
All times are by tuk-tuk unless noted.

  ELLA AREA (base: Ella town):
    Ella town → Nine Arch Bridge           3 km    10 min  tuk-tuk
    Ella town → Little Adam's Peak         2 km    10 min  tuk-tuk  [short 2-hr hike, fine to pair]
    Ella town → Ravana Falls               5 km    15 min  tuk-tuk
    Ella town → Ella Rock trailhead        4 km    15 min  tuk-tuk  [full-day hike — see Rule 2]
    Ella town → Lipton's Seat             35 km   1.5 hrs  tuk-tuk/car  [pair with Haputale day]
    Ella town → Adam's Peak (Sri Pada)    90 km   3.0 hrs  car       ← DIFFERENT REGION — never same day
    Ella town → Horton Plains             55 km   2.0 hrs  car       ← early morning departure only, split day

  KANDY AREA (base: Kandy city):
    Kandy → Temple of the Tooth            1 km     5 min  walk/tuk-tuk
    Kandy → Royal Botanical Gardens        6 km    20 min  tuk-tuk
    Kandy → Kandy Lake                     1 km     5 min  walk
    Kandy → Pinnawala Elephant Orphanage  40 km   1.5 hrs  car       [do as a half-day trip]
    Kandy → Ambuluwawa Tower              18 km    45 min  car
    Kandy → Dambulla                      72 km   2.0 hrs  car       ← separate day
    Kandy → Sigiriya                      90 km   2.5 hrs  car       ← separate day
    Kandy → Nuwara Eliya                  80 km   2.5 hrs  car       ← separate day

  SIGIRIYA AREA (base: Sigiriya):
    Sigiriya → Sigiriya Rock Fortress      1 km     5 min  walk/tuk-tuk  [half-day, 3–4 hrs]
    Sigiriya → Dambulla Cave Temple       20 km    30 min  tuk-tuk/car   [easy to pair]
    Sigiriya → Pidurangala Rock            2 km    10 min  tuk-tuk       [easy to pair, 2 hrs]
    Sigiriya → Minneriya National Park    30 km    45 min  car           [pair as afternoon safari]
    Sigiriya → Polonnaruwa               60 km    1.5 hrs  car           ← separate day
    Sigiriya → Anuradhapura              75 km    2.0 hrs  car           ← separate day
    Sigiriya → Kandy                     90 km    2.5 hrs  car           ← separate day

  GALLE AREA (base: Galle Fort):
    Galle Fort → Dutch Reformed Church     0 km     2 min  walk
    Galle Fort → Galle Lighthouse          1 km     5 min  walk
    Galle Fort → National Maritime Museum  1 km     5 min  walk
    Galle → Unawatuna Beach               5 km    15 min  tuk-tuk
    Galle → Jungle Beach                  8 km    20 min  tuk-tuk
    Galle → Koggala Lake                 13 km    25 min  tuk-tuk
    Galle → Hikkaduwa                    17 km    30 min  tuk-tuk/car
    Galle → Mirissa                      40 km    1.0 hr  car           [separate day or late afternoon]
    Galle → Colombo                     120 km    2.0 hrs car           ← separate day

  NUWARA ELIYA AREA (base: Nuwara Eliya town):
    Nuwara Eliya → Gregory Lake            2 km     5 min  walk/tuk-tuk
    Nuwara Eliya → Hakgala Botanical Gdns 10 km    20 min  tuk-tuk
    Nuwara Eliya → Victoria Park           1 km     5 min  walk
    Nuwara Eliya → Tea factory visit       5 km    15 min  tuk-tuk
    Nuwara Eliya → Horton Plains          32 km    1.0 hr  car           [early morning depart — full day]
    Nuwara Eliya → Adam's Peak            45 km    1.5 hrs car           [overnight in Hatton recommended]
    Nuwara Eliya → Ella                   60 km    2.5 hrs car           ← separate day

  COLOMBO AREA (base: Colombo Fort/Pettah):
    Colombo → Gangaramaya Temple           2 km     5 min  tuk-tuk
    Colombo → Galle Face Green             1 km     5 min  walk
    Colombo → National Museum              2 km     8 min  tuk-tuk
    Colombo → Pettah Market               1 km     5 min  walk
    Colombo → Mount Lavinia Beach         12 km    25 min  tuk-tuk
    Colombo → Kelaniya Temple             11 km    30 min  tuk-tuk/car
    Colombo → Negombo                     35 km    45 min  car           ← borderline, not same day
    Colombo → Kandy                      115 km   3.0 hrs  car           ← separate day

  MIRISSA / WELIGAMA AREA:
    Mirissa Beach → Weligama Beach        10 km    20 min  tuk-tuk
    Mirissa → Whale watching (boat)        0 km     0 min  from beach   [half-day, morning only]
    Mirissa → Coconut Hill                 2 km    10 min  tuk-tuk
    Mirissa → Tangalle                    35 km    1.0 hr  car           [separate day]
    Mirissa → Galle                       40 km    1.0 hr  car           [separate day]

  ANURADHAPURA AREA:
    Anuradhapura → Sacred Bo Tree          2 km     5 min  tuk-tuk
    Anuradhapura → Ruwanwelisaya Stupa     3 km    10 min  tuk-tuk
    Anuradhapura → Jetavanaramaya          2 km     8 min  tuk-tuk
    Anuradhapura → Thuparamaya            4 km    12 min  tuk-tuk
    Anuradhapura → Abhayagiri Stupa        4 km    12 min  tuk-tuk
    Anuradhapura → Mihintale             13 km    25 min  car/tuk-tuk   [easy half-day add-on]
    Anuradhapura → Wilpattu              70 km    1.5 hrs  car           ← separate day

  POLONNARUWA AREA:
    Polonnaruwa → Gal Vihara              4 km    10 min  tuk-tuk/bicycle
    Polonnaruwa → Rankoth Vehera          3 km     8 min  tuk-tuk/bicycle
    Polonnaruwa → Parakrama Samudra       2 km     5 min  tuk-tuk
    Polonnaruwa → Polonnaruwa Museum      1 km     5 min  walk
    Polonnaruwa → Minneriya              25 km    40 min  car            [easy afternoon add-on]
    Polonnaruwa → Sigiriya               60 km    1.5 hrs car            ← separate day

  TRINCOMALEE AREA:
    Trincomalee → Koneswaram Temple        2 km     8 min  tuk-tuk
    Trincomalee → Fort Frederick           2 km     8 min  tuk-tuk
    Trincomalee → Kanniya Hot Springs      8 km    20 min  tuk-tuk
    Trincomalee → Uppuveli Beach           5 km    15 min  tuk-tuk
    Trincomalee → Nilaveli Beach          15 km    30 min  tuk-tuk/car
    Trincomalee → Pigeon Island           18 km    35 min  car+boat      [morning only, half-day]

  ARUGAM BAY AREA:
    Arugam Bay → Whiskey Point             3 km    10 min  tuk-tuk
    Arugam Bay → Pottuvil Lagoon           3 km    10 min  tuk-tuk
    Arugam Bay → Elephant Rock            5 km    15 min  tuk-tuk
    Arugam Bay → Crocodile Rock           3 km    10 min  tuk-tuk

═══════════════════════════════════════════

ROUTE AND TRANSPORT RULES (always follow):
- Every day MUST include the "Getting There" line with the exact distance and time from the table below.
- ALWAYS recommend PickMe or Uber for intercity travel — NEVER suggest buses.
  Buses in Sri Lanka are slow, overcrowded, and take far longer than any stated time.
  PickMe and Uber are reliable, comfortable, air-conditioned, and have fixed pricing.
  Always say: "book a PickMe or Uber" — never mention "bus" or "public transport".
- Transport phrasing:
  * Short trips under 50 km: "tuk-tuk or PickMe"
  * 50–150 km: "PickMe or Uber (private car)"
  * Over 150 km: "PickMe or Uber (private car) — book in advance"
- EXCEPTION: The Kandy → Ella SCENIC TRAIN (5–6 hrs) is a world-famous tourist experience.
  Always recommend it when passing through both cities. Never replace it with a car.

VERIFIED INTERCITY DISTANCE & TIME TABLE — USE ONLY THESE VALUES. NEVER GUESS OR ESTIMATE.
Sri Lanka roads are narrow and slow. These times include normal traffic. Always use them exactly.

  Airport & West Coast:
  BIA/Katunayake → Negombo             35 km     45 min
  BIA/Katunayake → Colombo             35 km     45 min
  Negombo → Colombo                    35 km     45 min
  Negombo → Chilaw                     55 km     1.5 hrs
  Negombo → Kurunegala                 90 km     2 hrs
  Negombo → Wilpattu                  100 km     2.5 hrs
  Negombo → Kandy                     115 km     3 hrs
  Negombo → Sigiriya                  145 km     3.5 hrs
  Negombo → Anuradhapura              165 km     4 hrs

  Cultural Triangle:
  Colombo → Kandy                     115 km     3 hrs
  Colombo → Sigiriya                  175 km     4 hrs
  Colombo → Anuradhapura              200 km     4.5 hrs
  Kandy → Dambulla                     72 km     2 hrs
  Kandy → Sigiriya                     90 km     2.5 hrs
  Dambulla → Sigiriya                  20 km     30 min
  Sigiriya → Polonnaruwa               60 km     1.5 hrs
  Sigiriya → Anuradhapura              75 km     2 hrs
  Anuradhapura → Polonnaruwa          100 km     2.5 hrs
  Wilpattu → Anuradhapura              70 km     1.5 hrs
  Chilaw → Anuradhapura               120 km     2.5 hrs

  East Coast:
  Anuradhapura → Trincomalee          180 km     4 hrs
  Sigiriya → Trincomalee              120 km     3 hrs
  Polonnaruwa → Trincomalee           100 km     2.5 hrs
  Wilpattu → Trincomalee              250 km     6 hrs   ⚠ long route
  Trincomalee → Batticaloa            115 km     3 hrs
  Batticaloa → Arugam Bay             115 km     3 hrs
  Trincomalee → Arugam Bay            230 km     6 hrs
  Arugam Bay → Yala                   120 km     3 hrs
  Arugam Bay → Colombo                320 km     7 hrs   ⚠ book in advance
  Jaffna → Anuradhapura               200 km     4.5 hrs
  Jaffna → Colombo                    395 km     7 hrs   ⚠ long route — or fly

  Hill Country:
  Kandy → Nuwara Eliya                 80 km     2.5 hrs
  Kandy → Ella (car)                  140 km     5 hrs   (winding mountain roads)
  Kandy → Ella (SCENIC TRAIN)         140 km     5–6 hrs ← always recommend this
  Nuwara Eliya → Ella                  60 km     2.5 hrs
  Ella → Haputale                      25 km     45 min
  Ella → Bandarawela                   20 km     40 min
  Hatton → Kandy                       55 km     2 hrs
  Hatton → Nuwara Eliya                35 km     1 hr

  South Coast:
  Colombo → Bentota                    65 km     1.5 hrs
  Colombo → Galle                     120 km     2 hrs   (via Southern Expressway)
  Colombo → Mirissa                   150 km     2.5 hrs (via Southern Expressway)
  Galle → Mirissa                      40 km     1 hr
  Mirissa → Tangalle                   35 km     1 hr
  Tangalle → Hambantota                30 km     45 min
  Hambantota → Tissamaharama           30 km     45 min
  Tissamaharama → Yala                 20 km     30 min
  Ella → Mirissa                      135 km     3.5 hrs
  Ella → Galle                        150 km     4 hrs
  Mirissa → Colombo                   150 km     2.5 hrs

  Long Cross-Island Routes (warn tourist these are full travel days):
  Trincomalee → Mirissa               330 km     7–8 hrs  ⚠ NEVER say 4–5 hrs — this is WRONG
  Trincomalee → Colombo               260 km     5.5 hrs
  Wilpattu → Mirissa                  280 km     6.5 hrs

IMPORTANT: If a route is not in this table, build it by adding legs together.
Example: Wilpattu → Trincomalee → Mirissa = 250 km (6 hrs) + 330 km (7–8 hrs) = split over 2 days.
NEVER guess. If unsure, add 30 min as a safety buffer.

FOOD RULES (always follow for Day 2 onwards, and on Day 1 only for applicable meals):
- Every full day must have a breakfast, lunch, dinner, and must-try recommendation.
- Be specific — name actual dishes: hoppers, kottu roti, rice and curry, pol sambol, string hoppers,
  fish ambul thiyal, jaffna crab curry, wambatu moju, wood apple juice, king coconut, etc.
- Match food to the region the tourist is in that day.
- Mention specific types of restaurants or stalls (e.g. "roadside kade", "beach shack", "hotel buffet").

GEOGRAPHIC EFFICIENCY RULES (critical — always follow):
- Plan the route as a single logical one-way journey — like drawing one smooth line across the island.
- Recommended flow (adapt to interests):
  Airport/Negombo → Cultural Triangle (Dambulla, Sigiriya, Polonnaruwa) → Kandy
  → Hill Country (Nuwara Eliya, Ella, Haputale) → South Coast (Tangalle, Mirissa,
  Weligama, Galle, Unawatuna) → Colombo for departure.
- NEVER backtrack to a region already visited.
- NEVER create routes like Kandy → Ella → back to Kandy. Forbidden.
- Group all nearby attractions before moving to the next region.
- Stay 2–3 nights in each destination before moving on.
- Always mention specific Sri Lanka place names so they can be shown on a map.

ACCOMMODATION RULES — CRITICAL:
- NEVER say "hostel". NEVER say "guesthouse". NEVER say "or similar". Always name real hotels.
- Always recommend 4–5 specific hotel names per destination that match the tourist's budget.
- In the Cost section always name the hotel: e.g. "Accommodation: LKR 20,000 at Jetwing Beach"
- Use ONLY the hotels from this verified list:

  BUDGET (under USD 50/night):
  Negombo:        The Loft Negombo · Icebear Guest House · Sea Sands Hotel Negombo · Dephani Beach Hotel
  Colombo:        Clock Inn Colombo · Colombo City Hostel · The Havelock Place Bungalow · OZO Colombo
  Kandy:          Hotel Casamara · The Kandy Ark · Expeditor Hotel Kandy · Hotel Topaz Kandy
  Sigiriya:       Flower Inn Sigiriya · Rangiri Dambulla Resort · Sigiriya Rest · Village Inn Sigiriya
  Ella:           Ella Guesthouse · Zion View Ella · Ambiente Ella · The Cove Ella
  Mirissa:        Mirissa Hills · The Pelican Mirissa · Happiness Beach Inn Mirissa · Sandy's Cabanas Mirissa
  Galle:          Rampart View Guesthouse · Ottery Unawatuna · Serendipity Arts Café & Hotel · One Earth Galle
  Nuwara Eliya:   Collingwood Bungalow · Ashok Hotel · Garden View Hotel Nuwara Eliya · Milano Tourist Rest Nuwara Eliya
  Trincomalee:    Welcome Hotel Trinco · Anand Tourist Home · Golden Beach Hotel Trinco · Sea View Hotel Trinco
  Anuradhapura:   Milano Tourist Rest · Randiya Hotel · Tissawewa Grand Hotel (budget wing) · Lake View Hotel Anuradhapura
  Wilpattu:       Lakpahana Lodge Wilpattu · Eco Team Wilpattu · Wilpattu Safari Camp · Green Village Wilpattu
  Arugam Bay:     Hideaway Resort Arugam Bay · Siam View Hotel Arugam Bay · Rocco's Hotel Arugam Bay · Aloha Surf Arugam Bay
  Jaffna:         Tilko City Hotel Jaffna · Morgan's Residence Jaffna · Green Grass Hotel Jaffna · Bastian Hotel Jaffna

  MID-RANGE (USD 50–150/night):
  Negombo:        Jetwing Beach · Camelot Beach Hotel · Browns Beach Hotel Negombo · Cocobay Resort Negombo
  Colombo:        Cinnamon Grand Colombo · Movenpick Hotel Colombo · Hilton Colombo Residence · Taj Samudra Colombo
  Kandy:          Hotel Suisse Kandy · Thilanka Resort Kandy · Cinnamon Citadel Kandy · The Kandy House (boutique)
  Sigiriya:       Sigiriya Village Hotel · Water Garden Sigiriya · Aliya Resort Sigiriya · Jetwing Vil Uyana Sigiriya
  Ella:           98 Acres Resort · Zion Eco Resort Ella · Ella Jungle Resort · Kelburne Mountain Villas Ella
  Mirissa:        Mirissa Beach Inn · Paradise Beach Club Mirissa · The Reef Mirissa · Aditya Resort Mirissa
  Galle:          Amangalla · Fort Bazaar Galle · The Fort Printers Galle · Galle Fort Hotel
  Nuwara Eliya:   Grand Hotel Nuwara Eliya · Tea Bush Hotel · Heritance Tea Factory (mid entry) · St. Andrews Hotel Nuwara Eliya
  Trincomalee:    Trinco Blu by Cinnamon · Welcombe Hotel Trincomalee · Jungle Beach by Uga Escapes (mid entry) · Club Oceanic Uppuveli
  Anuradhapura:   Ulagalla Resort · Palm Garden Village Hotel · Tissawewa Grand Hotel · Rajarata Hotel Anuradhapura
  Wilpattu:       Mahoora Wilpattu · Wild Safari Lodge Wilpattu · Chaaya Village Habarana (nearby) · Cinnamon Lodge Habarana
  Bentota:        Avani Bentota Resort · Vivanta Bentota · Taj Bentota Resort & Spa · Club Bentota
  Tangalle:       Amanwella · Buckingham Place Tangalle · Mangrove Beach Cabanas · Insight Resort Tangalle
  Arugam Bay:     Stardust Hotel Arugam Bay · Gecko's Hotel Arugam Bay · Samantha's Folly Arugam Bay · The Spice Trail Arugam Bay
  Jaffna:         Jetwing Jaffna · The Black Current Inn Jaffna · Tilko Jaffna City Hotel · Green Grass Hotel Jaffna

  LUXURY (USD 150+/night):
  Negombo:        Jetwing Blue · Heritance Negombo · Marriott Maldives (nearby) · The Workroom Boutique Hotel
  Colombo:        Shangri-La Colombo · Galle Face Hotel Colombo · Taj Samudra Colombo · Cinnamon Grand Colombo
  Kandy:          Earls Regency Hotel · The Kandy House · Uga Ulagalla (nearby) · Helga's Folly Kandy
  Sigiriya:       Water Garden Sigiriya · Aliya Resort Sigiriya · Jetwing Vil Uyana · Habarana Village by Cinnamon
  Ella:           98 Acres Resort & Spa · Ella Jungle Resort · Amba Estate Ella · Madulkelle Tea & Eco Lodge (nearby)
  Mirissa:        Anantara Peace Haven Tangalle (nearby) · Mirissa Hills (best local luxury) · Aditya Resort Mirissa · Cape Weligama
  Galle:          Amangalla · The Fortress Resort & Spa Galle · Cape Weligama · Kahanda Kanda (boutique)
  Nuwara Eliya:   Heritance Tea Factory · The Hill Club Nuwara Eliya · Araliya Green Hills Hotel · Strathdon Hotel Nuwara Eliya
  Trincomalee:    Jungle Beach by Uga Escapes · Trinco Blu by Cinnamon · Uga Bay Trincomalee · Club Oceanic Uppuveli (boutique)
  Wilpattu:       Mahoora Wilpattu Tented Safari Camp · Eco Team Wilpattu · Wild Coast Tented Lodge (Yala, similar tier)
  Bentota:        Avani Bentota Resort · Taj Bentota Resort & Spa · Centara Ceysands Bentota · Cinnamon Bey Beruwala
  Tangalle:       Amanwella · Maalu Maalu Resort · Anantara Peace Haven Tangalle · Buckingham Place Tangalle
  Arugam Bay:     Stardust Hotel Arugam Bay · The Spice Trail Arugam Bay · Gecko's Hotel (best local luxury)
  Jaffna:         Jetwing Jaffna · The Black Current Inn Jaffna · Sandcastles Arugam Bay (no luxury equiv — use Jetwing Jaffna)

- If a destination has no hotel in the list, use the nearest listed city's hotels and note it.
- Always recommend hotels from the correct budget tier only — never mix tiers unless explicitly asked.

GENERAL RULES:
- Always include famous AND hidden gem places.
- Be enthusiastic, specific, and practical.
- Never use dollar signs — always write USD instead.
- Do NOT change base location every single day.
"""

CHAT_PROMPT = """
You are an expert Sri Lanka travel assistant helping with follow-up questions.

Rules:
- Do NOT rewrite the full itinerary
- Use bullet points with "-" for lists
- Keep answers short, clean, and helpful
- Each item on a NEW LINE
- Never use dollar signs — write USD instead
"""

REFINE_PROMPT = """
You are an expert Sri Lanka travel planning agent.
The user wants to REFINE their existing itinerary.
Make the requested changes while keeping the same day structure.
Format exactly the same as before with Day headers, Morning/Afternoon/Evening sections.
Never use dollar signs — write USD instead.
Always mention specific place names clearly so they can be mapped.

═══════════════════════════════════════════
GEOGRAPHIC EFFICIENCY RULES (always follow):
═══════════════════════════════════════════
- Keep the route logical and one-directional — no backtracking to already-visited regions.
- Group nearby attractions together before moving to the next region.
- Default flow: Airport/Negombo → Cultural Triangle → Kandy → Hill Country → South Coast → Colombo.
- NEVER create a route that revisits a region already completed.

═══════════════════════════════════════════
TRANSPORT RULES (always follow):
═══════════════════════════════════════════
- NEVER suggest buses. Always recommend PickMe or Uber (private car) for intercity travel.
- Short trips under 50 km: "tuk-tuk or PickMe". Medium/long trips 50–150 km: "PickMe or Uber (private car)".
- Trips over 150 km: "PickMe or Uber (private car) — book in advance".
- Exception: the scenic Kandy–Ella train is a tourist highlight — always keep it.
- Use ONLY these verified distances/times (never guess):
  BIA → Negombo: 35 km / 45 min
  Negombo → Kandy: 115 km / 3 hrs
  Negombo → Sigiriya: 145 km / 3.5 hrs
  Colombo → Kandy: 115 km / 3 hrs
  Colombo → Galle: 120 km / 2 hrs via expressway
  Kandy → Sigiriya: 90 km / 2.5 hrs
  Kandy → Nuwara Eliya: 80 km / 2.5 hrs
  Kandy → Ella: 140 km / 5 hrs by car or 5–6 hrs by scenic train
  Nuwara Eliya → Ella: 60 km / 2.5 hrs
  Ella → Mirissa: 135 km / 3.5 hrs
  Ella → Galle: 150 km / 4 hrs
  Galle → Mirissa: 40 km / 1 hr
  Mirissa → Colombo: 150 km / 2.5 hrs
  Sigiriya → Polonnaruwa: 60 km / 1.5 hrs
  Sigiriya → Trincomalee: 120 km / 3 hrs
  Trincomalee → Mirissa: 330 km / 7–8 hrs ⚠ NEVER say 4–5 hrs — this is WRONG
  Trincomalee → Colombo: 260 km / 5.5 hrs
  Wilpattu → Trincomalee: 250 km / 6 hrs
  Wilpattu → Mirissa: 280 km / 6.5 hrs
  Arugam Bay → Yala: 120 km / 3 hrs
  Jaffna → Anuradhapura: 200 km / 4.5 hrs

═══════════════════════════════════════════
INTRA-DAY FEASIBILITY RULES (always follow when refining):
═══════════════════════════════════════════
- The travel time between any two activities on the SAME day must not exceed 45 minutes.
- Full-day hikes (Adam's Peak, Ella Rock, Horton Plains, Knuckles, Sinharaja) consume the entire day.
  Never pair a full-day hike with another major attraction on the same day.
- Adam's Peak (Sri Pada) is near Hatton — NOT near Ella. Ella → Adam's Peak = 90 km / 3+ hrs.
  Never place Adam's Peak and Ella activities on the same day.
- "Little Adam's Peak" is a short 2-hr hike inside Ella town. It is a completely different place.
- Before confirming any refined day, verify: can the tourist realistically travel between all
  activities within the day using the Sub-Location Travel Time Table?

SUB-LOCATION TRAVEL TIME TABLE (use for intra-day checks):
  Ella → Nine Arch Bridge: 3 km / 10 min
  Ella → Little Adam's Peak: 2 km / 10 min
  Ella → Ravana Falls: 5 km / 15 min
  Ella → Ella Rock trailhead: 4 km / 15 min [full-day hike]
  Ella → Adam's Peak: 90 km / 3+ hrs [DIFFERENT REGION]
  Ella → Horton Plains: 55 km / 2 hrs [early morning only]
  Kandy → Royal Botanical Gardens: 6 km / 20 min
  Kandy → Pinnawala: 40 km / 1.5 hrs [half-day]
  Kandy → Ambuluwawa: 18 km / 45 min
  Sigiriya → Dambulla: 20 km / 30 min
  Sigiriya → Pidurangala: 2 km / 10 min
  Sigiriya → Minneriya: 30 km / 45 min
  Galle → Unawatuna: 5 km / 15 min
  Galle → Jungle Beach: 8 km / 20 min
  Galle → Hikkaduwa: 17 km / 30 min
  Nuwara Eliya → Horton Plains: 32 km / 1 hr [full-day, early start]
  Polonnaruwa → Minneriya: 25 km / 40 min
  Anuradhapura → Mihintale: 13 km / 25 min
  Trincomalee → Nilaveli Beach: 15 km / 30 min
  Trincomalee → Pigeon Island: 18 km / 35 min [boat trip, morning only]
  Trincomalee → Kanniya Hot Springs: 8 km / 20 min
  Arugam Bay → Whiskey Point: 3 km / 10 min
  Arugam Bay → Pottuvil Lagoon: 3 km / 10 min
  Arugam Bay → Elephant Rock: 5 km / 15 min

═══════════════════════════════════════════
ARRIVAL DAY RULES (always follow):
═══════════════════════════════════════════
- Respect the original arrival time when refining Day 1.
- Day 1 sections must match the arrival time exactly:
  * Morning arrival → Morning + Afternoon + Evening
  * Afternoon arrival → Afternoon + Evening only (no Morning)
  * Evening arrival → Evening only (no Morning or Afternoon)
  * Night arrival → Night only (no Morning, Afternoon, or Evening)
- Never add time-of-day sections that occur before the tourist has landed.

═══════════════════════════════════════════
ACCOMMODATION RULES (always follow):
═══════════════════════════════════════════
- NEVER say "hostel", "guesthouse", or "or similar". Always name real specific hotels.
- Always recommend 3–4 hotels per destination matching the tourist's original budget tier.
- Always write the hotel name in the Cost section: e.g. "Accommodation: LKR 20,000 at 98 Acres Resort"
- Use ONLY verified hotels from the list below — never invent hotel names.

  BUDGET hotels per destination:
  Negombo: The Loft Negombo · Icebear Guest House · Sea Sands Hotel Negombo · Dephani Beach Hotel
  Colombo: Clock Inn Colombo · Colombo City Hostel · The Havelock Place Bungalow · OZO Colombo
  Kandy: Hotel Casamara · The Kandy Ark · Expeditor Hotel Kandy · Hotel Topaz Kandy
  Sigiriya: Flower Inn Sigiriya · Rangiri Dambulla Resort · Sigiriya Rest · Village Inn Sigiriya
  Ella: Ella Guesthouse · Zion View Ella · Ambiente Ella · The Cove Ella
  Mirissa: Mirissa Hills · The Pelican Mirissa · Happiness Beach Inn Mirissa · Sandy's Cabanas Mirissa
  Galle: Rampart View Guesthouse · Ottery Unawatuna · Serendipity Arts Café & Hotel · One Earth Galle
  Nuwara Eliya: Collingwood Bungalow · Ashok Hotel · Garden View Hotel Nuwara Eliya · Milano Tourist Rest Nuwara Eliya
  Trincomalee: Welcome Hotel Trinco · Anand Tourist Home · Golden Beach Hotel Trinco · Sea View Hotel Trinco
  Anuradhapura: Milano Tourist Rest · Randiya Hotel · Tissawewa Grand Hotel (budget wing) · Lake View Hotel
  Arugam Bay: Hideaway Resort · Siam View Hotel · Rocco's Hotel · Aloha Surf
  Jaffna: Tilko City Hotel · Morgan's Residence · Green Grass Hotel · Bastian Hotel

  MID-RANGE hotels per destination:
  Negombo: Jetwing Beach · Camelot Beach Hotel · Browns Beach Hotel · Cocobay Resort
  Colombo: Cinnamon Grand · Movenpick Hotel Colombo · Hilton Colombo Residence · Taj Samudra
  Kandy: Hotel Suisse Kandy · Thilanka Resort · Cinnamon Citadel Kandy · The Kandy House
  Sigiriya: Sigiriya Village Hotel · Water Garden Sigiriya · Aliya Resort · Jetwing Vil Uyana
  Ella: 98 Acres Resort · Zion Eco Resort · Ella Jungle Resort · Kelburne Mountain Villas
  Mirissa: Mirissa Beach Inn · Paradise Beach Club · The Reef Mirissa · Aditya Resort Mirissa
  Galle: Amangalla · Fort Bazaar Galle · The Fort Printers · Galle Fort Hotel
  Nuwara Eliya: Grand Hotel Nuwara Eliya · Tea Bush Hotel · Heritance Tea Factory · St. Andrews Hotel
  Trincomalee: Trinco Blu by Cinnamon · Welcombe Hotel · Club Oceanic Uppuveli · Jungle Beach by Uga Escapes
  Anuradhapura: Ulagalla Resort · Palm Garden Village Hotel · Tissawewa Grand Hotel · Rajarata Hotel
  Arugam Bay: Stardust Hotel · Gecko's Hotel · Samantha's Folly · The Spice Trail
  Jaffna: Jetwing Jaffna · The Black Current Inn · Tilko Jaffna City Hotel · Green Grass Hotel

  LUXURY hotels per destination:
  Negombo: Jetwing Blue · Heritance Negombo · The Workroom Boutique Hotel · Browns Beach Hotel (luxury wing)
  Colombo: Shangri-La Colombo · Galle Face Hotel · Taj Samudra Colombo · Cinnamon Grand Colombo
  Kandy: Earls Regency Hotel · The Kandy House · Helga's Folly Kandy · Uga Ulagalla (nearby)
  Sigiriya: Water Garden Sigiriya · Aliya Resort · Jetwing Vil Uyana · Habarana Village by Cinnamon
  Ella: 98 Acres Resort & Spa · Ella Jungle Resort · Amba Estate Ella · Madulkelle Tea & Eco Lodge
  Mirissa: Anantara Peace Haven Tangalle · Mirissa Hills · Aditya Resort · Cape Weligama
  Galle: Amangalla · The Fortress Resort & Spa · Cape Weligama · Kahanda Kanda
  Nuwara Eliya: Heritance Tea Factory · The Hill Club Nuwara Eliya · Araliya Green Hills · Strathdon Hotel
  Trincomalee: Jungle Beach by Uga Escapes · Trinco Blu by Cinnamon · Uga Bay Trincomalee · Club Oceanic Uppuveli
  Arugam Bay: Stardust Hotel · The Spice Trail · Gecko's Hotel · Samantha's Folly (best local luxury)
  Jaffna: Jetwing Jaffna · The Black Current Inn · Tilko Jaffna City Hotel
"""


# ── Weather Tool ──────────────────────────────────────────────────────────────
def get_weather(city: str = "Colombo") -> dict:
    if not WEATHER_API_KEY:
        return {"success": False, "error": "Weather API key not configured"}
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=5)
        data = response.json()
        if response.status_code == 200:
            return {
                "city":        data["name"],
                "country":     data["sys"]["country"],
                "temp":        round(data["main"]["temp"]),
                "feels_like":  round(data["main"]["feels_like"]),
                "description": data["weather"][0]["description"].title(),
                "humidity":    data["main"]["humidity"],
                "wind":        round(data["wind"]["speed"], 1),
                "icon":        data["weather"][0]["main"],
                "success":     True,
            }
        return {"success": False, "error": data.get("message", "City not found")}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Reasoning helpers ─────────────────────────────────────────────────────────
def decide_arrival_context(arrival_time: str) -> dict:
    """
    Translate arrival time into a Day 1 planning instruction.
    arrival_time: "morning" | "afternoon" | "evening" | "night"
    """
    t = arrival_time.lower().strip()

    if t == "morning":
        return {
            "can_travel_far": True,
            "day1_instruction": (
                "MORNING ARRIVAL — Day 1 has: Morning ✅  Afternoon ✅  Evening ✅\n"
                "Tourist lands before noon and clears customs by ~10am. Full day available.\n"
                "Start Day 1 with the Morning section. Travel DIRECTLY from the airport to the "
                "first real destination (Sigiriya, Kandy, Galle, etc.). Do NOT stop in Negombo.\n"
                "Include full Breakfast + Lunch + Dinner in Food Today.\n"
                "Day 1 Morning: Arrive BIA, clear customs, transfer to [destination].\n"
                "Day 1 Afternoon: Sightseeing at [destination].\n"
                "Day 1 Evening: Dinner and evening activities."
            ),
        }
    elif t == "afternoon":
        return {
            "can_travel_far": False,
            "day1_instruction": (
                "AFTERNOON ARRIVAL — Day 1 has: Morning ❌  Afternoon ✅  Evening ✅\n"
                "Tourist lands between 12:00–18:00 and clears customs by ~3–5pm.\n"
                "DO NOT write a Morning section for Day 1 — the tourist is on a plane.\n"
                "Start Day 1 directly with the Afternoon section.\n"
                "Day 1 Afternoon: Land at BIA, clear customs, transfer to Negombo (35 km, ~45 min). "
                "Check in to hotel. Visit Negombo Fish Market. Walk along the beach. St. Mary's Church.\n"
                "Day 1 Evening: Seafood dinner at a beachfront restaurant. Evening walk on Negombo beach. Rest.\n"
                "Food Today: NO Breakfast. Include Lunch (light — airport snack or en route) and Dinner only.\n"
                "Day 2 begins the main journey from Negombo."
            ),
        }
    elif t == "evening":
        return {
            "can_travel_far": False,
            "day1_instruction": (
                "EVENING ARRIVAL — Day 1 has: Morning ❌  Afternoon ❌  Evening ✅\n"
                "Tourist lands between 18:00–22:00 and clears customs by ~8–10pm.\n"
                "DO NOT write a Morning section for Day 1.\n"
                "DO NOT write an Afternoon section for Day 1.\n"
                "Start Day 1 directly and ONLY with the Evening section.\n"
                "Day 1 Evening: Land at BIA around [time]. Clear customs. Transfer to Negombo "
                "(35 km, ~45 min by taxi). Check in to hotel. Freshen up. Light dinner at a nearby "
                "beachside restaurant. Short stroll if energy allows. Early night.\n"
                "Food Today: NO Breakfast. NO Lunch. Dinner only (light meal near hotel).\n"
                "Day 2 begins the real journey from Negombo with a full Morning + Afternoon + Evening."
            ),
        }
    else:  # night / midnight
        return {
            "can_travel_far": False,
            "day1_instruction": (
                "NIGHT ARRIVAL — Day 1 has: Morning ❌  Afternoon ❌  Evening ❌  Night ✅\n"
                "Tourist lands after 22:00. After customs it is midnight or later.\n"
                "DO NOT write a Morning section for Day 1.\n"
                "DO NOT write an Afternoon section for Day 1.\n"
                "DO NOT write an Evening section for Day 1.\n"
                "Use ONLY a Night section (heading: **Night:**) for Day 1. Keep it very short.\n"
                "Day 1 Night: Land at BIA after 10pm. Clear customs (~1 hr). Transfer to Negombo "
                "or a hotel near Katunayake airport (10–15 min). Check in. Sleep.\n"
                "NO Food Today section for Day 1 — tourist just wants to sleep.\n"
                "Day 2 is the real start with full Morning + Afternoon + Evening and all meals."
            ),
        }


def decide_travel_style(interests: list, budget: str) -> dict:
    if "Luxury" in budget:
        stay = "5-star resorts and luxury boutique hotels with private transport"
        cost_level = "high"
    elif "Budget" in budget:
        stay = "budget-friendly hotels and guesthouses with PickMe/tuk-tuk transport"
        cost_level = "low"
    else:
        stay = "comfortable mid-range hotels and PickMe/Uber transport"
        cost_level = "medium"

    if "Hiking & Adventure" in interests:
        pace = "active and fast-paced"
    elif "Relaxation" in interests:
        pace = "slow and relaxing"
    else:
        pace = "balanced"

    focus = []
    if "Beaches" in interests:            focus.append("coastal areas")
    if "History & Culture" in interests:  focus.append("ancient cities")
    if "Wildlife" in interests:           focus.append("national parks")
    if "Hiking & Adventure" in interests: focus.append("hill country")
    if "Food & Cuisine" in interests:     focus.append("local food hotspots")

    return {
        "stay":       stay,
        "pace":       pace,
        "cost_level": cost_level,
        "focus":      ", ".join(focus) if focus else "general sightseeing",
    }


def check_goal_achievement(itinerary: str) -> dict:
    checks = {
        "Day structure":        "Day 1" in itinerary,
        "Cost estimates":       "USD" in itinerary or "LKR" in itinerary,
        "Food recommendations": "Food Today" in itinerary or "food" in itinerary.lower(),
        "Travel tips":          "Travel Tips" in itinerary or "Tips" in itinerary,
        "Time sections":        "Morning" in itinerary and "Evening" in itinerary,
    }
    passed = sum(checks.values())
    total  = len(checks)

    if passed == total:
        status = "complete"
        label  = f"Goal achieved — all {total}/{total} criteria met"
    elif passed >= 3:
        status = "partial"
        label  = f"Mostly complete — {passed}/{total} criteria met"
    else:
        status = "incomplete"
        label  = f"Incomplete — only {passed}/{total} criteria met"

    return {"checks": checks, "passed": passed, "total": total,
            "status": status, "label": label}


def clean_text(text: str) -> str:
    return text.replace("$", "USD ").replace("**", "")


def extract_place_names(itinerary: str) -> list:
    """
    Extract only place names that appear as actual destinations in the itinerary.
    """
    known_places = [
        "Colombo", "Kandy", "Galle", "Sigiriya", "Ella", "Nuwara Eliya",
        "Mirissa", "Unawatuna", "Trincomalee", "Anuradhapura", "Polonnaruwa",
        "Dambulla", "Negombo", "Hikkaduwa", "Arugam Bay", "Yala", "Udawalawe",
        "Horton Plains", "Adam's Peak", "Pinnawala", "Bentota", "Matara",
        "Jaffna", "Batticaloa", "Haputale", "Nanu Oya", "Badulla", "Ratnapura",
        "Tangalle", "Weligama", "Koggala", "Ahungalla", "Beruwala", "Kalpitiya",
        "Wilpattu", "Minneriya", "Knuckles", "Kitulgala", "Sinharaja",
        "Hatton", "Talawakele", "Bandarawela", "Welimada", "Mahiyanganaya",
        "Katunayake", "Chilaw", "Puttalam", "Kurunegala", "Hambantota",
        "Tissamaharama", "Weeraketiya", "Dickwella", "Ambalangoda",
        "Aluthgama", "Kalutara", "Panadura", "Moratuwa", "Nugegoda",
        "Ampara", "Monaragala", "Wellawaya", "Embilipitiya", "Ratnapura",
        "Avissawella", "Nuwara Eliya", "Pelmadulla", "Balangoda",
        "Nilaveli", "Uppuveli", "Pigeon Island", "Koneswaram",
        "Mihintale", "Wilpattu", "Bundala", "Kanniya",
    ]

    FOOD_LINE_RE = re.compile(
        r'(breakfast|lunch|dinner|must.?try|food today|🍽)',
        re.I,
    )

    FOOD_WORD_RE = re.compile(
        r'\b(curry|crab|prawn|fish|seafood|sambol|roti|hopper|juice|cake|'
        r'rice|dish|recipe|cuisine|snack|stall|eatery|restaurant|buffet|meal|drink)\b',
        re.I,
    )

    found = []
    lines = itinerary.split('\n')

    for place in known_places:
        place_re = re.compile(r'\b' + re.escape(place) + r'\b', re.I)

        for line in lines:
            clean = re.sub(r'\*+', '', line).strip()
            lower = clean.lower()

            if FOOD_LINE_RE.search(lower):
                continue

            check = clean

            # Fix: "Galle Face Green" is in Colombo — don't map it as the city of Galle
            if place == "Galle":
                check = re.sub(r"\bGalle\s+Face\b", "", clean, flags=re.I)
                if not place_re.search(check):
                    continue

            if place == "Adam's Peak":
                # Strip "Little Adam's Peak" first
                check = re.sub(r"\bLittle\s+Adam's\s+Peak\b", "", clean, flags=re.I)
                # Strip comparison references like "compared to Adam's Peak", "easier than Adam's Peak"
                check = re.sub(r'\b(compared to|than|unlike|vs\.?|versus|easier than|harder than)\s+Adam\'s\s+Peak\b', "", check, flags=re.I)
                # Strip "version of / alternative to Adam's Peak" phrases
                check = re.sub(r'\b(version of|alternative to|substitute for)\s+Adam\'s\s+Peak\b', "", check, flags=re.I)
                if not place_re.search(check):
                    continue

            if place_re.search(check):
                if place not in found:
                    found.append(place)
                break

    return found


# ── Place coordinates ─────────────────────────────────────────────────────────
PLACE_COORDS = {
    "Colombo":       (6.9271,  80.0000),
    "Kandy":         (7.2906,  80.6337),
    "Galle":         (6.0535,  80.2210),
    "Sigiriya":      (7.9570,  80.7603),
    "Ella":          (6.8667,  81.0466),
    "Nuwara Eliya":  (6.9497,  80.7891),
    "Mirissa":       (5.9483,  80.4716),
    "Unawatuna":     (6.0108,  80.2498),
    "Trincomalee":   (8.5874,  81.2152),
    "Anuradhapura":  (8.3114,  80.4037),
    "Polonnaruwa":   (7.9403,  81.0188),
    "Dambulla":      (7.8742,  80.6511),
    "Negombo":       (7.2095,  79.8386),
    "Hikkaduwa":     (6.1395,  80.1002),
    "Arugam Bay":    (6.8395,  81.8353),
    "Yala":          (6.3729,  81.5213),
    "Udawalawe":     (6.4748,  80.8992),
    "Horton Plains": (6.8021,  80.8103),
    "Adam's Peak":   (6.8096,  80.4994),
    "Pinnawala":     (7.3003,  80.3861),
    "Bentota":       (6.4248,  79.9956),
    "Matara":        (5.9549,  80.5550),
    "Jaffna":        (9.6615,  80.0255),
    "Batticaloa":    (7.7170,  81.6924),
    "Haputale":      (6.7667,  80.9667),
    "Tangalle":      (6.0249,  80.7997),
    "Weligama":      (5.9749,  80.4296),
    "Hatton":        (6.8953,  80.5950),
    "Bandarawela":   (6.8297,  81.0007),
    "Wilpattu":      (8.4560,  79.8880),
    "Minneriya":     (8.0292,  80.8991),
    "Kitulgala":     (6.9896,  80.4171),
    "Sinharaja":     (6.3953,  80.4584),
    "Kalpitiya":     (8.2333,  79.7667),
    "Koggala":       (5.9942,  80.3284),
    "Katunayake":    (7.1696,  79.8878),
    "Chilaw":        (7.5760,  79.7953),
    "Puttalam":      (8.0408,  79.8394),
    "Kurunegala":    (7.4818,  80.3609),
    "Hambantota":    (6.1241,  81.1185),
    "Tissamaharama": (6.2858,  81.2877),
    "Dickwella":     (5.9716,  80.6957),
    "Ambalangoda":   (6.2337,  80.0561),
    "Aluthgama":     (6.4329,  79.9994),
    "Kalutara":      (6.5854,  79.9607),
    "Beruwala":      (6.4785,  79.9828),
    "Ampara":        (7.2975,  81.6724),
    "Monaragala":    (6.8728,  81.3506),
    "Wellawaya":     (6.7333,  81.1000),
    "Embilipitiya":  (6.3500,  80.8500),
    "Balangoda":     (6.6500,  80.7000),
    "Avissawella":   (6.9500,  80.2167),
    "Nilaveli":      (8.6833,  81.2000),
    "Uppuveli":      (8.6167,  81.2167),
    "Pigeon Island": (8.7000,  81.2000),
    "Koneswaram":    (8.5850,  81.2330),
    "Mihintale":     (8.3500,  80.5000),
    "Bundala":       (6.1500,  81.2000),
    "Kanniya":       (8.6167,  81.1833),
}


def get_place_locations(place_names: list) -> list:
    locations = []
    for name in place_names:
        if name in PLACE_COORDS:
            lat, lon = PLACE_COORDS[name]
            locations.append({"name": name, "latitude": lat, "longitude": lon})
    return locations


# ── Main Agent Functions ──────────────────────────────────────────────────────
def plan_trip(
    days: int,
    interests: list,
    budget: str,
    arrival_time: str = "morning",
    extra_info: str = "",
    memory_context: str = "",
) -> tuple[str, dict]:
    client = _get_client()
    if client is None:
        return "❌ Groq client not available. Check your API key.", {}

    style   = decide_travel_style(interests, budget)
    arrival = decide_arrival_context(arrival_time)
    interests_str = ", ".join(interests)

    arrival_labels = {
        "morning":   "Morning (before 12 pm) — full day available from the moment they land",
        "afternoon": "Afternoon (12 pm – 6 pm) — Day 1 starts from Afternoon section only",
        "evening":   "Evening (6 pm – 10 pm) — Day 1 starts from Evening section only",
        "night":     "Night (after 10 pm) — Day 1 has Night section only, pure rest",
    }

    user_prompt = f"""
Please plan a {days}-day Sri Lanka travel itinerary for me!

My interests: {interests_str}
My budget: {budget}
My arrival time at Bandaranaike International Airport: {arrival_labels.get(arrival_time, arrival_time)}

AI Travel Style Decisions:
- Accommodation: {style['stay']}
- Travel pace: {style['pace']}
- Cost level: {style['cost_level']}
- Focus areas: {style['focus']}

═══════════ DAY 1 STRUCTURE INSTRUCTION — MUST FOLLOW EXACTLY ═══════════
{arrival['day1_instruction']}
═════════════════════════════════════════════════════════════════════════

═══════════ INTRA-DAY FEASIBILITY CHECK — APPLY TO EVERY DAY ═══════════
Before writing EACH day, mentally run this checklist:
1. List all Morning / Afternoon / Evening activities you plan to include.
2. Check the Sub-Location Travel Time Table in the system prompt for travel time between each.
3. If Morning→Afternoon travel exceeds 45 min → split across days.
4. If Afternoon→Evening travel exceeds 45 min → split across days.
5. If Morning activity is a full-day hike (Adam's Peak, Ella Rock, Horton Plains,
   Knuckles, Sinharaja) → Afternoon must be light (short stroll, rest, nearby cafe only).
6. NEVER place Adam's Peak and Ella activities on the same day — they are 90 km apart.
═════════════════════════════════════════════════════════════════════════

═══════════ ACCOMMODATION — MUST FOLLOW EXACTLY ════════════════════════
Always recommend 3–4 real hotel names per destination from the verified list.
Budget tier for this trip: {budget}
Always name the hotel in the Cost section.
NEVER say "guesthouse", "hostel", or "or similar".
═════════════════════════════════════════════════════════════════════════

═══════════ ACTIVITIES — MUST FOLLOW EXACTLY ═══════════════════════════
Use the Destination Activity Seed List in the system prompt as a starting point.
Always include at least one hidden gem per destination.
Be specific — name exact attractions, state opening times where known, and include entry costs.
═════════════════════════════════════════════════════════════════════════

Geographic Routing Instruction (MUST FOLLOW):
Plan the route as a one-directional journey — no backtracking. Group nearby attractions together.
Default flow (adjust per interests):
Airport area -> Cultural Triangle -> Kandy -> Hill Country -> South Coast -> Colombo departure.

Additional info: {extra_info if extra_info else "None"}

{memory_context}

IMPORTANT: Mention specific Sri Lanka place names clearly in each day so they can be plotted on a map.
Create an amazing, practical, geographically smart day-by-day itinerary!
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    result = clean_text(response.choices[0].message.content)
    goal   = check_goal_achievement(result)
    return result, goal


def refine_trip(itinerary: str, refinement_request: str) -> tuple[str, dict]:
    client = _get_client()
    if client is None:
        return "❌ Groq client not available. Check your API key.", {}

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": REFINE_PROMPT},
            {"role": "user",   "content": (
                f"Here is my current itinerary:\n{itinerary}\n\n"
                f"Please make this change: {refinement_request}\n\n"
                "IMPORTANT: Mention specific Sri Lanka place names clearly in each day.\n"
                "IMPORTANT: Before finalising each day, verify all Morning→Afternoon→Evening "
                "activity transitions are within 45 minutes of each other using the "
                "Sub-Location Travel Time Table.\n"
                "IMPORTANT: Recommend 3–4 real hotel names per destination from the verified list.\n"
                "IMPORTANT: Always name the hotel in the Cost section — never say 'or similar'."
            )},
        ],
    )

    result = clean_text(response.choices[0].message.content)
    goal   = check_goal_achievement(result)
    return result, goal


def chat_with_agent(messages: list, user_message: str, itinerary: str) -> tuple[str, list]:
    client = _get_client()
    if client is None:
        return "❌ Groq client not available. Check your API key.", messages

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": CHAT_PROMPT + f"\n\nUser's itinerary:\n{itinerary}"},
            {"role": "user",   "content": user_message},
        ],
    )
    reply = clean_text(response.choices[0].message.content)
    messages.append({"role": "user",      "content": user_message})
    messages.append({"role": "assistant", "content": reply})
    return reply, messages