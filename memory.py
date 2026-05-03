"""
memory.py - Saves past trip plans to SQLite and provides context for the agent.
"""

import sqlite3
from datetime import datetime
from collections import Counter

DB_PATH = "travellk.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            days INTEGER,
            interests TEXT,
            budget TEXT,
            itinerary TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_trip(days, interests, budget, itinerary):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO trips (days, interests, budget, itinerary, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (days, interests, budget, itinerary, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_recent_trips(limit=5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT days, interests, budget, timestamp, itinerary
        FROM trips ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_total_trips():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM trips")
    count = c.fetchone()[0]
    conn.close()
    return count


def get_memory_context(limit=3):
    """
    Returns a formatted string of recent trips to inject into AI prompts.
    This is how the agent 'remembers' what kind of trips the user has planned before.
    """
    recent = get_recent_trips(limit)
    if not recent:
        return ""
    lines = ["User's previous trip history (use this to personalise recommendations):"]
    for i, (days, interests, budget, timestamp, _) in enumerate(recent, 1):
        date_str = timestamp[:10] if timestamp else "unknown date"
        lines.append(f"  Trip {i}: {days} days | {interests} | {budget} | planned on {date_str}")
    return "\n".join(lines)


def load_itinerary(trip_index):
    """Load a past itinerary by index (0 = most recent)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT days, interests, budget, itinerary, timestamp
        FROM trips ORDER BY id DESC LIMIT ? OFFSET ?
    """, (1, trip_index))
    row = c.fetchone()
    conn.close()
    return row


init_db()

def get_user_preferences():
    """
    Analyses past trips to figure out what the user prefers.
    Returns preferred budget, average trip length, and top interests.
    Used to auto-personalise future itineraries.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT interests, budget, days FROM trips")
    rows = c.fetchall()
    conn.close()

    if len(rows) < 2:
        return None  # not enough data yet

    # Most common budget tier
    budgets  = Counter(r[1] for r in rows)
    preferred_budget = budgets.most_common(1)[0][0]

    # Average trip length
    avg_days = round(sum(r[2] for r in rows) / len(rows))

    # Most common interests across all trips
    all_interests = []
    for r in rows:
        all_interests.extend([i.strip() for i in r[0].split(",")])
    top_interests = [i for i, _ in Counter(all_interests).most_common(3)]

    return {
        "preferred_budget":  preferred_budget,
        "avg_trip_length":   avg_days,
        "top_interests":     top_interests,
    }


def get_smart_memory_context(limit=3):
    """
    Enhanced version of get_memory_context().
    Not only injects past trips but also adds a preference summary
    so the LLM knows what this user consistently enjoys.

    Replace get_memory_context() with this in agent.py for smarter personalisation.
    """
    recent = get_recent_trips(limit)
    prefs  = get_user_preferences()

    if not recent:
        return ""

    lines = ["User's previous trip history (use this to personalise recommendations):"]
    for i, (days, interests, budget, timestamp, _) in enumerate(recent, 1):
        date_str = timestamp[:10] if timestamp else "unknown date"
        lines.append(
            f"  Trip {i}: {days} days | {interests} | {budget} | planned on {date_str}"
        )

    if prefs:
        lines.append("")
        lines.append("Based on their history, this user PREFERS:")
        lines.append(f"  - Budget tier: {prefs['preferred_budget']}")
        lines.append(f"  - Average trip length: {prefs['avg_trip_length']} days")
        lines.append(f"  - Top interests: {', '.join(prefs['top_interests'])}")
        lines.append(
            "Use these preferences to make smarter, more personalised suggestions."
        )

    return "\n".join(lines)


def save_trip_with_rating(days, interests, budget, itinerary, rating=None):
    """
    Enhanced save_trip() that also stores a user rating (1-5 stars).
    Requires adding a 'rating' column to the trips table.

    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Add rating column if it doesn't exist yet
    try:
        c.execute("ALTER TABLE trips ADD COLUMN rating INTEGER")
        conn.commit()
    except Exception:
        pass  # column already exists

    c.execute("""
        INSERT INTO trips (days, interests, budget, itinerary, timestamp, rating)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (days, interests, budget, itinerary,
          __import__('datetime').datetime.now().isoformat(), rating))
    conn.commit()
    conn.close()


def get_top_rated_trips(min_rating=4, limit=3):
    """
    Returns the highest rated past trips.
    Useful for showing the user their best itineraries.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT days, interests, budget, timestamp, itinerary, rating
            FROM trips
            WHERE rating >= ?
            ORDER BY rating DESC, id DESC
            LIMIT ?
        """, (min_rating, limit))
        rows = c.fetchall()
    except Exception:
        rows = []
    conn.close()
    return rows


def get_destination_frequency():
    """
    Counts how many times each destination appears across all saved itineraries.
    Shows the user their most visited places.
    Useful for: 'You always go to Ella — want to try somewhere new this time?'
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT itinerary FROM trips")
    rows = c.fetchall()
    conn.close()

    from agent import extract_place_names  
    destination_counts = Counter()

    for (itinerary,) in rows:
        places = extract_place_names(itinerary)
        destination_counts.update(places)

    return destination_counts.most_common(10)