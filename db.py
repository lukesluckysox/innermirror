"""SQLite database layer for Inner Mirror."""

import sqlite3
import hashlib
import json
import os
from datetime import datetime

DB_PATH = os.environ.get("INNER_MIRROR_DB", "data.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            date_written TEXT NOT NULL,
            date_analyzed TEXT NOT NULL,
            summary TEXT NOT NULL,
            emotions TEXT NOT NULL,
            disorders TEXT NOT NULL,
            quotes TEXT NOT NULL,
            word_frequencies TEXT NOT NULL,
            political_compass TEXT DEFAULT '{}',
            mbti_profile TEXT DEFAULT '{}',
            moral_foundations TEXT DEFAULT '{}',
            word_colors TEXT DEFAULT '{}'
        )
    """)
    # Migrate: add new columns if they don't exist yet
    try:
        conn.execute("SELECT political_compass FROM analyses LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE analyses ADD COLUMN political_compass TEXT DEFAULT '{}'")
    try:
        conn.execute("SELECT mbti_profile FROM analyses LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE analyses ADD COLUMN mbti_profile TEXT DEFAULT '{}'")
    try:
        conn.execute("SELECT moral_foundations FROM analyses LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE analyses ADD COLUMN moral_foundations TEXT DEFAULT '{}'")
    try:
        conn.execute("SELECT word_colors FROM analyses LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE analyses ADD COLUMN word_colors TEXT DEFAULT '{}'")

    # Seed default account: tester / tester123
    hashed = hashlib.sha256("tester123".encode()).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
        ("tester", hashed),
    )
    conn.commit()
    conn.close()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ── Auth ──────────────────────────────────────────────

def login(username: str, password: str):
    conn = get_connection()
    hashed = hash_password(password)
    row = conn.execute(
        "SELECT id, username FROM users WHERE username=? AND password=?",
        (username, hashed),
    ).fetchone()
    conn.close()
    if row:
        return {"id": row["id"], "username": row["username"]}
    return None

def register(username: str, password: str):
    conn = get_connection()
    try:
        hashed = hash_password(password)
        cur = conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed),
        )
        conn.commit()
        uid = cur.lastrowid
        conn.close()
        return {"id": uid, "username": username}
    except sqlite3.IntegrityError:
        conn.close()
        return None

# ── Analyses ──────────────────────────────────────────

def save_analysis(user_id, text, date_written, date_analyzed, summary,
                  emotions, disorders, quotes, word_frequencies,
                  political_compass=None, mbti_profile=None,
                  moral_foundations=None, word_colors=None):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO analyses
           (user_id, text, date_written, date_analyzed, summary,
            emotions, disorders, quotes, word_frequencies,
            political_compass, mbti_profile, moral_foundations,
            word_colors)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, text, date_written, date_analyzed, summary,
         json.dumps(emotions), json.dumps(disorders),
         json.dumps(quotes), json.dumps(word_frequencies),
         json.dumps(political_compass or {}),
         json.dumps(mbti_profile or {}),
         json.dumps(moral_foundations or {}),
         json.dumps(word_colors or {})),
    )
    conn.commit()
    aid = cur.lastrowid
    conn.close()
    return aid

def get_analyses(user_id, start_date=None, end_date=None, search=None):
    conn = get_connection()
    query = "SELECT * FROM analyses WHERE user_id=?"
    params = [user_id]
    if start_date:
        query += " AND date_written >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date_written <= ?"
        params.append(end_date)
    if search:
        query += " AND text LIKE ?"
        params.append(f"%{search}%")
    query += " ORDER BY date_written DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    results = []
    for r in rows:
        pc_raw = r["political_compass"] if "political_compass" in r.keys() else "{}"
        mb_raw = r["mbti_profile"] if "mbti_profile" in r.keys() else "{}"
        mf_raw = r["moral_foundations"] if "moral_foundations" in r.keys() else "{}"
        wc_raw = r["word_colors"] if "word_colors" in r.keys() else "{}"
        results.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "text": r["text"],
            "date_written": r["date_written"],
            "date_analyzed": r["date_analyzed"],
            "summary": r["summary"],
            "emotions": json.loads(r["emotions"]),
            "disorders": json.loads(r["disorders"]),
            "quotes": json.loads(r["quotes"]),
            "word_frequencies": json.loads(r["word_frequencies"]),
            "political_compass": json.loads(pc_raw) if pc_raw else {},
            "mbti_profile": json.loads(mb_raw) if mb_raw else {},
            "moral_foundations": json.loads(mf_raw) if mf_raw else {},
            "word_colors": json.loads(wc_raw) if wc_raw else {},
        })
    return results

def delete_analysis(analysis_id, user_id):
    conn = get_connection()
    cur = conn.execute(
        "DELETE FROM analyses WHERE id=? AND user_id=?",
        (analysis_id, user_id),
    )
    conn.commit()
    changed = cur.rowcount > 0
    conn.close()
    return changed

# Initialize on import
init_db()
