"""
Inner Mirror — Reflective Writing Analysis
A Streamlit app that analyzes poetry and personal writing to reveal
emotional patterns, potential diagnostic themes, and curated quotes.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from datetime import datetime, date
import json
import math

import db
from analysis import analyze_text, extract_word_frequencies, STOP_WORDS

# ── Page config ───────────────────────────────────────

st.set_page_config(
    page_title="Inner Mirror — Reflective Writing Analysis",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────

st.markdown("""
<style>
    /* Global font & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sage green accent */
    .sage-header {
        background: linear-gradient(135deg, #e8f0ec 0%, #d4e4db 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #c3d5ca;
    }
    .sage-header h1 {
        color: #2d6a4f;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }
    .sage-header p {
        color: #52796f;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Stats cards */
    .stat-card {
        background: #f8faf9;
        border: 1px solid #d4e4db;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-card .number {
        font-size: 1.75rem;
        font-weight: 700;
        color: #2d6a4f;
    }
    .stat-card .label {
        font-size: 0.75rem;
        color: #6b8f7e;
        margin-top: 2px;
    }

    /* Quote cards */
    .quote-card {
        background: #f8faf9;
        border-left: 3px solid #52796f;
        padding: 0.85rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.6rem;
    }
    .quote-card .text {
        font-style: italic;
        color: #333;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .quote-card .author {
        color: #52796f;
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 0.35rem;
    }

    /* Entry card */
    .entry-card {
        background: #fafcfb;
        border: 1px solid #d4e4db;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }

    /* Emotion badge */
    .emotion-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 4px;
        margin-bottom: 4px;
    }

    /* Summary reflection box */
    .reflection-box {
        background: linear-gradient(135deg, #edf5f0 0%, #e0ece5 100%);
        border: 1px solid #b8d4c4;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .reflection-box .label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #2d6a4f;
        margin-bottom: 0.5rem;
    }
    .reflection-box .text {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #333;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.7rem;
        color: #888;
        font-style: italic;
    }

    /* Attribution */
    .attribution {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-size: 0.75rem;
    }
    .attribution a { color: #52796f; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────

if "user" not in st.session_state:
    st.session_state.user = None
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None


# ── Helper: get API key ──────────────────────────────

def get_api_key():
    """Get Anthropic API key from secrets or env."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")


# ── Visualization helpers ─────────────────────────────

def render_word_cloud(word_freqs, height=280):
    """Render a word cloud from [{word, count}, ...] list."""
    if not word_freqs:
        st.info("No words to display.")
        return
    freq_dict = {w["word"]: w["count"] for w in word_freqs[:20]}
    wc = WordCloud(
        width=800,
        height=height,
        background_color="white",
        colormap="BuGn",
        max_words=20,
        prefer_horizontal=0.7,
        min_font_size=14,
        max_font_size=80,
        relative_scaling=0.5,
    ).generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(8, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)


def render_emotion_radar(emotions):
    """Render a radar/polar chart for emotions."""
    if not emotions:
        return
    labels = [e["emotion"] for e in emotions]
    values = [e["intensity"] for e in emotions]
    colors = [e.get("color", "#52796f") for e in emotions]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(45, 106, 79, 0.15)",
        line=dict(color="#2d6a4f", width=2),
        marker=dict(size=6, color=colors + [colors[0]]),
        name="Emotions",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=9, color="#888"),
                gridcolor="#e0e0e0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#333"),
            ),
            bgcolor="white",
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=20, b=20),
        height=350,
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_disorder_chart(disorders):
    """Render a horizontal bar chart for disorder relevance."""
    if not disorders:
        return
    names = [d["disorder"] for d in disorders]
    values = [d["relevance"] for d in disorders]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, "#d4e4db"], [0.5, "#52796f"], [1, "#2d6a4f"]],
            line=dict(width=0),
        ),
        text=[f"{v}%" for v in values],
        textposition="auto",
        textfont=dict(size=11, color="white"),
    ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 100], title="Relevance Score",
            gridcolor="#f0f0f0", titlefont=dict(size=11),
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        margin=dict(l=10, r=20, t=10, b=40),
        height=max(200, len(names) * 45 + 60),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_quotes(quotes):
    """Render quote cards."""
    for q in quotes:
        st.markdown(f"""
        <div class="quote-card">
            <div class="text">"{q['text']}"</div>
            <div class="author">— {q['author']}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Cumulative helpers ────────────────────────────────

def compute_cumulative_words(entries):
    aggregated = {}
    for entry in entries:
        for wf in entry["word_frequencies"]:
            w = wf["word"]
            if w not in STOP_WORDS:
                aggregated[w] = aggregated.get(w, 0) + wf["count"]
    sorted_words = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:20]
    return [{"word": w, "count": c} for w, c in sorted_words]


def compute_cumulative_emotions(entries):
    n = len(entries)
    if n == 0:
        return []
    emotion_map = {}
    for entry in entries:
        for em in entry["emotions"]:
            name = em["emotion"]
            if name not in emotion_map:
                emotion_map[name] = {"total": 0, "count": 0, "color": em.get("color", "#52796f")}
            emotion_map[name]["total"] += em["intensity"]
            emotion_map[name]["count"] += 1
    result = []
    for emotion, data in emotion_map.items():
        if n == 1:
            intensity = round(data["total"] / data["count"])
        else:
            intensity = max(round(data["total"] / n), 1)
        result.append({"emotion": emotion, "intensity": intensity, "color": data["color"]})
    result.sort(key=lambda x: x["intensity"], reverse=True)
    return result[:10]


def compute_top_emotions(entries):
    n = len(entries)
    if n == 0:
        return []
    emotion_map = {}
    for entry in entries:
        for em in entry["emotions"]:
            name = em["emotion"]
            if name not in emotion_map:
                emotion_map[name] = {"total": 0, "count": 0, "color": em.get("color", "#52796f")}
            emotion_map[name]["total"] += em["intensity"]
            emotion_map[name]["count"] += 1
    result = []
    for emotion, data in emotion_map.items():
        result.append({
            "emotion": emotion,
            "avg_intensity": round(data["total"] / data["count"]),
            "recurrence_rate": round((data["count"] / n) * 100),
            "occurrences": data["count"],
            "color": data["color"],
        })
    result.sort(key=lambda x: (-x["recurrence_rate"], -x["avg_intensity"]))
    return result[:10]


def compute_cumulative_disorders(entries):
    n = len(entries)
    if n == 0:
        return []
    disorder_map = {}
    for entry in entries:
        for d in entry["disorders"]:
            name = d["disorder"]
            if name not in disorder_map:
                disorder_map[name] = {
                    "total_relevance": 0, "count": 0,
                    "symptoms": set(), "descriptions": [],
                }
            disorder_map[name]["total_relevance"] += d["relevance"]
            disorder_map[name]["count"] += 1
            for s in d.get("symptoms", []):
                disorder_map[name]["symptoms"].add(s)
            desc = d.get("description", "")
            if desc and desc not in disorder_map[name]["descriptions"]:
                disorder_map[name]["descriptions"].append(desc)
    result = []
    for disorder, data in disorder_map.items():
        if n == 1:
            relevance = round(data["total_relevance"] / data["count"])
        else:
            relevance = max(round(data["total_relevance"] / n), 1)
        pct = round((data["count"] / n) * 100)
        desc = f"Detected in {data['count']} of {n} {'analysis' if n == 1 else 'analyses'} ({pct}%)."
        if data["descriptions"]:
            desc += f" {data['descriptions'][0]}"
        result.append({
            "disorder": disorder,
            "relevance": relevance,
            "symptoms": list(data["symptoms"])[:6],
            "description": desc,
        })
    result.sort(key=lambda x: x["relevance"], reverse=True)
    return result[:8]


# ── PAGES ─────────────────────────────────────────────

def page_auth():
    """Login / Register page."""
    # If already logged in, go home
    if st.session_state.user:
        st.session_state.page = "home"
        st.rerun()
        return

    st.markdown("""
    <div class="sage-header" style="text-align:center; max-width:450px; margin:4rem auto 2rem auto;">
        <h1>🪞 Inner Mirror</h1>
        <p>Sign in to access your analysis history</p>
    </div>
    """, unsafe_allow_html=True)

    col_spacer1, col_form, col_spacer2 = st.columns([1, 1.5, 1])
    with col_form:
        tab_login, tab_register = st.tabs(["Sign In", "Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        user = db.login(username, password)
                        if user:
                            st.session_state.user = user
                            st.session_state.page = "home"
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")

        with tab_register:
            with st.form("register_form"):
                new_user = st.text_input("Choose a username", key="reg_user")
                new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
                submitted = st.form_submit_button("Register", use_container_width=True)
                if submitted:
                    if not new_user or not new_pass:
                        st.error("Please fill in both fields.")
                    elif len(new_pass) < 4:
                        st.error("Password must be at least 4 characters.")
                    else:
                        user = db.register(new_user, new_pass)
                        if user:
                            st.session_state.user = user
                            st.session_state.page = "home"
                            st.rerun()
                        else:
                            st.error("Username already taken.")


def page_home():
    """Main analysis page."""
    # Header
    user = st.session_state.user
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown("""
        <div class="sage-header">
            <h1>🪞 Inner Mirror</h1>
            <p>Reflective writing analysis</p>
        </div>
        """, unsafe_allow_html=True)
    with hdr_right:
        st.write("")  # spacing
        nav_cols = st.columns(3)
        with nav_cols[0]:
            if st.button("📜 History", use_container_width=True):
                if user:
                    st.session_state.page = "history"
                    st.rerun()
                else:
                    st.session_state.page = "auth"
                    st.rerun()
        with nav_cols[1]:
            if user:
                st.markdown(f"<div style='text-align:center;padding-top:8px;font-size:0.85rem;color:#52796f;font-weight:500;'>{user['username']}</div>", unsafe_allow_html=True)
            else:
                if st.button("🔑 Sign In", use_container_width=True):
                    st.session_state.page = "auth"
                    st.rerun()
        with nav_cols[2]:
            if user:
                if st.button("Sign Out", use_container_width=True):
                    st.session_state.user = None
                    st.session_state.current_analysis = None
                    st.rerun()

    # Input section
    st.subheader("Share your writing")
    if user:
        st.caption("Paste poetry, journal entries, or any personal writing. Your analysis will be saved to your history.")
    else:
        st.caption("Paste poetry, journal entries, or any personal writing. Sign in to save analyses and track patterns over time.")

    text_input = st.text_area(
        "Your writing",
        placeholder="Paste your poetry, journal entry, or personal writing here...",
        height=200,
        label_visibility="collapsed",
    )
    col_date, col_spacer, col_btn = st.columns([1, 2, 1])
    with col_date:
        date_written = st.date_input("Date written (optional)", value=None)
    with col_btn:
        st.write("")  # spacing
        analyze_clicked = st.button("✨ Analyze Writing", type="primary", use_container_width=True)

    if analyze_clicked:
        api_key = get_api_key()
        if not api_key:
            st.error("No Anthropic API key found. Please set ANTHROPIC_API_KEY in your Streamlit secrets or environment variables.")
            return
        if not text_input or len(text_input.strip()) < 10:
            st.warning("Please provide at least 10 characters of text.")
            return

        with st.spinner("Analyzing your writing..."):
            try:
                result = analyze_text(text_input.strip(), api_key)
                dw = str(date_written) if date_written else datetime.now().strftime("%Y-%m-%d")
                result["dateWritten"] = dw
                result["dateAnalyzed"] = datetime.now().isoformat()

                # Save if logged in
                if user:
                    aid = db.save_analysis(
                        user_id=user["id"],
                        text=text_input.strip(),
                        date_written=dw,
                        date_analyzed=result["dateAnalyzed"],
                        summary=result["summary"],
                        emotions=result["emotions"],
                        disorders=result["disorders"],
                        quotes=result["quotes"],
                        word_frequencies=result["wordFrequencies"],
                    )
                    result["id"] = aid

                st.session_state.current_analysis = result
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return

    # Display results
    analysis = st.session_state.current_analysis
    if analysis:
        st.divider()

        # Summary
        st.markdown(f"""
        <div class="reflection-box">
            <div class="label">🪞 Reflection</div>
            <div class="text">{analysis['summary']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Word Cloud + Emotion Radar side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Word Cloud**")
            render_word_cloud(analysis["wordFrequencies"])
        with col2:
            st.markdown("**Emotional Landscape**")
            render_emotion_radar(analysis["emotions"])

        # Disorders
        st.markdown("**Potential Symptom Patterns**")
        st.markdown('<p class="disclaimer">Not a diagnosis. Patterns that may warrant reflection or professional consultation.</p>', unsafe_allow_html=True)
        render_disorder_chart(analysis["disorders"])

        # Symptoms detail
        for d in analysis["disorders"]:
            with st.expander(f"{d['disorder']} — {d['relevance']}% relevance"):
                st.write(d.get("description", ""))
                if d.get("symptoms"):
                    st.write("**Observed symptoms:** " + ", ".join(d["symptoms"]))

        # Quotes
        st.markdown("**Words for Reflection**")
        render_quotes(analysis.get("quotes", []))

    elif not analyze_clicked:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#888;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📝</div>
            <p>Paste your writing above and click "Analyze Writing" to explore the emotional patterns within your words.</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="attribution">
        <a href="https://www.perplexity.ai/computer" target="_blank" rel="noopener noreferrer">
            Created with Perplexity Computer
        </a>
    </div>
    """, unsafe_allow_html=True)


def page_history():
    """History dashboard with cumulative analytics."""
    user = st.session_state.user
    if not user:
        st.session_state.page = "auth"
        st.rerun()
        return

    # Header
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown("""
        <div class="sage-header">
            <h1>🪞 Inner Mirror</h1>
            <p>Reflective writing analysis</p>
        </div>
        """, unsafe_allow_html=True)
    with hdr_right:
        st.write("")
        nav_cols = st.columns(3)
        with nav_cols[0]:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
        with nav_cols[1]:
            st.markdown(f"<div style='text-align:center;padding-top:8px;font-size:0.85rem;color:#52796f;font-weight:500;'>{user['username']}</div>", unsafe_allow_html=True)
        with nav_cols[2]:
            if st.button("Sign Out", use_container_width=True, key="hist_signout"):
                st.session_state.user = None
                st.session_state.page = "home"
                st.rerun()

    st.subheader("Analysis History")
    st.caption("Browse past analyses, track cumulative emotional and diagnostic patterns, and explore your word map over time.")

    # Filters
    with st.expander("🔍 Search & Filter", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            search_text = st.text_input("Search your writing", key="hist_search")
        with fcol2:
            start_date = st.date_input("From date", value=None, key="hist_start")
        with fcol3:
            end_date = st.date_input("To date", value=None, key="hist_end")

    # Fetch entries
    entries = db.get_analyses(
        user["id"],
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
        search=search_text if search_text else None,
    )

    if not entries:
        st.info("No analyses yet. Go to the home page and analyze some writing to start building your history.")
        return

    # Compute cumulative data
    cum_words = compute_cumulative_words(entries)
    cum_emotions = compute_cumulative_emotions(entries)
    top_emotions = compute_top_emotions(entries)
    cum_disorders = compute_cumulative_disorders(entries)

    # Stats row
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="number">{len(entries)}</div>
            <div class="label">{"Entry" if len(entries) == 1 else "Entries"}</div>
        </div>""", unsafe_allow_html=True)
    with scol2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="number">{len(cum_words)}</div>
            <div class="label">Unique Words</div>
        </div>""", unsafe_allow_html=True)
    with scol3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="number">{len(cum_emotions)}</div>
            <div class="label">Emotions Tracked</div>
        </div>""", unsafe_allow_html=True)
    with scol4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="number">{len(cum_disorders)}</div>
            <div class="label">Patterns Tracked</div>
        </div>""", unsafe_allow_html=True)

    st.write("")

    # Cumulative Word Map
    st.markdown(f"**Cumulative Word Map** — words aggregated across {len(entries)} {'analysis' if len(entries)==1 else 'analyses'}")
    render_word_cloud(cum_words, height=300)

    st.write("")

    # Emotion radar + Recurring emotions
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.markdown(f"**Cumulative Emotional Landscape**")
        st.caption(f"Weighted intensity across {len(entries)} {'analysis' if len(entries)==1 else 'analyses'} — emotions that recur more often score higher")
        if cum_emotions:
            render_emotion_radar(cum_emotions)
    with ecol2:
        st.markdown("**Recurring Emotions**")
        st.caption("How often each emotion recurs across your writing")
        for em in top_emotions:
            badge_html = f"""
            <div style="display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border-radius:20px; background:#f0f4f2; margin-bottom:6px;">
                <div style="width:8px;height:8px;border-radius:50%;background:{em['color']};"></div>
                <span style="font-size:0.8rem;font-weight:500;">{em['emotion']}</span>
                <span style="font-size:0.75rem;color:#666;">{em['recurrence_rate']}% recurrence</span>
                <span style="font-size:0.65rem;color:#999;">avg {em['avg_intensity']}%</span>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

    st.write("")

    # Cumulative disorders
    if cum_disorders:
        st.markdown("**Cumulative Symptom Patterns**")
        st.caption("Average relevance across all analyses. Not a diagnosis — patterns that may warrant reflection or professional consultation.")
        render_disorder_chart(cum_disorders)

    st.divider()

    # Entry list
    st.markdown("**All Entries**")
    for entry in entries:
        ecol_main, ecol_del = st.columns([20, 1])
        with ecol_main:
            # Emotion badges
            emotion_html = ""
            for em in entry["emotions"][:3]:
                emotion_html += f'<span class="emotion-badge" style="border:1px solid {em.get("color","#52796f")};color:{em.get("color","#52796f")};">{em["emotion"]}</span>'

            st.markdown(f"""
            <div class="entry-card">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px;">
                    <span style="font-size:0.8rem;font-weight:500;">📅 {entry['date_written']}</span>
                    {emotion_html}
                </div>
                <p style="font-size:0.85rem;color:#666;line-height:1.5;margin:0;">
                    {entry['summary'][:200]}{'...' if len(entry['summary'])>200 else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)
        with ecol_del:
            st.write("")  # spacing
            if st.button("🗑️", key=f"del_{entry['id']}", help="Delete this entry"):
                db.delete_analysis(entry["id"], user["id"])
                st.rerun()

        # Expandable detail
        with st.expander(f"View full analysis — {entry['date_written']}", expanded=False):
            # Reflection
            st.markdown(f"""
            <div class="reflection-box">
                <div class="label">🪞 Reflection — Written {entry['date_written']}</div>
                <div class="text">{entry['summary']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Text preview
            preview = entry["text"][:300] + ("..." if len(entry["text"]) > 300 else "")
            st.markdown(f"*\"{preview}\"*")

            # Charts
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.markdown("**Word Cloud**")
                render_word_cloud(entry["word_frequencies"], height=220)
            with dcol2:
                st.markdown("**Emotional Landscape**")
                render_emotion_radar(entry["emotions"])

            st.markdown("**Potential Symptom Patterns**")
            st.markdown('<p class="disclaimer">Not a diagnosis. Patterns that may warrant reflection or professional consultation.</p>', unsafe_allow_html=True)
            render_disorder_chart(entry["disorders"])

            st.markdown("**Words for Reflection**")
            render_quotes(entry.get("quotes", []))

            st.caption(f"Analyzed: {entry['date_analyzed'][:10]}")

    # Footer
    st.markdown("""
    <div class="attribution">
        <a href="https://www.perplexity.ai/computer" target="_blank" rel="noopener noreferrer">
            Created with Perplexity Computer
        </a>
    </div>
    """, unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    page = st.session_state.page

    if page == "auth":
        page_auth()
    elif page == "history":
        page_history()
    else:
        page_home()


# Streamlit always runs the full script
main()
