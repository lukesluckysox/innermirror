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

# ── Emotion color mapping for word cloud ─────────────

# Words associated with emotional categories and their standard colors
EMOTION_WORD_MAP = {
    "sadness": {"color": "#4A6FA5", "words": [
        "sad", "sorrow", "grief", "loss", "tears", "cry", "weep", "mourn",
        "pain", "hurt", "ache", "lonely", "alone", "empty", "hollow",
        "broken", "lost", "miss", "gone", "fade", "wither", "drown",
        "bleed", "wound", "scar", "dark", "darkness", "shadow", "grey",
        "gray", "cold", "winter", "rain", "storm", "grave", "death",
        "die", "dead", "end", "fall", "fallen", "despair", "hopeless",
        "misery", "suffer", "agony", "melancholy", "gloom", "somber",
        "heavy", "burden", "weight", "chains", "cage", "cages", "trap",
        "prison", "bound", "blindness", "blind",
    ]},
    "anger": {"color": "#EB5757", "words": [
        "anger", "angry", "rage", "fury", "hate", "hatred", "fight",
        "war", "destroy", "burn", "fire", "blood", "kill", "violent",
        "scream", "shout", "curse", "bitter", "resent", "spite",
        "disgust", "revolt", "rebel", "defy", "crush", "smash",
        "frustration", "frustrated", "injustice",
    ]},
    "fear": {"color": "#9B51E0", "words": [
        "fear", "afraid", "scared", "terror", "dread", "panic",
        "anxiety", "anxious", "worry", "nervous", "threat", "danger",
        "risk", "hide", "flee", "run", "escape", "trapped", "haunted",
        "ghost", "nightmare", "horror", "phobia", "paralyzed",
        "obligations", "expectations",
    ]},
    "joy": {"color": "#F2C94C", "words": [
        "joy", "happy", "happiness", "love", "laugh", "smile", "play",
        "dance", "sing", "celebrate", "bright", "light", "sun", "warm",
        "bloom", "flower", "spring", "free", "freedom", "peace",
        "bliss", "delight", "pleasure", "wonder", "magic", "dream",
        "hope", "wish", "stars", "beautiful", "beauty", "gentle",
        "kind", "sweet", "tender", "embrace", "hold", "home",
        "celebration", "innovation",
    ]},
    "love": {"color": "#E84393", "words": [
        "love", "heart", "soul", "beloved", "darling", "dear",
        "passion", "desire", "romance", "kiss", "touch", "caress",
        "intimate", "devotion", "adore", "cherish", "bond",
        "together", "forever", "connect", "connection",
    ]},
    "contemplation": {"color": "#52796f", "words": [
        "think", "thought", "mind", "wonder", "question", "truth",
        "wisdom", "intuition", "knowledge", "understand", "meaning",
        "purpose", "exist", "existence", "life", "time", "memory",
        "remember", "forget", "past", "future", "present", "eternity",
        "infinite", "universe", "nature", "earth", "sky", "sea",
        "ocean", "river", "mountain", "vision", "society", "world",
        "living", "another", "eye", "status", "wages", "rapid",
    ]},
}

def get_emotion_color(word):
    """Return an emotion-based color for a word, or a neutral default."""
    w = word.lower()
    for emotion, data in EMOTION_WORD_MAP.items():
        if w in data["words"]:
            return data["color"]
    # Neutral default — muted teal
    return "#6b8f7e"

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
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sage green header — dark-mode compatible */
    .sage-header {
        background: rgba(45, 106, 79, 0.15);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(45, 106, 79, 0.25);
    }
    .sage-header h1 {
        color: #6fcf97;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }
    .sage-header p {
        color: #a8d5ba;
        font-size: 0.9rem;
        margin: 0;
    }

    /* Stats cards */
    .stat-card {
        background: rgba(45, 106, 79, 0.1);
        border: 1px solid rgba(45, 106, 79, 0.2);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-card .number {
        font-size: 1.75rem;
        font-weight: 700;
        color: #6fcf97;
    }
    .stat-card .label {
        font-size: 0.75rem;
        color: #a0a0a0;
        margin-top: 2px;
    }

    /* Quote cards */
    .quote-card {
        background: rgba(82, 121, 111, 0.1);
        border-left: 3px solid #6fcf97;
        padding: 0.85rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.6rem;
    }
    .quote-card .text {
        font-style: italic;
        color: #d0d0d0;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .quote-card .author {
        color: #6fcf97;
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 0.35rem;
    }

    /* Entry card */
    .entry-card {
        background: rgba(45, 106, 79, 0.08);
        border: 1px solid rgba(45, 106, 79, 0.18);
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
        background: rgba(45, 106, 79, 0.12);
        border: 1px solid rgba(45, 106, 79, 0.25);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .reflection-box .label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6fcf97;
        margin-bottom: 0.5rem;
    }
    .reflection-box .text {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #d0d0d0;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.7rem;
        color: #999;
        font-style: italic;
    }

    /* Attribution */
    .attribution {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-size: 0.75rem;
    }
    .attribution a { color: #6fcf97; }

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


# ── Quote of the Day ─────────────────────────────────

DAILY_QUOTES = [
    {"text": "The wound is the place where the Light enters you.", "author": "Rumi"},
    {"text": "Out of your vulnerabilities will come your strength.", "author": "Sigmund Freud"},
    {"text": "The soul always knows what to do to heal itself. The challenge is to silence the mind.", "author": "Caroline Myss"},
    {"text": "There is no greater agony than bearing an untold story inside you.", "author": "Maya Angelou"},
    {"text": "What we achieve inwardly will change outer reality.", "author": "Plutarch"},
    {"text": "The only journey is the one within.", "author": "Rainer Maria Rilke"},
    {"text": "To know yourself, you must sacrifice the illusion that you already do.", "author": "Vironika Tugaleva"},
    {"text": "One does not become enlightened by imagining figures of light, but by making the darkness conscious.", "author": "Carl Jung"},
    {"text": "Your task is not to seek for love, but merely to seek and find all the barriers within yourself that you have built against it.", "author": "Rumi"},
    {"text": "The curious paradox is that when I accept myself just as I am, then I can change.", "author": "Carl Rogers"},
    {"text": "In the middle of difficulty lies opportunity.", "author": "Albert Einstein"},
    {"text": "We are not what happened to us. We are what we wish to become.", "author": "Carl Jung"},
    {"text": "The privilege of a lifetime is to become who you truly are.", "author": "Carl Jung"},
    {"text": "Between stimulus and response there is a space. In that space is our power to choose our response.", "author": "Viktor Frankl"},
    {"text": "Knowing your own darkness is the best method for dealing with the darknesses of other people.", "author": "Carl Jung"},
    {"text": "Almost everything will work again if you unplug it for a few minutes, including you.", "author": "Anne Lamott"},
    {"text": "You don't have to control your thoughts. You just have to stop letting them control you.", "author": "Dan Millman"},
    {"text": "The greatest discovery of my generation is that a human being can alter his life by altering his attitudes.", "author": "William James"},
    {"text": "What lies behind us and what lies before us are tiny matters compared to what lies within us.", "author": "Ralph Waldo Emerson"},
    {"text": "Ring the bells that still can ring. Forget your perfect offering. There is a crack in everything. That's how the light gets in.", "author": "Leonard Cohen"},
    {"text": "The emotion that can break your heart is sometimes the very one that heals it.", "author": "Nicholas Sparks"},
    {"text": "You are not a drop in the ocean. You are the entire ocean in a drop.", "author": "Rumi"},
    {"text": "Healing is not an overnight process. It is a daily cleansing of pain, it is a daily healing of your life.", "author": "Leon Brown"},
    {"text": "The most beautiful people we have known are those who have known defeat, known suffering, known loss, and have found their way out of the depths.", "author": "Elisabeth Kübler-Ross"},
    {"text": "We are healed of a suffering only by experiencing it to the full.", "author": "Marcel Proust"},
    {"text": "There is a voice that doesn't use words. Listen.", "author": "Rumi"},
    {"text": "Unexpressed emotions will never die. They are buried alive and will come forth later in uglier ways.", "author": "Sigmund Freud"},
    {"text": "The best way out is always through.", "author": "Robert Frost"},
    {"text": "Stars can't shine without darkness.", "author": "D.H. Sidebottom"},
    {"text": "Vulnerability is the birthplace of innovation, creativity, and change.", "author": "Brené Brown"},
    {"text": "Your heart knows the way. Run in that direction.", "author": "Rumi"},
]

def get_daily_quote():
    """Return a quote that changes once per day."""
    day_index = date.today().toordinal() % len(DAILY_QUOTES)
    return DAILY_QUOTES[day_index]


# ── Session state init ────────────────────────────────

if "user" not in st.session_state:
    st.session_state.user = None
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None


# ── Helper: get API key ──────────────────────────────

def get_api_key():
    """Get Groq API key from secrets or env."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        import os
        return os.environ.get("GROQ_API_KEY", "")


# ── Visualization helpers ─────────────────────────────

def _emotion_color_func(word, **kwargs):
    """Color function for WordCloud — maps words to emotion colors."""
    return get_emotion_color(word)


def render_word_cloud(word_freqs, height=220):
    """Render an emotion-color-coded word cloud from [{word, count}, ...] list."""
    if not word_freqs:
        st.info("No words to display.")
        return
    freq_dict = {w["word"]: w["count"] for w in word_freqs[:20]}
    wc = WordCloud(
        width=700,
        height=height,
        background_color=None,
        mode="RGBA",
        max_words=20,
        prefer_horizontal=0.7,
        min_font_size=12,
        max_font_size=64,
        relative_scaling=0.5,
        color_func=_emotion_color_func,
    ).generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(7, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    plt.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)


def render_emotion_radar(emotions, chart_key=None):
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
                gridcolor="rgba(255,255,255,0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#ccc"),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=20, b=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_disorder_chart(disorders, chart_key=None):
    """Render a horizontal bar chart for disorder relevance with hover descriptions."""
    if not disorders:
        return
    names = [d["disorder"] for d in disorders]
    values = [d["relevance"] for d in disorders]

    # Build hover text with symptoms and description
    hover_texts = []
    for d in disorders:
        parts = [f"<b>{d['disorder']}</b> — {d['relevance']}% relevance"]
        if d.get("symptoms"):
            parts.append(f"<br><b>Symptoms:</b> {', '.join(d['symptoms'][:4])}")
        if d.get("description"):
            # Wrap long descriptions
            desc = d["description"]
            if len(desc) > 120:
                desc = desc[:120] + "..."
            parts.append(f"<br><i>{desc}</i>")
        hover_texts.append("".join(parts))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, "#3a7d5c"], [0.5, "#52796f"], [1, "#6fcf97"]],
            line=dict(width=0),
        ),
        text=[f"{v}%" for v in values],
        textposition="auto",
        textfont=dict(size=11, color="white"),
        hovertext=hover_texts,
        hoverinfo="text",
    ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            title=dict(text="Relevance Score", font=dict(size=11, color="#aaa")),
            gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#aaa"),
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#ccc")),
        margin=dict(l=10, r=20, t=10, b=40),
        height=max(200, len(names) * 50 + 60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#1a1a2e",
            bordercolor="#6fcf97",
            font=dict(size=12, color="#e0e0e0"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_quotes(quotes):
    """Render quote cards."""
    for q in quotes:
        st.markdown(f"""
        <div class="quote-card">
            <div class="text">"{q['text']}"</div>
            <div class="author">— {q['author']}</div>
        </div>
        """, unsafe_allow_html=True)


def render_political_compass(pc_data, chart_key=None):
    """Render a political compass quad chart."""
    econ = pc_data.get("economic", 0)
    social = pc_data.get("social", 0)
    label = pc_data.get("label", "")

    fig = go.Figure()

    # Quadrant background shading
    quads = [
        {"x0": -5, "x1": 0, "y0": 0, "y1": 5, "color": "rgba(235,87,87,0.08)", "label": "Authoritarian\nLeft"},
        {"x0": 0, "x1": 5, "y0": 0, "y1": 5, "color": "rgba(74,111,165,0.08)", "label": "Authoritarian\nRight"},
        {"x0": -5, "x1": 0, "y0": -5, "y1": 0, "color": "rgba(0,184,148,0.08)", "label": "Libertarian\nLeft"},
        {"x0": 0, "x1": 5, "y0": -5, "y1": 0, "color": "rgba(155,81,224,0.08)", "label": "Libertarian\nRight"},
    ]
    for q in quads:
        fig.add_shape(
            type="rect", x0=q["x0"], x1=q["x1"], y0=q["y0"], y1=q["y1"],
            fillcolor=q["color"], line=dict(width=0), layer="below",
        )

    # Quadrant labels
    label_positions = [
        {"x": -3.5, "y": 3.5, "text": "Authoritarian<br>Left", "color": "rgba(235,87,87,0.35)"},
        {"x": 3.5, "y": 3.5, "text": "Authoritarian<br>Right", "color": "rgba(74,111,165,0.35)"},
        {"x": -3.5, "y": -3.5, "text": "Libertarian<br>Left", "color": "rgba(0,184,148,0.35)"},
        {"x": 3.5, "y": -3.5, "text": "Libertarian<br>Right", "color": "rgba(155,81,224,0.35)"},
    ]
    for lp in label_positions:
        fig.add_annotation(
            x=lp["x"], y=lp["y"], text=lp["text"],
            showarrow=False, font=dict(size=10, color=lp["color"]),
            xref="x", yref="y",
        )

    # Axis lines through center
    fig.add_shape(type="line", x0=-5, x1=5, y0=0, y1=0,
                  line=dict(color="rgba(255,255,255,0.2)", width=1))
    fig.add_shape(type="line", x0=0, x1=0, y0=-5, y1=5,
                  line=dict(color="rgba(255,255,255,0.2)", width=1))

    # User's position
    fig.add_trace(go.Scatter(
        x=[econ], y=[social],
        mode="markers+text",
        marker=dict(size=14, color="#6fcf97", line=dict(width=2, color="white")),
        text=[label],
        textposition="top center",
        textfont=dict(size=11, color="#e0e0e0"),
        hovertext=f"Economic: {econ:.1f}<br>Social: {social:.1f}<br>{label}",
        hoverinfo="text",
    ))

    fig.update_layout(
        xaxis=dict(
            range=[-5.5, 5.5],
            title=dict(text="Economic Left ←  → Right", font=dict(size=10, color="#aaa")),
            tickfont=dict(size=9, color="#888"),
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        yaxis=dict(
            range=[-5.5, 5.5],
            title=dict(text="Libertarian ←  → Authoritarian", font=dict(size=10, color="#aaa")),
            tickfont=dict(size=9, color="#888"),
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        ),
        height=380,
        margin=dict(l=50, r=20, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1a1a2e", bordercolor="#6fcf97",
            font=dict(size=12, color="#e0e0e0"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_mbti_radar(mbti_data, chart_key=None):
    """Render an MBTI profile as a radar/web chart."""
    mbti_type = mbti_data.get("type", "????")

    # 8 dimensions on the radar
    dims = ["E", "I", "S", "N", "T", "F", "J", "P"]
    dim_labels = [
        "Extraversion", "Introversion",
        "Sensing", "Intuition",
        "Thinking", "Feeling",
        "Judging", "Perceiving",
    ]
    values = [mbti_data.get(d, 50) for d in dims]

    # Colors for each dimension
    dim_colors = [
        "#F2C94C", "#4A6FA5",  # E=gold, I=blue
        "#00B894", "#9B51E0",  # S=green, N=purple
        "#EB5757", "#E84393",  # T=red, F=pink
        "#636E72", "#FDCB6E",  # J=gray, P=amber
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=dim_labels + [dim_labels[0]],
        fill="toself",
        fillcolor="rgba(111, 207, 151, 0.15)",
        line=dict(color="#6fcf97", width=2),
        marker=dict(size=7, color=dim_colors + [dim_colors[0]]),
        name="MBTI",
        hovertext=[f"{l}: {v}%" for l, v in zip(dim_labels, values)] + [f"{dim_labels[0]}: {values[0]}%"],
        hoverinfo="text",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=8, color="#888"),
                gridcolor="rgba(255,255,255,0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#ccc"),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=30, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text=f"<b>{mbti_type}</b>",
                x=0.5, y=1.05, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=18, color="#6fcf97"),
            )
        ],
        hoverlabel=dict(
            bgcolor="#1a1a2e", bordercolor="#6fcf97",
            font=dict(size=12, color="#e0e0e0"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


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


def compute_cumulative_political_compass(entries):
    """Average political compass coordinates across all entries."""
    econ_vals = []
    social_vals = []
    for e in entries:
        pc = e.get("political_compass", {})
        if pc and "economic" in pc and "social" in pc:
            try:
                econ_vals.append(float(pc["economic"]))
                social_vals.append(float(pc["social"]))
            except (ValueError, TypeError):
                pass
    if not econ_vals:
        return None
    avg_econ = sum(econ_vals) / len(econ_vals)
    avg_social = sum(social_vals) / len(social_vals)
    # Determine label
    lr = "Left" if avg_econ < -0.5 else ("Right" if avg_econ > 0.5 else "Center")
    al = "Libertarian" if avg_social < -0.5 else ("Authoritarian" if avg_social > 0.5 else "Centrist")
    if lr == "Center" and al == "Centrist":
        label = "Centrist"
    elif al == "Centrist":
        label = lr
    elif lr == "Center":
        label = al
    else:
        label = f"{al}-{lr}"
    return {"economic": round(avg_econ, 2), "social": round(avg_social, 2), "label": label}


def compute_cumulative_mbti(entries):
    """Average MBTI profiles across all entries."""
    dims = ["E", "I", "S", "N", "T", "F", "J", "P"]
    totals = {d: 0 for d in dims}
    count = 0
    for e in entries:
        mb = e.get("mbti_profile", {})
        if mb and "E" in mb:
            try:
                for d in dims:
                    totals[d] += float(mb.get(d, 50))
                count += 1
            except (ValueError, TypeError):
                pass
    if count == 0:
        return None
    avg = {d: round(totals[d] / count) for d in dims}
    # Determine type from averages
    t = ""
    t += "E" if avg["E"] >= avg["I"] else "I"
    t += "S" if avg["S"] >= avg["N"] else "N"
    t += "T" if avg["T"] >= avg["F"] else "F"
    t += "J" if avg["J"] >= avg["P"] else "P"
    avg["type"] = t
    return avg


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
    qotd = get_daily_quote()
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown(f"""
        <div class="sage-header">
            <div style="display:flex;align-items:flex-start;gap:1.5rem;flex-wrap:wrap;">
                <div style="flex-shrink:0;">
                    <h1>🪞 Inner Mirror</h1>
                    <p>Reflective writing analysis</p>
                </div>
                <div style="flex:1;min-width:200px;padding-top:4px;border-left:2px solid rgba(111,207,151,0.3);padding-left:1rem;">
                    <div style="font-size:0.7rem;color:#6fcf97;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Quote of the Day</div>
                    <div style="font-size:0.82rem;font-style:italic;color:#c0c0c0;line-height:1.4;">"{qotd['text']}"</div>
                    <div style="font-size:0.72rem;color:#888;margin-top:3px;">— {qotd['author']}</div>
                </div>
            </div>
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
                st.markdown(f"<div style='text-align:center;padding-top:8px;font-size:0.85rem;color:#6fcf97;font-weight:500;'>{user['username']}</div>", unsafe_allow_html=True)
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
            st.error("No Groq API key found. Please set GROQ_API_KEY in your Streamlit secrets or environment variables. Get a free key at https://console.groq.com/keys (no credit card required)")
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
                        political_compass=result.get("political_compass"),
                        mbti_profile=result.get("mbti_profile"),
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
            render_emotion_radar(analysis["emotions"], chart_key="home_emotion_radar")

        # Disorders
        st.markdown("**Potential Symptom Patterns**")
        st.markdown('<p class="disclaimer">Not a diagnosis. Patterns that may warrant reflection or professional consultation.</p>', unsafe_allow_html=True)
        render_disorder_chart(analysis["disorders"], chart_key="home_disorder_chart")

        # Symptoms detail
        for d in analysis["disorders"]:
            with st.expander(f"{d['disorder']} — {d['relevance']}% relevance"):
                st.write(d.get("description", ""))
                if d.get("symptoms"):
                    st.write("**Observed symptoms:** " + ", ".join(d["symptoms"]))

        # Political Compass + MBTI
        pc = analysis.get("political_compass")
        mb = analysis.get("mbti_profile")
        if pc or mb:
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                if pc and "economic" in pc:
                    st.markdown("**Political Compass**")
                    render_political_compass(pc, chart_key="home_pc")
            with pcol2:
                if mb and "E" in mb:
                    st.markdown("**MBTI Profile**")
                    render_mbti_radar(mb, chart_key="home_mbti")

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
    qotd = get_daily_quote()
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown(f"""
        <div class="sage-header">
            <div style="display:flex;align-items:flex-start;gap:1.5rem;flex-wrap:wrap;">
                <div style="flex-shrink:0;">
                    <h1>🪞 Inner Mirror</h1>
                    <p>Reflective writing analysis</p>
                </div>
                <div style="flex:1;min-width:200px;padding-top:4px;border-left:2px solid rgba(111,207,151,0.3);padding-left:1rem;">
                    <div style="font-size:0.7rem;color:#6fcf97;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Quote of the Day</div>
                    <div style="font-size:0.82rem;font-style:italic;color:#c0c0c0;line-height:1.4;">"{qotd['text']}"</div>
                    <div style="font-size:0.72rem;color:#888;margin-top:3px;">— {qotd['author']}</div>
                </div>
            </div>
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
            st.markdown(f"<div style='text-align:center;padding-top:8px;font-size:0.85rem;color:#6fcf97;font-weight:500;'>{user['username']}</div>", unsafe_allow_html=True)
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

    # Cumulative Word Map + Emotion Radar side by side (like single entry page)
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        st.markdown(f"**Cumulative Word Map** — {len(entries)} {'analysis' if len(entries)==1 else 'analyses'}")
        render_word_cloud(cum_words, height=220)
    with wcol2:
        st.markdown(f"**Cumulative Emotional Landscape**")
        st.caption(f"Weighted intensity across {len(entries)} {'analysis' if len(entries)==1 else 'analyses'}")
        if cum_emotions:
            render_emotion_radar(cum_emotions, chart_key="hist_cum_emotion_radar")

    st.write("")

    # Recurring emotions
    st.markdown("**Recurring Emotions**")
    st.caption("How often each emotion recurs across your writing")
    # Display as a horizontal row of badges
    cols = st.columns(min(len(top_emotions), 5)) if top_emotions else []
    for i, em in enumerate(top_emotions):
        with cols[i % min(len(top_emotions), 5)]:
            badge_html = f"""
            <div style="display:flex; flex-direction:column; align-items:center; padding:8px 10px; border-radius:12px; background:rgba(45,106,79,0.12); border:1px solid rgba(45,106,79,0.2); margin-bottom:6px; text-align:center;">
                <div style="width:10px;height:10px;border-radius:50%;background:{em['color']};margin-bottom:4px;"></div>
                <span style="font-size:0.78rem;font-weight:600;color:#e0e0e0;">{em['emotion']}</span>
                <span style="font-size:0.7rem;color:#a0a0a0;">{em['recurrence_rate']}%</span>
                <span style="font-size:0.62rem;color:#888;">avg {em['avg_intensity']}%</span>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

    st.write("")

    # Cumulative disorders
    if cum_disorders:
        st.markdown("**Cumulative Symptom Patterns**")
        st.caption("Average relevance across all analyses. Not a diagnosis — patterns that may warrant reflection or professional consultation.")
        render_disorder_chart(cum_disorders, chart_key="hist_cum_disorder_chart")

    st.write("")

    # Political Compass + MBTI Profile side by side
    cum_pc = compute_cumulative_political_compass(entries)
    cum_mbti = compute_cumulative_mbti(entries)
    if cum_pc or cum_mbti:
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("**Political Compass**")
            st.caption("Where your writing's themes and values fall on the political spectrum")
            if cum_pc:
                render_political_compass(cum_pc, chart_key="hist_cum_pc")
            else:
                st.info("Not enough data yet — submit a new analysis to see your political compass.")
        with pcol2:
            st.markdown("**MBTI Profile**")
            st.caption("Cognitive style inferred from your writing patterns")
            if cum_mbti:
                render_mbti_radar(cum_mbti, chart_key="hist_cum_mbti")
            else:
                st.info("Not enough data yet — submit a new analysis to see your MBTI profile.")

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
                <p style="font-size:0.85rem;color:#b0b0b0;line-height:1.5;margin:0;">
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
                render_emotion_radar(entry["emotions"], chart_key=f"entry_{entry['id']}_emotion")

            st.markdown("**Potential Symptom Patterns**")
            st.markdown('<p class="disclaimer">Not a diagnosis. Patterns that may warrant reflection or professional consultation.</p>', unsafe_allow_html=True)
            render_disorder_chart(entry["disorders"], chart_key=f"entry_{entry['id']}_disorder")

            # Political Compass + MBTI for this entry
            epc = entry.get("political_compass", {})
            emb = entry.get("mbti_profile", {})
            if (epc and "economic" in epc) or (emb and "E" in emb):
                epcol1, epcol2 = st.columns(2)
                with epcol1:
                    if epc and "economic" in epc:
                        st.markdown("**Political Compass**")
                        render_political_compass(epc, chart_key=f"entry_{entry['id']}_pc")
                with epcol2:
                    if emb and "E" in emb:
                        st.markdown("**MBTI Profile**")
                        render_mbti_radar(emb, chart_key=f"entry_{entry['id']}_mbti")

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
