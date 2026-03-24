# 🪞 Inner Mirror — Reflective Writing Analysis

A wellness-focused Streamlit app that analyzes poetry, journal entries, and personal writing to reveal emotional patterns, potential diagnostic themes, and curated quotes for reflection.

## Features

- **Word Cloud** — Top 20 most meaningful words extracted from your writing (common stop words filtered)
- **Emotional Landscape** — Radar chart showing detected emotions with intensity scores
- **Symptom Patterns** — Bar chart of potential diagnostic themes grounded in the text's language
- **Quotes for Reflection** — 5 curated quotes that address the emotions detected in your writing
- **History & Tracking** — Cumulative word map, weighted emotion radar, recurrence rates, and disorder trends across all entries
- **Date Filtering** — Filter history by date written or search by keywords
- **User Authentication** — Login to save and track analyses over time
- **Delete entries** — Remove individual analyses from your history

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Pre-seeded Account
A default test account is created on first run:
- **Username:** tester
- **Password:** tester123

## Deploy on Streamlit Community Cloud

1. **Push to GitHub** — Create a repo and push all these files
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Connect your GitHub repo** and select `app.py` as the main file
4. **Add your secret** — In the app settings, under "Secrets", add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-actual-key-here"
   ```
5. **Deploy** — Click deploy and your app will be live in minutes

## Project Structure

```
├── app.py                  # Main Streamlit app (all pages)
├── analysis.py             # Text analysis: word extraction + Anthropic LLM call
├── db.py                   # SQLite database layer (users + analyses)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   ├── config.toml         # Streamlit theme (sage green)
│   └── secrets.toml.example # Template for API key
├── .gitignore
└── README.md
```

## How It Works

1. Paste your writing (poetry, journal entry, etc.)
2. The app extracts the top 20 meaningful words by frequency
3. Claude (Anthropic) analyzes the text for emotions, diagnostic patterns, and quotes
4. Results are displayed as interactive charts and saved to your history
5. The history page shows cumulative trends across all your entries

## Tech Stack

- **Streamlit** — UI framework
- **Anthropic Claude** — AI text analysis
- **Plotly** — Interactive charts (radar, bar)
- **WordCloud** — Word frequency visualization
- **SQLite** — Local database for persistence

## License

MIT
