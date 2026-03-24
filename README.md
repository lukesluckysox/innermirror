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

# Set your Gemini API key (free — no credit card required)
export GEMINI_API_KEY="your-gemini-api-key-here"

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Get a Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key — no credit card or billing required

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
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```
5. **Deploy** — Click deploy and your app will be live in minutes

## Project Structure

```
├── app.py                  # Main Streamlit app (all pages)
├── analysis.py             # Text analysis: word extraction + Google Gemini LLM call
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
3. Google Gemini analyzes the text for emotions, diagnostic patterns, and quotes
4. Results are displayed as interactive charts and saved to your history
5. The history page shows cumulative trends across all your entries

## Tech Stack

- **Streamlit** — UI framework
- **Google Gemini** — AI text analysis (free tier, no credit card required)
- **Plotly** — Interactive charts (radar, bar)
- **WordCloud** — Word frequency visualization
- **SQLite** — Local database for persistence

## License

MIT
