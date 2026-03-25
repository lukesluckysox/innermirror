"""Text analysis logic: word extraction + Groq LLM call."""

import json
import re
import os
from groq import Groq

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "into", "over", "after", "before",
    "between", "under", "again", "through", "because", "while", "during",
    "until", "upon", "about", "against", "along", "among", "around",
    "its", "my", "your", "his", "her", "our", "their", "me", "him", "them",
    "we", "you", "he", "she", "they", "i", "who", "whom", "whose", "which",
    "what", "that", "this", "these", "those", "myself", "yourself", "itself",
    "is", "it", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "so", "if", "then", "than", "too", "very", "just", "also",
    "now", "even", "still", "already", "yet", "ever", "never", "always",
    "often", "sometimes", "really", "quite", "well", "much", "many",
    "only", "own", "such", "more", "most", "less", "least",
    "get", "got", "make", "made", "take", "took", "come", "came",
    "go", "went", "gone", "say", "said", "let", "put", "give", "gave",
    "know", "knew", "see", "saw", "want", "think", "like",
    "up", "out", "each", "every", "both", "few", "other", "some",
    "same", "there", "here", "once", "when", "where", "how", "all",
    "any", "way", "back", "down", "off", "one", "two",
    "don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "doesn't", "didn't", "couldn't", "shouldn't", "wouldn't",
}


def extract_word_frequencies(text: str, top_n: int = 20):
    cleaned = re.sub(r"[^a-z'\s-]", " ", text.lower())
    words = [w for w in cleaned.split() if len(w) > 2 and w not in STOP_WORDS]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"word": w, "count": c} for w, c in sorted_words]


def analyze_text(text: str, api_key: str):
    """Call Groq LLM to analyze the writing. Returns parsed dict."""
    client = Groq(api_key=api_key)
    word_frequencies = extract_word_frequencies(text)
    top_words_str = ", ".join(w["word"] for w in word_frequencies[:30])

    prompt = f"""You are a literary therapist analyzing writing for emotional content. Analyze the following text and provide your response as valid JSON with NO markdown formatting, NO code blocks, just pure JSON.

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

KEY WORDS BY FREQUENCY (most repeated words in this writing — use these to ground your analysis):
{top_words_str}

Respond with this exact JSON structure:
{{
  "emotions": [
    {{"emotion": "name", "intensity": 0-100, "color": "hex color that represents this emotion"}}
  ],
  "disorders": [
    {{"disorder": "name", "relevance": 0-100, "symptoms": ["symptom1", "symptom2"], "description": "brief clinical description referencing specific words/phrases from the text that indicate this pattern"}}
  ],
  "quotes": [
    {{"text": "the quote", "author": "who said it"}}
  ],
  "political_compass": {{
    "economic": -5.0 to 5.0,
    "social": -5.0 to 5.0,
    "label": "brief label like 'Left-Libertarian' or 'Centrist'"
  }},
  "mbti_profile": {{
    "E": 0-100,
    "I": 0-100,
    "S": 0-100,
    "N": 0-100,
    "T": 0-100,
    "F": 0-100,
    "J": 0-100,
    "P": 0-100,
    "type": "e.g. INFP"
  }},
  "moral_foundations": {{
    "care": 0-100,
    "fairness": 0-100,
    "loyalty": 0-100,
    "authority": 0-100,
    "sanctity": 0-100,
    "liberty": 0-100
  }},
  "summary": "A compassionate 2-3 sentence summary of the emotional state reflected in this writing, referencing key words from the text"
}}

RULES:
- emotions: Identify 5-8 emotions present in the writing. Intensity should be calibrated carefully:
  - 80-100 = dominant, unmistakable emotion (the text is overwhelmingly about this)
  - 50-79 = clearly present, significant theme
  - 20-49 = detectable undertone, secondary theme
  - 0-19 = faint trace
  The intensities must add context: a joyful poem should NOT have high sadness. Make sure the relative ordering makes sense. Include a hex color for each (e.g. sadness=#4A6FA5, joy=#F2C94C, anger=#EB5757, fear=#9B51E0, love=#E84393, hope=#00B894, anxiety=#FDCB6E, loneliness=#636E72).
- disorders: Identify 3-5 potential psychological conditions suggested by the emotional patterns and the KEY WORDS above. Relevance 0-100 should be calibrated:
  - 70-100 = strong textual evidence with multiple indicators
  - 40-69 = moderate evidence, some indicators present
  - 10-39 = speculative, faint signal
  Use ONLY these standardized disorder names to enable consistent tracking across multiple analyses: "Major Depression", "Generalized Anxiety", "Complicated Grief", "PTSD", "Adjustment Disorder", "Existential Crisis", "Attachment Anxiety", "Dissociative Tendencies", "Social Isolation", "Emotional Dysregulation", "Anticipatory Grief", "Burnout", "Identity Disturbance". Pick 3-5 that fit best. Include 2-4 specific symptoms for each. In the description, reference specific words from the KEY WORDS list that support this pattern.
- quotes: Provide exactly 5 meaningful quotes from poets, philosophers, psychologists, or authors that address the emotional themes. Choose quotes that relate to the key words and themes.
- political_compass: Based on the VALUES, WORLDVIEW, and THEMES expressed in the writing (not the emotions), estimate where the author falls on the standard political compass:
  - "economic": -5.0 (far left, collectivist, anti-capitalist) to +5.0 (far right, free market, individualist). 0 = center.
  - "social": -5.0 (libertarian, anti-authority, personal freedom) to +5.0 (authoritarian, tradition, order, hierarchy). 0 = center.
  - "label": A concise label like "Left-Libertarian", "Right-Authoritarian", "Centrist", "Libertarian-Left", etc.
  Look for clues in the text: themes of freedom vs order, individual vs collective, critique of institutions, authority, tradition, rebellion, equality, hierarchy, capitalism, community.
- mbti_profile: Based on the WRITING STYLE, THEMES, and EMOTIONAL PATTERNS, estimate the Myers-Briggs cognitive preferences. Each pair must sum to 100:
  - E (Extraversion) + I (Introversion) = 100. E = focus on outer world/action/people. I = inner world/reflection/solitude.
  - S (Sensing) + N (Intuition) = 100. S = concrete/practical/detail. N = abstract/metaphorical/big-picture.
  - T (Thinking) + F (Feeling) = 100. T = logic/analysis/detachment. F = values/empathy/personal.
  - J (Judging) + P (Perceiving) = 100. J = structure/closure/planning. P = openness/flexibility/exploration.
  - "type": The 4-letter MBTI type (e.g. "INFP", "ENTJ") based on whichever letter scores higher in each pair.
- moral_foundations: Based on Jonathan Haidt's Moral Foundations Theory, score how strongly the writing engages each of the six moral foundations (0-100):
  - "care": Sensitivity to suffering, empathy, compassion, protection. High if the text focuses on pain, helping, kindness, vulnerability.
  - "fairness": Justice, equality, reciprocity, rights. High if the text addresses what's deserved, being wronged, balance.
  - "loyalty": Group bonds, tribalism, belonging, betrayal. High if the text addresses trust, in-groups, faithfulness, or being let down.
  - "authority": Respect for hierarchy, tradition, structure, order. High if the text defers to or rebels against rules, institutions, elders.
  - "sanctity": Purity, the sacred, spiritual themes, disgust. High if the text touches on spiritual elevation, contamination, body, or what feels deeply wrong.
  - "liberty": Freedom, autonomy, resistance to oppression or control. High if the text expresses feeling trapped, fighting for independence, or valuing self-determination.
  Calibrate carefully: 70-100 = dominant theme, 40-69 = clearly present, 10-39 = faint signal, 0-9 = absent.
- summary: Write a warm, empathetic summary that references the most frequent words and what they reveal emotionally.

IMPORTANT: Return ONLY valid JSON. No markdown, no code fences, no explanation."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_completion_tokens=4096,
    )

    response_text = response.choices[0].message.content if response.choices else ""

    # Strip markdown code fences if present
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (e.g. ```json or ```)
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        # Remove closing fence
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        analysis = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            analysis = json.loads(match.group(0))
        else:
            raise ValueError("Failed to parse LLM response as JSON")

    analysis["wordFrequencies"] = word_frequencies
    return analysis
