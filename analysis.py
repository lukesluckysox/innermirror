"""Text analysis logic: word extraction + Groq LLM call."""

import json
import re
import os
from groq import Groq

STOP_WORDS = {
    # Articles, conjunctions, prepositions
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "into", "over", "after", "before",
    "between", "under", "again", "through", "because", "while", "during",
    "until", "upon", "about", "against", "along", "among", "around",
    # Pronouns
    "its", "my", "your", "his", "her", "our", "their", "me", "him", "them",
    "we", "you", "he", "she", "they", "i", "who", "whom", "whose", "which",
    "what", "that", "this", "these", "those", "myself", "yourself", "itself",
    "himself", "herself", "ourselves", "themselves",
    # Be / have / do
    "is", "it", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did",
    # Modals
    "will", "would", "could", "should", "may", "might", "shall", "can",
    # Adverbs & fillers
    "not", "no", "so", "if", "then", "than", "too", "very", "just", "also",
    "now", "even", "still", "already", "yet", "ever", "never", "always",
    "often", "sometimes", "really", "quite", "well", "much", "many",
    "only", "own", "such", "more", "most", "less", "least",
    # Generic verbs
    "get", "got", "make", "made", "take", "took", "come", "came",
    "go", "went", "gone", "say", "said", "let", "put", "give", "gave",
    "know", "knew", "see", "saw", "want", "think", "like",
    "tell", "told", "ask", "asked", "try", "tried", "use", "used",
    "need", "seem", "keep", "kept", "become", "became", "left",
    "find", "found", "call", "called", "look", "looked", "turn", "turned",
    "start", "started", "show", "showed", "set", "run", "ran",
    # Positional & misc
    "up", "out", "each", "every", "both", "few", "other", "some",
    "same", "there", "here", "once", "when", "where", "how", "all",
    "any", "way", "back", "down", "off", "one", "two", "new", "old",
    "first", "last", "long", "great", "little", "right", "big",
    "thing", "things", "something", "everything", "nothing", "anything",
    "someone", "everyone", "anyone", "people", "time", "times", "day",
    "days", "year", "years", "life", "world",
    # Contractions
    "don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "doesn't", "didn't", "couldn't", "shouldn't", "wouldn't",
    "i'm", "i've", "i'll", "i'd", "it's", "that's", "what's",
    "there's", "here's", "let's", "who's", "he's", "she's",
    "we're", "they're", "you're", "we've", "they've", "you've",
    "we'll", "they'll", "you'll", "we'd", "they'd", "you'd",
    # Common journaling filler
    "feel", "feeling", "felt", "maybe", "though", "although",
    "also", "another", "able", "without", "within", "enough",
    "rather", "away", "around", "since", "however", "whether",
    "almost", "already", "instead", "either", "perhaps",
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
  "regulation_prompts": [
    {{"prompt": "a writing prompt to help regulate the emotion", "target_emotion": "which emotion this addresses", "technique": "the therapeutic approach used"}}
  ],
  "recommended_reading": [
    {{"title": "book or poem title", "author": "author name", "why": "one sentence on why this speaks to the writing's themes"}}
  ],
  "word_colors": {{
    "word1": {{"color": "#hex", "emotion": "emotion category"}},
    "word2": {{"color": "#hex", "emotion": "emotion category"}}
  }},
  "metaphors": [
    {{"image": "the metaphor or symbolic image", "quote": "exact phrase from the text", "interpretation": "what it reveals psychologically"}}
  ],
  "unspoken_emotions": [
    {{"emotion": "the absent emotion", "expected_because": "why it would normally appear given the subject", "interpretation": "what its absence may reveal"}}
  ],
  "letter_to_self": "A short compassionate letter from the writer's wisest future self",
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
- mbti_profile: Estimate the Myers-Briggs cognitive preferences. Each pair must sum to 100. CRITICAL ANTI-BIAS RULES: Do NOT default to INFP. The fact that someone is journaling or writing poetry is NOT evidence of Introversion or Feeling. ALL personality types write reflectively. You must infer type from HOW they write, not THAT they write. Analyze BOTH the person AND the genre:
  STRUCTURAL CUES TO USE:
  - E vs I: E = writing references other people, social situations, external events, action, dialogue. I = solitary imagery, internal monologue, self-referential loops. Short declarative sentences and outward focus suggest E. Long introspective passages suggest I. But writing ABOUT loneliness doesn't mean I — it could be an E in distress.
  - S vs N: S = concrete sensory details (colors, sounds, textures, specific places/times), practical concerns, sequential narrative. N = metaphor, symbolism, abstract concepts, philosophical questioning, pattern-seeking. Count the ratio of concrete nouns to abstract nouns.
  - T vs F: T = cause-and-effect reasoning, analytical tone, detachment, critique, problem-solving language, "because/therefore" logic. F = value judgments, empathy language, personal impact, "it felt like", relational framing. Anger and frustration can be T (analytical grievance) or F (emotional wound) — look at HOW the anger is expressed.
  - J vs P: J = structured writing, clear beginnings/endings, resolution-seeking, decisive language, lists, closure. P = open-ended exploration, stream of consciousness, tangents, ambiguity tolerance, unresolved endings, "maybe/perhaps/I wonder".
  TONE ANALYSIS: Sarcasm and wit suggest T. Raw vulnerability suggests F. Commanding tone suggests E+J. Wandering philosophical tone suggests N+P. Pragmatic advice-giving suggests S+J. Playful irreverence suggests E+P.
  FOR EACH DIMENSION: Before scoring, identify one specific quote or structural feature from the text that supports EACH side of the pair, then decide which has stronger evidence.
  - "type": The 4-letter MBTI type based on whichever letter scores higher in each pair. Produce genuine variety — ESTJ, ISTP, ENFJ, INTP, etc. are all valid outcomes.
- moral_foundations: Based on Jonathan Haidt's Moral Foundations Theory, score how strongly the writing engages each of the six moral foundations (0-100):
  - "care": Sensitivity to suffering, empathy, compassion, protection. High if the text focuses on pain, helping, kindness, vulnerability.
  - "fairness": Justice, equality, reciprocity, rights. High if the text addresses what's deserved, being wronged, balance.
  - "loyalty": Group bonds, tribalism, belonging, betrayal. High if the text addresses trust, in-groups, faithfulness, or being let down.
  - "authority": Respect for hierarchy, tradition, structure, order. High if the text defers to or rebels against rules, institutions, elders.
  - "sanctity": Purity, the sacred, spiritual themes, disgust. High if the text touches on spiritual elevation, contamination, body, or what feels deeply wrong.
  - "liberty": Freedom, autonomy, resistance to oppression or control. High if the text expresses feeling trapped, fighting for independence, or valuing self-determination.
  Calibrate carefully: 70-100 = dominant theme, 40-69 = clearly present, 10-39 = faint signal, 0-9 = absent.
- regulation_prompts: Provide exactly 3 writing prompts designed to help the writer process and regulate the emotions detected. Each prompt should:
  - Target a specific emotion from the analysis (especially the most intense or distressing ones)
  - Use an evidence-based therapeutic technique: expressive writing, cognitive reframing, self-compassion, gratitude reorientation, narrative distancing, or emotion labeling
  - Be warm, inviting, and specific to the themes in THIS text (not generic)
  - Include "target_emotion" (which emotion it addresses) and "technique" (which approach it uses)
- recommended_reading: Suggest exactly 3 specific books, poems, or essays that speak to the emotional themes in the writing. Choose works that would genuinely help someone feeling what this writer is feeling. Include a one-sentence "why" that connects the recommendation to the specific themes detected. Prefer well-known, accessible works.
- letter_to_self: Write a 3-4 sentence compassionate letter as if from the writer's wisest, most loving future self, speaking directly to the pain or themes in the writing. Use "you" to address the writer. Reference specific words or images from the text. This should feel like a warm hug in words — not clinical, not preachy, just deeply understanding.
- word_colors: For EACH of the KEY WORDS listed above, assign an emotion-based color based on HOW that word is used IN THIS SPECIFIC TEXT (not its generic dictionary meaning). The same word can mean different emotions in different contexts:
  - "cold" in a poem about loneliness = sadness (blue), but in a rant = anger (red)
  - "fire" in a love poem = passion (pink), but in a war poem = anger (red)
  - "silence" in a meditation piece = contemplation (teal), but in a breakup poem = sadness (blue)
  Use these emotion categories and colors:
  - sadness = #4A6FA5 (steel blue)
  - anger = #EB5757 (red)
  - fear = #9B51E0 (purple)
  - joy = #F2C94C (gold)
  - love = #E84393 (pink)
  - contemplation = #20B2AA (light teal)
  - shame = #E07C4F (burnt orange)
  - hope = #00B894 (emerald green)
  Every word MUST be assigned a color. No word should be left uncolored. Think carefully about the contextual meaning.
- metaphors: Identify the 2-3 most psychologically significant metaphors, symbolic images, or figurative language in the writing. These are where the subconscious speaks — people rarely choose metaphors consciously. For each:
  - "image": Name the metaphor or symbol (e.g. "drowning", "locked room", "armor/walls")
  - "quote": The exact phrase from the text where this appears
  - "interpretation": A 1-2 sentence psychological interpretation of what this metaphor reveals about the writer's inner state. Be specific and insightful, not generic. Connect it to the emotional patterns you've detected.
  If the writing is very literal with no figurative language, identify the most emotionally loaded concrete images instead and interpret those.
- unspoken_emotions: Identify 1-2 emotions that are conspicuously ABSENT given the subject matter. What the writer does NOT say is often more revealing than what they do say. For each:
  - "emotion": The missing emotion (e.g. "anger", "joy", "fear", "grief")
  - "expected_because": Why this emotion would normally be present given what the person is writing about (1 sentence)
  - "interpretation": What its absence may reveal — suppression, dissociation, avoidance, numbness, or a deliberate coping mechanism (1-2 sentences). Be compassionate, not accusatory.
  If ALL expected emotions are present, identify the one that seems underrepresented relative to how central it is to the subject.
- summary: Write a warm, empathetic summary that references the most frequent words and what they reveal emotionally.

IMPORTANT: Return ONLY valid JSON. No markdown, no code fences, no explanation."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_completion_tokens=7000,
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
