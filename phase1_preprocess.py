# phase1_preprocess.py
from langdetect import detect
from deep_translator import GoogleTranslator
from textblob import TextBlob
from mesh_parser import load_mesh_synonyms

# Load MeSH once
mesh_dict = load_mesh_synonyms("desc2025.xml")

# Basic Roman Hindi to English dictionary
SYMPTOM_SYNONYMS = {
    "bukhar": "fever",
    "sardi": "cold",
    "khansi": "cough",
    "sir dard": "headache",
    "pet dard": "stomach pain",
    "chakkar": "dizziness",
    "ghabrahat": "anxiety",
    "kamzori": "weakness",
    "neend nahi aati": "insomnia",
    "seene me jalan": "chest burning"
}

def preprocess_input(user_input):
    # 1. Language Detection
    try:
        lang = detect(user_input)
    except:
        lang = "unknown"

    # 2. Translation if Hindi/Marathi
    if lang in ["hi", "mr"]:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(user_input)
        except:
            translated = user_input
    else:
        translated = user_input

    # 3. Replace Roman Hindi words
    translated_lower = translated.lower()
    for k, v in SYMPTOM_SYNONYMS.items():
        translated_lower = translated_lower.replace(k.lower(), v.lower())

    # 4. Spell correction (skip long strings to avoid distortion)
    if len(translated_lower.split()) <= 12:
        try:
            corrected = str(TextBlob(translated_lower).correct())
        except:
            corrected = translated_lower
    else:
        corrected = translated_lower

    # 5. MeSH match (exact match only for now)
    mesh_match = mesh_dict.get(corrected.lower(), None)

    return {
        "original": user_input,
        "language": lang,
        "translated": translated,
        "corrected": corrected,
        "mesh_term": mesh_match
    }
