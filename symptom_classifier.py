# symptom_classifier.py
def is_probable_symptom(text):
    keywords = [
        "pain", "fever", "vomit", "nausea", "cough", "rash", "bleeding",
        "infection", "ache", "itch", "swelling", "dizzy", "burning",
        "headache", "chills", "diarrhea", "fatigue", "sore", "cramp", "throat"
    ]
    return any(kw in text.lower() for kw in keywords)

def is_symptom_input_llm(text, client):
    prompt = f"""
Is the following input describing a medical symptom or health condition? 
Reply with only YES or NO.

Input: "{text}"
"""
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return "yes" in response.choices[0].message.content.lower()

def is_valid_symptom(text, llm=None, openai_client=None):
    """
    Returns True if input is a symptom/health concern, using LLM if available.
    """
    prompt = f"""Is the following input describing a medical symptom or health condition? Reply with only YES or NO.\nInput: "{text}" """
    if llm:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else response
        return "yes" in content.lower()
    elif openai_client:
        response = openai_client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=[{"role": "user", "content": prompt}]
        )
        return "yes" in response.choices[0].message.content.lower()
    else:
        # Fallback: basic keyword check (not recommended for production)
        keywords = [
            "pain", "fever", "vomit", "nausea", "cough", "rash", "bleeding",
            "infection", "ache", "itch", "swelling", "dizzy", "burning",
            "headache", "chills", "diarrhea", "fatigue", "sore", "cramp", "throat"
        ]
        return any(kw in text.lower() for kw in keywords)
