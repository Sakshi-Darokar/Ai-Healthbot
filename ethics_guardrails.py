# ethics_guardrails.py

from langchain_core.language_models import BaseLanguageModel

# Unsafe phrasing detection (patterns we want to avoid)
UNSAFE_KEYWORDS = [
    "you have", "you are suffering from", "this is definitely",
    "will cure", "take this medicine", "diagnose"
]

# Function to detect unsafe language

def detect_unethical_phrases(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in UNSAFE_KEYWORDS)

# Use LLM to safely rephrase any unsafe text

def rewrite_with_guardrails(llm: BaseLanguageModel, response: str) -> str:
    prompt = f"""
    The following response may contain unsafe or overly certain medical language.
    Please rewrite it to be cautious, avoid direct diagnoses or treatment instructions,
    and include gentle disclaimers if appropriate. Keep it friendly, informative, and ethical.

    ---
    ORIGINAL:
    {response}

    REWRITE:
    """
    result = llm.invoke(prompt)
    return result.content.strip()

# Master function

def apply_guardrails_if_needed(llm: BaseLanguageModel, response: str) -> str:
    if detect_unethical_phrases(response):
        return rewrite_with_guardrails(llm, response)
    return response

# Example usage (optional test)
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
    test_text = "You have COVID and should take this medicine."
    print("Original:", test_text)
    print("Guarded:", apply_guardrails_if_needed(llm, test_text))
