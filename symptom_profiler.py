# symptom_profiler.py
import ast
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate


def extract_symptom_info(user_text, profile, llm):
    """
    Uses the LLM to extract structured symptom information from user input and
    merges it into the existing profile dictionary.

    Args:
        user_text (str): The user's raw input (already preprocessed).
        profile (dict): Current symptom profile.
        llm (BaseLanguageModel): LangChain-compatible LLM instance.

    Returns:
        dict: Updated symptom profile.
    """

    schema_prompt = """
You are a medical assistant extracting structured data from patient symptom descriptions.
From the text, extract the following key details if available:

- symptom: the main complaint (e.g., fever, sore throat, headache)
- location: where it hurts (e.g., chest, throat, stomach)
- duration: how long it has been happening (e.g., 3 days, since yesterday)
- severity: mild / moderate / severe
- frequency: constant / intermittent / comes and goes
- triggers: what makes it worse or better (e.g., at night, after eating)

Output only a Python dictionary. If a field is missing, omit it.

Example output:
{"symptom": "fever", "location": "throat", "duration": "2 days", "severity": "moderate"}
"""

    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=schema_prompt.strip()),
            HumanMessage(content=f"User said: \"{user_text}\"")
        ])

        response = llm.invoke(prompt.format_messages())
        content = response.content
        if content is None:
            return profile
        if isinstance(content, list):
            content = content[0]  # Take the first item if it's a list
        extracted = ast.literal_eval(str(content).strip())

        if isinstance(extracted, dict):
            for key, value in extracted.items():
                profile[key] = value.strip() if isinstance(value, str) else value

    except Exception as e:
        print("❌ LLM profile extraction failed:", e)

    return profile


def is_profile_complete(profile):
    """
    Checks if minimum required keys exist in the profile to make a prediction.
    """
    required = ["symptom", "duration", "severity", "location"]
    return all(k in profile and profile[k] for k in required)
