# disease_predictor.py

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

def predict_disease_from_profile(llm, profile, original_input):
    """
    Uses an LLM to predict possible conditions, reasoning, and care tips
    based on structured symptom profile and original input.
    - If symptoms suggest a medical emergency, lead with an urgent warning and provide only immediate, condition-specific risk-reduction actions (not general home care).
    - Otherwise, provide 2-4 safe, specific care tips for the most likely condition.
    - Always list possible conditions, not just one.
    - Always include a disclaimer.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a responsible AI Health Assistant.\n"
            "You are given the user's symptom profile and their description.\n"
            "If the symptoms could indicate a medical emergency (such as chest pain, severe shortness of breath, signs of stroke, etc.):\n"
            "- Start your reply with a clear, urgent warning: "
            "🚨 This may be a medical emergency. Stop what you're doing and seek emergency care immediately.\n"
            "- List possible conditions (not just one).\n"
            "- Provide only immediate, condition-specific risk-reduction actions or first aid (e.g., for heart attack: 'chew aspirin if not allergic', for asthma: 'use inhaler if prescribed', for stroke: 'note the time symptoms started').\n"
            "- Do NOT provide general home care tips or suggest waiting.\n"
            "- Bold any critical red flags.\n"
            "If the symptoms are not emergent:\n"
            "- List possible conditions (not just one).\n"
            "- Provide 2-4 safe, specific care tips that are tailored to the most likely condition. Do NOT give generic advice (such as 'rest', 'drink water', 'see a doctor') unless it is uniquely important for this condition. Do NOT suggest prescription drugs or invasive procedures.\n"
            "- Each care tip must be directly relevant to the predicted condition and not general wellness advice.\n"
            "Always include: 'This is not medical advice. Please consult a licensed doctor.'\n"
            "Always include PubMed links if available.\n"
            "Respond in the following format, using these exact headers (each header bold, each section starts on a new line, do not omit or rename any section, even if you have to write 'None found' or 'Not applicable'). DO NOT use any other headers or section names. DO NOT change the header wording or order:\n\n"
            "**Possible Condition:** <List possible conditions>\n"
            "**Reason:** <Explain clearly and specifically why these conditions are likely, based on the user's symptoms>\n"
            "**Care Tips:**\n"
            "• Tip 1\n"
            "• Tip 2\n"
            "• Tip 3 (if appropriate)\n"
            "**PubMed Sources:**\n"
            "<link1 or 'None found'>\n"
            "<link2 or 'None found'>\n"
            
        )),
        HumanMessage(content=f"Symptom profile: {profile}"),
        HumanMessage(content=f"User input: {original_input}")
    ])

    response = llm.invoke(prompt.format_messages())
    return response.content.strip()
