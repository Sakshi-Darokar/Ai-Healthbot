# smart_dialog_manager.py

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

def generate_followup_question(
    llm, memory, symptom_profile, last_user_answer=None, asked_questions=None, conversation_history=None
):
    """
    Uses LLM to decide whether more info is needed, and asks smart follow-up questions.
    Returns "DONE" if no more questions are needed.
    """

    if asked_questions is None:
        asked_questions = []
    if last_user_answer is None:
        last_user_answer = ""
    if conversation_history is None:
        conversation_history = ""
    elif isinstance(conversation_history, list):
        conversation_history = "\n".join(conversation_history)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a medical assistant.\n"
            "Here is the full conversation so far (with all previous questions and answers):\n"
            f"{conversation_history}\n"
            "The user's initial message describes their main complaint. "
            "ALL follow-up questions must be directly related to clarifying and understanding this main complaint, "
            "unless the user clearly introduces a new, unrelated issue and confirms it is a separate concern.\n"
            "Never switch topics or ask about unrelated symptoms unless the user confirms a new, unrelated issue.\n"
            "If all important symptom details about the main complaint are collected, return exactly: DONE.\n"
            "Never repeat or rephrase a question from this list: {asked_questions}\n"
            "Never ask about symptoms or details the user has already mentioned or just answered, even if rephrased.\n"
            "If the user's most recent answer already contains the information you would ask for, do NOT ask again.\n"
            "Only ask one question at a time.\n"
            "If you are unsure, err on the side of NOT repeating questions.\n"
            "Do NOT ask about any body system or symptom that is not part of the user's main complaint or their confirmed related symptoms. "
            "For example, if the user has not mentioned digestive issues, do NOT ask about stool, bowel movements, or digestion.\n"
            "If the user's most recent answer is a direct response to your previous question, your next question must logically follow from that answer and stay on the same topic.\n"
            "Do NOT return DONE or ask a catch-all unless you are certain all relevant details about the main complaint have been collected.\n"
            "Do NOT ask about anything already covered in the conversation history.\n"
            "Do not repeat questions already answered. Only ask for missing information."
        )),
        HumanMessage(content=f"Collected symptom profile so far: {symptom_profile}"),
        HumanMessage(content=f"User's most recent answer: {last_user_answer}"),
        HumanMessage(content="What should I ask next?")
    ])
    response = llm.invoke(prompt.format_messages(
        asked_questions=asked_questions
    ))
    return response.content.strip()
