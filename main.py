# main.py
import streamlit as st
from phase1_preprocess import preprocess_input
from pubmed_utils import get_evidence_links
from symptom_classifier import is_valid_symptom
from langchain_memory import get_memory_llm
from smart_dialog_manager import generate_followup_question
from symptom_profiler import extract_symptom_info
from disease_predictor import predict_disease_from_profile
from ethics_guardrails import apply_guardrails_if_needed
import openai
import re
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

def extract_condition_name(reply):
    import re
    match = re.search(r"Possible Condition:\s*([^\n\r\*]+)", reply)
    if match:
        name = match.group(1).strip()
        name = re.split(r"Reason:|Care Tips:|\*\*", name)[0].strip()
        return name
    return None

def format_bot_reply(content):
    # All headers in red, each on new line
    content = re.sub(r"\*\*Possible Condition:\*\*", "<br><b style='color:#d84315;'>Possible Condition:</b>", content)
    content = re.sub(r"\*\*Reason:\*\*", "<br><b style='color:#d84315;'>Reason:</b>", content)
    content = re.sub(r"\*\*Care Tips:\*\*", "<br><b style='color:#d84315;'>Care Tips:</b>", content)
    content = re.sub(r"\*\*Disclaimer:\*\*", "<br><b style='color:#d84315;'>Disclaimer:</b>", content)
    content = re.sub(r"\*\*PubMed Sources:\*\*", "<br><b style='color:#d84315;'>PubMed Sources:</b>", content)
    content = re.sub(r"•", "<br>•", content)
    content = re.sub(r"(\d+\.)", r"<br>\1", content)
    return content

def check_emergency_symptoms(text):
    emergency_keywords = [
        "chest pain", "severe shortness of breath", "pressure in chest", "pain spreading to arm", "pain spreading to jaw",
        "sudden weakness", "loss of consciousness", "severe headache", "confusion", "heavy pressure"
    ]
    text = text.lower()
    return any(kw in text for kw in emergency_keywords)


def filter_relevant_links(links, condition_name):
    cond = condition_name.lower().replace(" ", "")
    return [url for url in links if cond in url.lower().replace(" ", "")]

def llm_assess_urgency_tone_risk(llm, conversation):
    prompt = f"""
You are a medical triage assistant. Given the following conversation, classify the user's current situation as one of:
- "emergency" (needs immediate medical attention)
- "urgent" (see a doctor soon)
- "routine" (can wait for self-care or regular doctor visit)

Also, analyze:
- User's tone (anxious, calm, confused, etc.)
- Any risk factors mentioned (age, chronic illness, pregnancy, etc.)
- Whether escalation to telehealth is recommended (yes/no)

Conversation:
{conversation}

Respond in this JSON format:
{{
  "urgency": "...",
  "tone": "...",
  "risk_factors": "...",
  "telehealth_recommended": "yes/no"
}}
"""
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        response = response.content
    import json
    import re
    # Extract only the first {...} block
    match = re.search(r"\{[\s\S]*?\}", response)
    if match:
        try:
            data = json.loads(match.group(0))
        except Exception as e:
            print("LLM JSON parse error:", e)
            data = {"urgency": "routine", "tone": "", "risk_factors": "", "telehealth_recommended": "no"}
    else:
        data = {"urgency": "routine", "tone": "", "risk_factors": "", "telehealth_recommended": "no"}
    return data

def split_conditions(condition_name):
    # Split on commas, "or", "and", etc.
    import re
    # Remove "or"/"and" and split
    parts = re.split(r",|\bor\b|\band\b", condition_name, flags=re.IGNORECASE)
    # Clean up whitespace and filter empty
    return [p.strip() for p in parts if p.strip()]

# --- Custom CSS for Modern Look ---
st.markdown("""
    <style>
    body { background-color: #f7f9fa !important; }
    .chat-container { display: flex; flex-direction: column; gap: 1.2rem; margin-bottom: 2rem; max-width: 700px; margin-left: auto; margin-right: auto; padding-bottom: 120px; }
    .chat-bubble-user { background: #e3f2fd; color: #222; padding: 14px 22px; border-radius: 22px 22px 6px 22px; align-self: flex-end; max-width: 75%; font-size: 1.13em; border: 1.5px solid #90caf9; margin-bottom: 8px; box-shadow: 0 2px 8px #e3f2fd55; font-weight: 500; text-align: right; }
    .chat-bubble-bot { background: #f3e5f5; color: #222; padding: 16px 24px; border-radius: 22px; align-self: center; max-width: 85%; font-size: 1.13em; border: 1.5px solid #ce93d8; margin-bottom: 8px; box-shadow: 0 2px 12px #ce93d855; font-weight: 500; text-align: left; }
    .fixed-input-box { position: fixed; bottom: 0; left: 0; width: 100vw; background: #f7f9fa; z-index: 100; padding: 1.2rem 0 1.2rem 0; box-shadow: 0 -2px 12px #e0e0e0; }
    .stButton>button { background: linear-gradient(90deg, #80deea 0%, #ba68c8 100%); color: white; border-radius: 10px; font-weight: bold; font-size: 1.13em; border: none; padding: 0.6em 2.2em; margin-top: 0.5em; box-shadow: 0 2px 8px #ba68c855; }
    .stSpinner { color: #ba68c8 !important; }
    .block-container { padding-bottom: 30px !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

# --- App Title and Logo ---
st.markdown("<h1 style='text-align:center; color:#6a1b9a;'>🤖 AI HealthBot - Symptom Checker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#616161;'>Your private AI-powered health assistant.<br><b>Note:</b> This is not medical advice.</p>", unsafe_allow_html=True)

# --- Session State ---
if "symptom_profile" not in st.session_state:
    st.session_state.symptom_profile = {}
if "main_complaint_profile" not in st.session_state:
    st.session_state.main_complaint_profile = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "followup_count" not in st.session_state:
    st.session_state.followup_count = 0
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "main_complaint" not in st.session_state:
    st.session_state.main_complaint = ""
if "awaiting_relation_clarification" not in st.session_state:
    st.session_state.awaiting_relation_clarification = False
if "awaiting_followup_answer" not in st.session_state:
    st.session_state.awaiting_followup_answer = False
if "show_spinner" not in st.session_state:
    st.session_state.show_spinner = False

llm, memory = get_memory_llm()

# --- Chat Display ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        content = format_bot_reply(msg["content"])
        st.markdown(f'<div class="chat-bubble-bot">🤖 {content}</div>', unsafe_allow_html=True)
if st.session_state.show_spinner:
    st.markdown('<div class="chat-bubble-bot"><span class="stSpinner">🤖 HealthBot is thinking...</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Fixed Input Box at Bottom ---
st.markdown('<div class="fixed-input-box">', unsafe_allow_html=True)
with st.form(key="symptom_form", clear_on_submit=True):
    user_input = st.text_area(
        "Describe your symptoms:",
        key="user_input",
        height=68,
        max_chars=500,
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Send")
st.markdown('</div>', unsafe_allow_html=True)

# --- Handle User Input ---
if submitted and user_input:
    if not is_valid_symptom(user_input, llm=llm):

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"I see you wrote: <i>{user_input}</i><br>"
                "I'm here to help with health-related questions. "
                "Could you please describe a medical symptom or concern?"
            )
        })
        st.session_state.last_user_input = ""
        st.session_state.show_spinner = False
        st.rerun()
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.last_user_input = user_input
    st.session_state.show_spinner = True
    st.rerun()

# --- Processing Spinner and Bot Logic ---
if st.session_state.last_user_input:
    user_input = st.session_state.last_user_input.strip()
    result = preprocess_input(user_input)
    clean_input = result["corrected"]

    st.session_state.symptom_profile = extract_symptom_info(
        clean_input, st.session_state.symptom_profile, llm
    )

    if not st.session_state.main_complaint:
        st.session_state.main_complaint = clean_input

    # Build conversation context for LLM triage
    conversation = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Bot"
        conversation += f"{role}: {msg['content']}\n"
    conversation += f"User: {clean_input}\n"

    # LLM-based triage, tone, and risk factor analysis
    triage = llm_assess_urgency_tone_risk(llm, conversation)
    urgency = triage.get("urgency", "routine")
    tone = triage.get("tone", "")
    risk_factors = triage.get("risk_factors", "")
    telehealth_recommended = triage.get("telehealth_recommended", "no")

    # 1. Emergency check (LLM only)
    if urgency == "emergency":
        urgent_msg = (
            "🚨 <span style='color:#d84315; font-weight:bold;'>Your symptoms may indicate a life-threatening emergency. "
            "Stop what you're doing and seek emergency care immediately. Do not delay.</span>"
            "<br><b>If chest pain lasts more than a few minutes, feels like pressure, spreads to the arm/jaw, or occurs with breathlessness—call emergency services.</b>"
            "<br><br><b style='color:#d84315;'>Disclaimer:</b> This is not medical advice. Please consult a licensed doctor."
        )
        if telehealth_recommended == "yes":
            urgent_msg += "<br><b>For urgent online help, consider contacting a telehealth provider immediately.</b>"
        st.session_state.urgent_msg = urgent_msg
    else:
        st.session_state.urgent_msg = None

    # 2. If not emergency, check if profile is complete
    from symptom_profiler import is_profile_complete
    if not is_profile_complete(st.session_state.symptom_profile):
        conversation_history = "\n".join([f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}" for m in st.session_state.messages])
        followup = generate_followup_question(
            llm, memory, st.session_state.symptom_profile,
            last_user_answer=clean_input,
            asked_questions=st.session_state.asked_questions,
            conversation_history=conversation_history
        )
        if followup.strip().lower() == "done":
            pass
        else:
            st.session_state.messages.append({"role": "assistant", "content": followup.strip()})
            st.session_state.asked_questions.append(followup.strip())
            st.session_state.followup_count += 1
            st.session_state.last_user_input = ""
            st.session_state.awaiting_followup_answer = True
            st.session_state.show_spinner = False
            st.rerun()

    # 3. If profile is complete or LLM says "DONE", predict and show structured answer
    reply = predict_disease_from_profile(
        llm, st.session_state.symptom_profile, clean_input
    )
    condition_name = extract_condition_name(reply)
    if not condition_name:
        condition_name = st.session_state.main_complaint

    condition_list = split_conditions(condition_name)[:4]  # Only top 4 conditions

    def fetch_links(cond):
        return get_evidence_links(cond, clean_input)

    all_links = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_links, condition_list))
        for links in results:
            all_links.extend(links)

    # Remove duplicates, keep order
    seen = set()
    unique_links = []
    for link in all_links:
        if link not in seen:
            unique_links.append(link)
            seen.add(link)

    # Remove LLM-generated PubMed and Disclaimer sections
    reply = re.sub(r"\*\*?PubMed Sources:?[\s\S]*?(?=(\*\*|$))", "", reply, flags=re.IGNORECASE)
    reply = re.sub(r"\*\*?Disclaimer:?[\s\S]*?(?=(\*\*|$))", "", reply, flags=re.IGNORECASE)
    reply = re.sub(r"Disclaimer: This is not medical advice\. Please consult a licensed doctor\.", "", reply, flags=re.IGNORECASE)

    # Append your own PubMed links
    if unique_links:
        reply += "\n\n**PubMed Sources:**\n" + "\n".join(
            f"{i+1}. {url}" for i, url in enumerate(unique_links))
    else:
        reply += "\n\n**PubMed Sources:**\nNone found."

    reply += "\n\n**Disclaimer:** This is not medical advice. Please consult a licensed doctor."

    # if condition_name:
    #     medline_link = get_medlineplus_link(condition_name)
    #     reply += f"\n\n🌐 <b>Learn more (external link, may not be exact):</b> <a href='{medline_link}' target='_blank'>{medline_link}</a>"

    if telehealth_recommended == "yes":
        reply += "\n\n<b>For further help, you may wish to contact a telehealth provider.</b>"

    reply = apply_guardrails_if_needed(llm, reply)

    # Prepend emergency message if needed (so user sees both the warning and the structured output)
    if getattr(st.session_state, "urgent_msg", None):
        reply = st.session_state.urgent_msg + "<br><br>" + reply

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.followup_count = 0
    st.session_state.asked_questions = []
    st.session_state.last_user_input = ""
    st.session_state.show_spinner = False
    st.rerun()