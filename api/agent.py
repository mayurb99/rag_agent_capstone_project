import os
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from tools import book_appointment, get_appointments, cancel_appointment, get_doctors
from logger import log

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
MODEL = "allenai/Olmo-3-7B-Instruct:publicai"

PATIENT_ID = 1


# ─────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────

def call_llm(messages: list) -> str:
    log({"step": "llm_call_start", "prompt": messages[-1]["content"][:80]})
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
    )
    log({"step": "llm_call_end"})
    return response.choices[0].message.content.strip()


def detect_intent(user_input: str) -> str:
    log({"step": "detect_intent_start"})

    prompt = f"""You are an intent classifier for a medical appointment assistant.

Classify the user message into EXACTLY ONE of these labels:
- book_appointment
- view_appointments
- cancel_appointment
- faq
- unknown

Return ONLY the label.

User message: "{user_input}"
Label:"""

    result = call_llm([{"role": "user", "content": prompt}])
    result = result.strip().lower().split()[0]

    log({"step": "detect_intent_end", "intent": result})

    valid = {"book_appointment", "view_appointments", "cancel_appointment", "faq", "unknown"}
    return result if result in valid else "unknown"

from datetime import datetime
def extract_booking_details(user_input: str, existing: dict) -> dict:
    log({"step": "extract_details_start"})
    today = datetime.now().strftime("%Y-%m-%d") 

    prompt = f"""Extract booking details from the user message.
Return ONLY valid JSON:
{{
  "time": "HH:MM" or null,
  "date": "YYYY-MM-DD" or null,
  "symptom": "short description" or null
}}

Today's date is {today}.


User message: "{user_input}"
JSON:"""

    try:
        raw = call_llm([{"role": "user", "content": prompt}])
        start = raw.find("{")
        end = raw.rfind("}") + 1
        extracted = json.loads(raw[start:end])

        for key in ["time", "date", "symptom"]:
            if existing.get(key) is None and extracted.get(key) is not None:
                existing[key] = extracted[key]

        log({"step": "extract_details_end", "data": existing})
        return existing

    except Exception as e:
        log({"step": "extract_details_error", "error": str(e)})
        return existing


def map_symptom_to_specialty(symptom: str) -> str:
    log({"step": "map_specialty_start", "symptom": symptom})

    prompt = f"""Map this symptom to a medical specialty.

Symptom: "{symptom}"

Return ONLY one from:
General, Cardiology, Gastroenterology, Orthopedics, Neurology, Dermatology, Pediatrics
"""

    result = call_llm([{"role": "user", "content": prompt}]).strip()

    log({"step": "map_specialty_end", "specialty": result})
    return result


def extract_appointment_id(user_input: str) -> int | None:
    log({"step": "extract_appt_id_start"})

    prompt = f"""Extract appointment ID number.
Return only number or null.

Message: "{user_input}"
ID:"""

    try:
        raw = call_llm([{"role": "user", "content": prompt}]).strip()
        log({"step": "extract_appt_id_end", "value": raw})
        return int(raw)
    except Exception:
        log({"step": "extract_appt_id_failed"})
        return None


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

def agent_loop(user_input: str, session_state: dict, rag_answer_fn):
    log({"step": "agent_start", "input": user_input})

    # Pending flow
    if session_state.get("pending_booking"):
        log({"step": "pending_flow"})
        pending = session_state["pending_booking"]
        awaiting = pending.get("awaiting")

        if awaiting == "reason":
            pending["symptom"] = user_input
            pending["awaiting"] = None
            return _complete_booking_if_ready(pending, session_state)

        elif awaiting == "slot_choice":
            pending = extract_booking_details(user_input, pending)
            pending["awaiting"] = None
            return _complete_booking_if_ready(pending, session_state)

        elif awaiting == "cancel_id":
            appt_id = extract_appointment_id(user_input)
            session_state.pop("pending_booking", None)

            if appt_id:
                log({"step": "cancel_execute", "id": appt_id})
                result = cancel_appointment(appt_id, PATIENT_ID)
                return result, session_state
            else:
                return "Invalid ID. Please reply with number.", session_state

    # Intent
    log({"step": "before_intent"})
    intent = detect_intent(user_input)
    log({"step": "after_intent", "intent": intent})

    # FAQ
    if intent == "faq":
        log({"step": "faq_branch"})
        return rag_answer_fn(user_input), session_state

    # View
    elif intent == "view_appointments":
        log({"step": "view_branch"})
        return get_appointments(PATIENT_ID), session_state

    # Cancel
    elif intent == "cancel_appointment":
        log({"step": "cancel_branch"})
        appts = get_appointments(PATIENT_ID, raw=True)

        if not appts:
            return "No appointments.", session_state

        session_state["pending_booking"] = {"awaiting": "cancel_id"}
        return "Reply with appointment ID to cancel.", session_state

    # Book
    elif intent == "book_appointment":
        log({"step": "book_branch"})

        pending = extract_booking_details(user_input, {})
        pending.setdefault("time", None)
        pending.setdefault("date", None)
        pending.setdefault("symptom", None)

        if not pending["symptom"]:
            session_state["pending_booking"] = {**pending, "awaiting": "reason"}
            return "What is the reason for visit?", session_state

        return _complete_booking_if_ready(pending, session_state)

    # Fallback
    else:
        log({"step": "fallback_llm"})
        return call_llm([
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_input}
        ]), session_state


def _complete_booking_if_ready(pending: dict, session_state: dict):
    log({"step": "complete_booking_start", "pending": pending})

    if not pending.get("symptom"):
        session_state["pending_booking"] = {**pending, "awaiting": "reason"}
        return "Provide reason.", session_state

    specialty = map_symptom_to_specialty(pending["symptom"])
    doctors = get_doctors(specialty=specialty)

    if not doctors:
        doctors = get_doctors()

    if not doctors:
        log({"step": "no_doctors"})
        return "No doctors available.", session_state

    doctor = doctors[0]
    log({"step": "doctor_selected", "doctor": doctor["name"]})

    if pending.get("date") and pending.get("time"):
        log({"step": "booking_db_call"})
        result = book_appointment(
            patient_id=PATIENT_ID,
            doctor_id=doctor["id"],
            appt_date=pending["date"],
            appt_time=pending["time"],
        )
        log({"step": "booking_done"})
        return result, session_state

    pending["doctor_id"] = doctor["id"]
    pending["awaiting"] = "slot_choice"
    session_state["pending_booking"] = pending

    return "Provide preferred date and time.", session_state