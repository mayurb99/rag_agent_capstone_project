import os
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from tools import book_appointment, get_appointments, cancel_appointment, get_doctors
from logger import log

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
MODEL = "allenai/Olmo-3-7B-Instruct:publicai"

PATIENT_ID = 1  # hardcoded demo_user maps to id=1 in patients table


# ─────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────

def call_llm(messages: list) -> str:
    log({"step": "llm_call_start", "prompt": messages[-1]["content"][:100]})
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
    )
    log({"step": "llm_call_end"})
    return response.choices[0].message.content.strip()


def detect_intent(user_input: str) -> str:
    prompt = f"""You are an intent classifier for a medical appointment assistant.

Classify the user message into EXACTLY ONE of these labels:
- book_appointment   (user wants to schedule/book a new appointment)
- view_appointments  (user wants to see/list/check their appointments)
- cancel_appointment (user wants to cancel/delete an appointment)
- faq                (user is asking a medical question, health info, symptoms, drugs)
- unknown            (none of the above)

Return ONLY the label, nothing else.

User message: "{user_input}"
Label:"""

    result = call_llm([{"role": "user", "content": prompt}])
    result = result.strip().lower().split()[0]
    valid = {"book_appointment", "view_appointments", "cancel_appointment", "faq", "unknown"}
    return result if result in valid else "unknown"


def extract_booking_details(user_input: str, existing: dict) -> dict:
    """Extract time, date, and symptom from user message."""
    prompt = f"""Extract booking details from the user message.
Return ONLY valid JSON with these keys (use null if not found):
{{
  "time": "HH:MM" or null,
  "date": "YYYY-MM-DD" or null,
  "symptom": "short description" or null
}}

Today's date is {os.popen("date +%Y-%m-%d").read().strip()}.
If user says "tomorrow", calculate tomorrow's date.
If user says "today", use today's date.

User message: "{user_input}"
JSON:"""

    try:
        raw = call_llm([{"role": "user", "content": prompt}])
        start = raw.find("{")
        end = raw.rfind("}") + 1
        extracted = json.loads(raw[start:end])
        # Merge with existing partial booking — don't overwrite already-found values
        for key in ["time", "date", "symptom"]:
            if existing.get(key) is None and extracted.get(key) is not None:
                existing[key] = extracted[key]
        return existing
    except Exception:
        return existing


def map_symptom_to_specialty(symptom: str) -> str:
    prompt = f"""Map this symptom/reason to a medical specialty.

Symptom: "{symptom}"

Choose from: General, Cardiology, Gastroenterology, Orthopedics, Neurology, Dermatology, Pediatrics
Return ONLY the specialty name."""

    return call_llm([{"role": "user", "content": prompt}]).strip()


def extract_appointment_id(user_input: str) -> int | None:
    prompt = f"""Extract the appointment ID number from this message.
Return ONLY the number, nothing else. If no number found, return null.

Message: "{user_input}"
ID:"""
    try:
        raw = call_llm([{"role": "user", "content": prompt}]).strip()
        return int(raw)
    except Exception:
        return None


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

def agent_loop(user_input: str, session_state: dict, rag_answer_fn) -> tuple[str, dict]:
    log({"step": "agent_start", "input": user_input})
    """
    Main agent loop.
    Returns (response_text, updated_session_state).
    session_state holds pending booking slots across turns.
    """

    # ── CHECK FOR PENDING BOOKING (multi-turn slot filling) ──────────────
    if session_state.get("pending_booking"):
        pending = session_state["pending_booking"]
        awaiting = pending.get("awaiting")

        if awaiting == "reason":
            # User just gave their symptom/reason
            pending["symptom"] = user_input
            pending["awaiting"] = None
            session_state["pending_booking"] = pending
            return _complete_booking_if_ready(pending, session_state)

        elif awaiting == "slot_choice":
            # User picked a date/time from the offered slots
            pending = extract_booking_details(user_input, pending)
            pending["awaiting"] = None
            session_state["pending_booking"] = pending
            return _complete_booking_if_ready(pending, session_state)

        elif awaiting == "cancel_id":
            # User picked which appointment to cancel
            appt_id = extract_appointment_id(user_input)
            session_state.pop("pending_booking", None)
            if appt_id:
                result = cancel_appointment(appt_id, PATIENT_ID)
                log({"action": "cancel_appointment", "id": appt_id, "result": result})
                return result, session_state
            else:
                return "I couldn't find a valid appointment ID. Please reply with the number (e.g. '3').", session_state

    # ── FRESH INTENT DETECTION ────────────────────────────────────────────
    log({"step": "before_detect_intent"})
    intent = detect_intent(user_input)
    log({"step": "after_detect_intent", "intent": intent})
    log({"input": user_input, "intent": intent})

    # ── FAQ → RAG ─────────────────────────────────────────────────────────
    if intent == "faq":
        log({"step": "faq_branch"})
        answer = rag_answer_fn(user_input)
        return answer, session_state

    # ── VIEW APPOINTMENTS ─────────────────────────────────────────────────
    elif intent == "view_appointments":
        log({"step": "view_appointments_branch"})
        result = get_appointments(PATIENT_ID)
        return result, session_state

    # ── CANCEL APPOINTMENT ────────────────────────────────────────────────
    elif intent == "cancel_appointment":
        log({"step": "cancel_branch"})
        appts = get_appointments(PATIENT_ID, raw=True)
        if not appts:
            return "You have no upcoming appointments to cancel.", session_state

        lines = ["Here are your scheduled appointments:\n"]
        for a in appts:
            lines.append(f"  #{a['id']} — {a['doctor_name']} ({a['specialty']})  |  {a['appt_date']}  {a['appt_time']}")
        lines.append("\nReply with the appointment ID to cancel (e.g. '3').")

        session_state["pending_booking"] = {"awaiting": "cancel_id"}
        return "\n".join(lines), session_state

    # ── BOOK APPOINTMENT ──────────────────────────────────────────────────
    elif intent == "book_appointment":
        log({"step": "book_branch"})
        pending = extract_booking_details(user_input, {})
        pending.setdefault("time", None)
        pending.setdefault("date", None)
        pending.setdefault("symptom", None)

        # Ask for missing reason first (most important for doctor matching)
        if not pending["symptom"]:
            session_state["pending_booking"] = {**pending, "awaiting": "reason"}
            return (
                "Sure! Could you briefly tell me what the visit is for? "
                "(e.g. fever, chest pain, routine checkup, stomach issue)"
            ), session_state

        session_state["pending_booking"] = pending
        return _complete_booking_if_ready(pending, session_state)

    # ── UNKNOWN / FALLBACK ────────────────────────────────────────────────
    else:
        answer = call_llm([
            {"role": "system", "content": "You are a helpful medical assistant. Be concise and friendly."},
            {"role": "user", "content": user_input}
        ])
        return answer, session_state


def _complete_booking_if_ready(pending: dict, session_state: dict) -> tuple[str, dict]:
    """Try to complete booking. Ask for missing slots if not ready."""
    log({"step": "complete_booking_start", "pending": pending})
    # Need symptom to find doctor
    if not pending.get("symptom"):
        session_state["pending_booking"] = {**pending, "awaiting": "reason"}
        return (
            "What is the reason for your visit? "
            "(e.g. fever, chest pain, routine checkup)"
        ), session_state

    # Map symptom to specialty and find doctor
    specialty = map_symptom_to_specialty(pending["symptom"])
    doctors = get_doctors(specialty=specialty)

    if not doctors:
        doctors = get_doctors()  # fallback to all doctors

    if not doctors:
        session_state.pop("pending_booking", None)
        return "Sorry, no doctors are currently available. Please try again later.", session_state

    doctor = doctors[0]  # pick first available match

    # If we have both date and time, book immediately
    if pending.get("date") and pending.get("time"):
        session_state.pop("pending_booking", None)
        result = book_appointment(
            patient_id=PATIENT_ID,
            doctor_id=doctor["id"],
            appt_date=pending["date"],
            appt_time=pending["time"],
        )
        log({"action": "book_appointment", "doctor": doctor["name"], "result": result})
        return result, session_state

    # Need date/time — show available slots
    slots_text = doctor.get("available_slots", "09:00, 10:00, 11:00, 14:00")
    days_text = doctor.get("available_days", "Mon-Fri")

    pending["doctor_id"] = doctor["id"]
    pending["awaiting"] = "slot_choice"
    session_state["pending_booking"] = pending

    return (
        f"I found **{doctor['name']}** ({doctor['specialty']}) for your concern.\n"
        f"Available days: {days_text}\n"
        f"Available slots: {slots_text}\n\n"
        f"Please tell me your preferred date and time. "
        f"(e.g. 'tomorrow at 10 AM' or '2026-04-25 at 11:00')"
    ), session_state
