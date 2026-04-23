import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://api-service:8000")

st.set_page_config(
    page_title="City Clinic — AI Assistant",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🏥 City Clinic")
st.sidebar.markdown("---")
st.sidebar.markdown("### What I can help with")
st.sidebar.markdown("""
- 📋 **Medical questions** — symptoms, medications, conditions
- 📅 **Book an appointment** — just describe your concern
- 👁️ **View appointments** — check your upcoming visits
- ❌ **Cancel appointment** — cancel by ID
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Example queries")
examples = [
    "What are the symptoms of diabetes?",
    "I have chest pain, I need an appointment",
    "Book me an appointment for tomorrow at 10 AM",
    "Show my appointments",
    "Cancel my appointment",
    "What are the side effects of metformin?",
]
for ex in examples:
    if st.sidebar.button(ex, use_container_width=True):
        st.session_state["prefill"] = ex

st.sidebar.markdown("---")
st.sidebar.caption("Powered by RAG + AI Agent · HuggingFace · Pinecone · GCP")

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "input" not in st.session_state:
    st.session_state.input = ""

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🏥 City Clinic AI Assistant")
st.caption("Ask medical questions or manage your appointments — all in one place.")
st.markdown("---")

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.history:
        st.info(
            "👋 Hello! I'm your clinic assistant. Ask me a medical question "
            "or say something like **'I need an appointment for a stomach issue'**."
        )
    else:
        for role, msg in st.session_state.history:
            if role == "You":
                with st.chat_message("user"):
                    st.markdown(msg)
            else:
                with st.chat_message("assistant", avatar="🏥"):
                    st.markdown(msg)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
st.markdown("---")

# Handle prefill from sidebar button
prefill = st.session_state.pop("prefill", None)
default_val = prefill if prefill else ""

user_input = st.chat_input("Type your message here...")

# Allow sidebar button to inject a message
if prefill and not user_input:
    user_input = prefill


def send_message(message: str):
    if not message.strip():
        return

    st.session_state.history.append(("You", message))

    try:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"patient_id": "demo_user", "message": message},
            timeout=600,
        )
        if resp.status_code == 200:
            answer = resp.json().get("response", "No response.")
        else:
            answer = f"API error {resp.status_code}: {resp.text}"
    except Exception as e:
        answer = f"Could not reach the API: {str(e)}"

    st.session_state.history.append(("Assistant", answer))
    st.rerun()


if user_input:
    send_message(user_input)

# ─────────────────────────────────────────────
# CLEAR CHAT BUTTON
# ─────────────────────────────────────────────
if st.session_state.history:
    if st.button("🗑️ Clear chat", type="secondary"):
        st.session_state.history = []
        st.rerun()
