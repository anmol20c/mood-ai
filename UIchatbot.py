import json
import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

APP_TITLE = "Mood AI"

MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# Limits that protect your API key when the app is public
MAX_MESSAGES_PER_SESSION = 50   # user messages per browser session
COOLDOWN_SECONDS = 2            # minimum gap between messages
MAX_INPUT_CHARS = 1500          # longest allowed user message

PERSONALITIES = {
    "😡 Angry": {
        "prompt": "you are an angry AI, you respond aggressively and impatiently.",
        "tagline": "Angry personality activated",
        "starters": ["Why is the sky blue?", "Give me a study tip", "Tell me about yourself"],
    },
    "😂 Funny": {
        "prompt": "you are a very funny AI, you respond with humor and jokes.",
        "tagline": "Funny personality activated",
        "starters": ["Tell me a joke", "Roast my coding skills", "Explain AI like a stand-up comic"],
    },
    "😢 Sad": {
        "prompt": "you are very sad, and respond negatively.",
        "tagline": "Sad personality activated",
        "starters": ["How's your day?", "Cheer me up", "What's the meaning of life?"],
    },
    "🏴‍☠️ Pirate": {
        "prompt": "you are a pirate AI, you talk like a swashbuckling pirate and use sea slang.",
        "tagline": "Pirate personality activated",
        "starters": ["Where's the treasure?", "Teach me Python", "What's for dinner?"],
    },
    "🎭 Shakespeare": {
        "prompt": "you are an AI that answers in the dramatic, poetic style of Shakespeare.",
        "tagline": "Shakespearean personality activated",
        "starters": ["Describe the internet", "Give me life advice", "What is love?"],
    },
    "💪 Motivator": {
        "prompt": "you are a hyper-energetic motivational coach AI, you hype the user up in every reply.",
        "tagline": "Motivator personality activated",
        "starters": ["I'm feeling lazy", "Help me plan my day", "I have an exam tomorrow"],
    },
    "😏 Sarcastic": {
        "prompt": "you are a sarcastic AI, you reply with dry wit but still give a correct answer.",
        "tagline": "Sarcastic personality activated",
        "starters": ["What's 2 + 2?", "Should I learn AI?", "Recommend a movie"],
    },
}

# --------------------------------------------------
# PAGE SETUP + CSS
# --------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a, #17172b, #10101c);
        color: white;
    }

    /* Hide only the footer and top color bar; keep the toolbar so the
       sidebar toggle, Deploy button and menu all keep working */
    footer, [data-testid="stDecoration"] {
        visibility: hidden;
    }
    header[data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] { background: #12121f; }

    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #ff6b9d, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #aaaabd;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }
    .personality {
        text-align: center;
        padding: 6px;
        margin: 0 0 20px 0;
        color: #c4b5fd;
        font-size: 0.95rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 8px;
        margin-bottom: 10px;
    }
    [data-testid="stChatInput"] { border-radius: 18px; }

    div[data-baseweb="select"] > div {
        background-color: #1c1c2e;
        border: 1px solid #3a3a55;
        border-radius: 12px;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------


def get_api_key():
    """Look in Streamlit secrets first (for hosting), then environment/.env."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner=False)
def get_model(model_name: str, temperature: float, api_key: str):
    """Create the model once per settings combo and reuse it across reruns/users."""
    return ChatGroq(model=model_name, temperature=temperature, api_key=api_key)


def init_state():
    st.session_state.setdefault("histories", {})   # separate chat per personality
    st.session_state.setdefault("msg_count", 0)
    st.session_state.setdefault("last_ts", 0.0)
    st.session_state.setdefault("pending", None)   # starter-button prompt


def get_history(mode_name: str):
    histories = st.session_state.histories
    if mode_name not in histories:
        histories[mode_name] = [SystemMessage(content=PERSONALITIES[mode_name]["prompt"])]
    return histories[mode_name]


def trim_history(messages, keep_last: int):
    """Always keep the system message, plus only the most recent messages."""
    return [messages[0]] + messages[1:][-keep_last:]


def stream_reply(model, messages):
    for chunk in model.stream(messages):
        if chunk.content:
            yield chunk.content


def export_text(messages, mode_name):
    lines = [f"{APP_TITLE} chat - {mode_name} - {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in messages[1:]:
        who = "You" if isinstance(m, HumanMessage) else "Bot"
        lines.append(f"{who}: {m.content}\n")
    return "\n".join(lines)


def export_json(messages, mode_name):
    data = {
        "app": APP_TITLE,
        "personality": mode_name,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages[1:]
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

init_state()

with st.sidebar:
    st.header("⚙️ Settings")

    mode_name = st.selectbox("🎭 Personality", list(PERSONALITIES.keys()))
    model_name = st.selectbox("🧠 Model", MODELS)
    temperature = st.slider("🌡️ Creativity", 0.0, 1.5, 0.9, 0.1,
                            help="Higher = more random and creative, lower = more focused.")
    keep_last = st.slider("🧵 Memory (recent messages)", 2, 40, 20, 2,
                          help="How many recent messages the bot remembers. Lower = faster and cheaper.")

    st.divider()

    history = get_history(mode_name)
    chat_messages = history[1:]

    if st.button("🗑️ Clear this chat", use_container_width=True, disabled=not chat_messages):
        st.session_state.histories[mode_name] = [SystemMessage(content=PERSONALITIES[mode_name]["prompt"])]
        st.rerun()

    if chat_messages:
        st.download_button("⬇️ Download as .txt", export_text(history, mode_name),
                           file_name="mood_ai_chat.txt", use_container_width=True)
        st.download_button("⬇️ Download as .json", export_json(history, mode_name),
                           file_name="mood_ai_chat.json", mime="application/json",
                           use_container_width=True)

    st.divider()
    remaining = MAX_MESSAGES_PER_SESSION - st.session_state.msg_count
    st.caption(f"💬 Messages left this session: {max(remaining, 0)}")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(f'<div class="main-title">🤖 {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Pick a personality in the sidebar and start chatting.</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="personality">{mode_name} · {PERSONALITIES[mode_name]["tagline"]}</div>',
            unsafe_allow_html=True)

# --------------------------------------------------
# API KEY CHECK
# --------------------------------------------------

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it to your `.env` file locally, "
             "or to Streamlit secrets when hosting.")
    st.stop()

model = get_model(model_name, temperature, api_key)

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in chat_messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# Starter prompts on an empty chat
if not chat_messages:
    st.caption("Try one of these:")
    cols = st.columns(len(PERSONALITIES[mode_name]["starters"]))
    for col, starter in zip(cols, PERSONALITIES[mode_name]["starters"]):
        if col.button(starter, use_container_width=True):
            st.session_state.pending = starter
            st.rerun()

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

prompt = st.chat_input("Type your message...")
if not prompt and st.session_state.pending:
    prompt = st.session_state.pending
st.session_state.pending = None

if prompt:
    prompt = prompt.strip()

    # Public-use safeguards
    if len(prompt) > MAX_INPUT_CHARS:
        st.warning(f"Message too long. Please keep it under {MAX_INPUT_CHARS} characters.")
    elif st.session_state.msg_count >= MAX_MESSAGES_PER_SESSION:
        st.warning("You've reached the message limit for this session. Refresh the page to start again.")
    elif time.time() - st.session_state.last_ts < COOLDOWN_SECONDS:
        st.warning("Slow down a little 🙂 and try again in a moment.")
    else:
        st.session_state.last_ts = time.time()
        history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)

        try:
            with st.chat_message("assistant"):
                reply = st.write_stream(stream_reply(model, trim_history(history, keep_last)))
            history.append(AIMessage(content=reply))
            st.session_state.msg_count += 1
        except Exception as e:
            history.pop()  # remove the unanswered user message so history stays valid
            st.error("Something went wrong while contacting the AI. Please try again in a moment.")
            with st.expander("Error details"):
                st.code(str(e))
        else:
            st.rerun()  # refresh so sidebar buttons and counters update