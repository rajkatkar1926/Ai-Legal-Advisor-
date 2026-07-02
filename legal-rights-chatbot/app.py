import os

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from retriever import retrieve_top_k

load_dotenv()

SYSTEM_INSTRUCTION = """You are Raksha, a warm and knowledgeable legal-rights assistant for women in India. You are having an ongoing conversation with someone who may be dealing with a real, sometimes stressful situation — your job is to actually help her work through it, not just recite a law.

For every turn:
- Use the conversation so far AND the "Context" block provided with this message (drawn from a curated legal knowledge base) to ground your answer. Don't invent legal claims that aren't supported by that context.
- Go beyond a bare definition: explain what the law means for HER situation, and give concrete, practical next steps (who to contact, what to file or write, what evidence/documents to keep, realistic timelines, what happens after she takes that step).
- If her message is vague or you're missing a detail that would change your advice (e.g. is this happening right now, who exactly is involved, has she already taken any action, is she safe), ask ONE short, specific clarifying question — you can ask it before giving general guidance, or alongside a first-pass answer. Don't interrogate her with a list of questions at once.
- Remember and build on earlier parts of this conversation — don't make her repeat context she already gave you, and refer back to it naturally (e.g. "since you mentioned this started a few weeks ago...").
- Be empathetic and validating in tone, without being alarmist or making assumptions about how she feels.
- Keep answers thorough but readable — use short paragraphs or bullet points for action steps rather than one dense block of text.

Always end your answer with: (1) the relevant helpline number(s) from the context, and (2) this exact disclaimer: 'This is general legal information, not a substitute for professional legal advice. For your specific situation, please consult a lawyer or contact the helpline above.' If the provided context doesn't clearly cover the question, say so honestly and give the general emergency/NALSA helpline instead of guessing."""


def get_api_key():
    return os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")


@st.cache_resource
def get_model():
    genai.configure(api_key=get_api_key())
    return genai.GenerativeModel(
        "gemini-flash-latest", system_instruction=SYSTEM_INSTRUCTION
    )


def format_context(entries):
    blocks = []
    for e in entries:
        note = " (NOTE: this is a weak match — say so if it doesn't clearly fit.)" if e["low_confidence"] else ""
        blocks.append(
            f"- Topic: {e['topic']}\n"
            f"  Law: {e['law']}\n"
            f"  Summary: {e['summary']}\n"
            f"  Helpline: {e['helpline']}{note}"
        )
    return "\n".join(blocks)


def get_chat_session():
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = get_model().start_chat(history=[])
    return st.session_state.chat_session


def build_retrieval_query(query, prior_user_messages, lookback=1):
    """Widen short follow-ups (e.g. 'what if they refuse?') with recent
    conversation context so TF-IDF still finds a relevant KB entry."""
    recent = prior_user_messages[-lookback:] if lookback else []
    return " ".join(recent + [query])


def generate_answer(query, prior_user_messages):
    retrieval_query = build_retrieval_query(query, prior_user_messages)
    retrieved = retrieve_top_k(retrieval_query, k=2)
    context = format_context(retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    chat = get_chat_session()
    response = chat.send_message(prompt)
    return response.text, retrieved


st.set_page_config(page_title="Raksha — Know Your Rights", page_icon="⚖️")

with st.sidebar:
    st.header("About")
    st.write(
        "This is a student project demonstrating a retrieval-augmented Q&A "
        "chatbot for Indian women's legal rights. It provides general "
        "informational content only — not legal advice."
    )
    st.header("Emergency Numbers")
    st.markdown(
        "- **112** — National Emergency\n"
        "- **181** — Women Helpline\n"
        "- **100** — Police"
    )
    if st.button("Start new conversation"):
        st.session_state.pop("messages", None)
        st.session_state.pop("chat_session", None)
        st.rerun()

st.title("⚖️ Raksha — Know Your Rights")
st.caption("Ask a question about Indian women's legal rights in plain English.")

if not get_api_key():
    st.error(
        "GEMINI_API_KEY is not set. Add it to a local `.env` file, or, if "
        "deployed on Streamlit Community Cloud, add it under your app's "
        "Settings → Secrets as `GEMINI_API_KEY = \"...\"`."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    flag = " _(low-confidence match)_" if s["low_confidence"] else ""
                    st.markdown(f"- **{s['topic']}** — {s['law']}{flag}")

if query := st.chat_input("e.g. what can I do if my landlord harasses me"):
    prior_user_messages = [
        m["content"] for m in st.session_state.messages if m["role"] == "user"
    ]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Looking this up..."):
            answer, sources = generate_answer(query, prior_user_messages)
        st.markdown(answer)
        with st.expander("Sources"):
            for s in sources:
                flag = " _(low-confidence match)_" if s["low_confidence"] else ""
                st.markdown(f"- **{s['topic']}** — {s['law']}{flag}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
