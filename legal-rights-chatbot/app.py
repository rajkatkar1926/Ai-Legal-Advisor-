import os

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from retriever import retrieve_top_k

load_dotenv()

SYSTEM_INSTRUCTION = """You are a helpful legal-rights assistant for women in India. Answer ONLY using the provided context below. Be clear, warm, and practical. Always end your answer with: (1) the relevant helpline number(s) from the context, and (2) this exact disclaimer: 'This is general legal information, not a substitute for professional legal advice. For your specific situation, please consult a lawyer or contact the helpline above.' If the provided context doesn't clearly cover the question, say so honestly and give the general emergency/NALSA helpline instead of guessing."""


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


def generate_answer(query):
    retrieved = retrieve_top_k(query, k=2)
    context = format_context(retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    model = get_model()
    response = model.generate_content(prompt)
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
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Looking this up..."):
            answer, sources = generate_answer(query)
        st.markdown(answer)
        with st.expander("Sources"):
            for s in sources:
                flag = " _(low-confidence match)_" if s["low_confidence"] else ""
                st.markdown(f"- **{s['topic']}** — {s['law']}{flag}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
