# Raksha — Legal Rights Q&A Chatbot for Women

A small Streamlit chatbot that answers plain-English questions about Indian women's
legal rights using retrieval-augmented generation (RAG). Ask something like *"what is
the POSH Act"* or *"what can I do if my landlord harasses me"* and get a clear,
conversational answer grounded in a curated legal knowledge base, with a helpline
number and disclaimer attached to every response.

This is a student project built for demonstration/portfolio purposes — it is **not**
legal advice.

## Tech stack

- **Python 3.10+**
- **Streamlit** — chat UI
- **google-generativeai** (Gemini API, free tier — `gemini-flash-latest`) — answer
  generation
- **scikit-learn** (`TfidfVectorizer` + `cosine_similarity`) — retrieval step
- **python-dotenv** — loads the Gemini API key from a local `.env` file

## How RAG works in this project

The knowledge base (`knowledge_base.py`) is a small, hand-curated list of ~15 entries,
each covering one area of Indian law relevant to women (POSH Act, Domestic Violence
Act, IPC sections on harassment/stalking, tenancy rights, and more). At startup,
`retriever.py` fits a TF-IDF vectorizer over each entry's topic, law name, and summary
text. When a user asks a question, the app vectorizes the query with that same
vectorizer, computes cosine similarity against every KB entry, and retrieves the top 2
matches (flagging the result as low-confidence if even the best match scores poorly).
Those retrieved entries — not the whole knowledge base — are passed as context to
Gemini along with a system instruction that restricts it to answering from that
context, so the model's response stays grounded in real, sourced legal information
instead of freely generating (and potentially hallucinating) legal claims. Because the
knowledge base is so small, TF-IDF keyword similarity is sufficient here — no vector
database or embedding model is needed.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your free Gemini API key (get one at
   [Google AI Studio](https://aistudio.google.com/apikey)):
   ```bash
   cp .env.example .env
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploy (Streamlit Community Cloud)

The app is ready to deploy on [Streamlit Community Cloud](https://share.streamlit.io)
for free:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, then select:
   - Repository: your fork/copy of this repo
   - Branch: `main`
   - Main file path: `legal-rights-chatbot/app.py`
4. Before deploying, open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Click **Deploy**. The app reads the key from Streamlit's secrets automatically —
   no code changes needed.

## Project structure

```
legal-rights-chatbot/
  app.py             # Streamlit UI + Gemini generation logic
  retriever.py        # TF-IDF retrieval logic
  knowledge_base.py    # curated knowledge base (~15 entries)
  requirements.txt
  .env.example
  README.md
```

## Screenshot

<!-- Add a screenshot of the running app here, e.g.: ![Raksha chatbot screenshot](screenshot.png) -->

## Disclaimer

This tool provides general legal information for educational purposes only and is not
a substitute for professional legal advice. For real situations, please consult a
lawyer or contact one of the helplines shown in the app (Emergency 112, Women Helpline
181, Police 100).
