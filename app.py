import os, re, json, uuid, fitz, requests, chromadb
from gtts import gTTS
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemma-3-4b-it"

SYSTEM_PROMPT = """
You are a legal assistant AI.
Translate legal language into plain words **very concisely**.
For each clause:
- 1 sentence explanation max
- Risk level (Low / Medium / High)
- 1 very short example if needed
Overall:
- Max 2 questions for a lawyer
- Max 3 next steps, 3–5 words each
Always concise. Avoid repetition.
Always add: "Disclaimer: This is not legal advice."
"""

LANG_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Bengali": "bn"
}

# ---------------- CHROMADB ----------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("indian_laws")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- HELPERS ----------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def call_gemma(messages):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": messages, "max_tokens": 300}
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def make_tts(text, lang_code="en"):
    uid = str(uuid.uuid4())[:6]
    path = f"tts_{uid}.mp3"
    gTTS(text=text, lang=lang_code).save(path)
    return path

def split_into_clauses(text):
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return paras if paras else [text]

def retrieve_law_context(query):
    q_embedding = embedder.encode([query])[0]
    results = collection.query(query_embeddings=[q_embedding], n_results=3)
    if results and "documents" in results:
        return " ".join(results["documents"][0])
    return ""

def translate_text(text, target_language):
    if target_language == "English":
        return text
    prompt = f"Translate this text into {target_language} very concisely:\n{text}"
    return call_gemma([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Legal Assistant (Super Concise + Multi-Language)", layout="wide")
st.title("⚖️ AI Legal Assistant ")
st.caption("Clause-level summaries, Indian law RAG, multi-language TTS.\n**Disclaimer: This is not legal advice.**")

uploaded_file = st.file_uploader("📂 Upload a contract (PDF)", type=["pdf"])
text_input = st.text_area("Or paste contract text", height=200)
language = st.selectbox("🌐 Output Language", list(LANG_OPTIONS.keys()))

if st.button("🔍 Analyze Contract"):
    if uploaded_file:
        contract_text = extract_text_from_pdf(uploaded_file)
    else:
        contract_text = text_input

    if not contract_text.strip():
        st.warning("Please upload or paste contract text.")
    else:
        with st.spinner("Analyzing contract concisely..."):
            law_context = retrieve_law_context(contract_text)
            clauses_raw = split_into_clauses(contract_text)
            concise_clauses = []

            # Clause-level analysis + translation + TTS
            for i, c in enumerate(clauses_raw):
                clause_prompt = f"Analyze this clause concisely:\n{c}\nRelevant Indian Law Context:\n{law_context}"
                raw = call_gemma([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": clause_prompt}
                ])
                try:
                    cleaned = re.sub(r"```json|```", "", raw).strip()
                    clause_json = json.loads(cleaned)
                    clause_json["explanation"] = clause_json["explanation"].split(".")[0] + "."
                except:
                    clause_json = {"clause": c, "explanation": "(AI summary unavailable)", "risk": "Medium", "example": ""}
                
                # Translate clause explanation if needed
                clause_json["explanation_translated"] = translate_text(clause_json["explanation"], language)
                # Generate clause-level TTS
                clause_json["audio_path"] = make_tts(clause_json["explanation_translated"], LANG_OPTIONS[language])
                
                concise_clauses.append(clause_json)

            # Overall summary, questions, next steps
            summary_prompt = f"Give 1 short summary paragraph with top risks:\n{contract_text}\nIndian Law Context:\n{law_context}"
            summary = call_gemma([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":summary_prompt}])
            summary = summary.split(".")[0] + "."

            questions_prompt = f"Suggest up to 2 questions for a lawyer:\n{contract_text}\nIndian Law Context:\n{law_context}"
            questions = call_gemma([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":questions_prompt}])

            next_steps_prompt = f"List up to 3 next steps (3–5 words each):\n{contract_text}\nIndian Law Context:\n{law_context}"
            next_steps = call_gemma([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":next_steps_prompt}])

            # Translate overall summary and next steps
            translated_summary = translate_text(summary, language)
            translated_next_steps = translate_text(next_steps, language)
            overall_audio_path = make_tts(translated_summary + "\n\nNext Steps:\n" + translated_next_steps, LANG_OPTIONS[language])

        # ---------------- DISPLAY ----------------
        st.subheader("📑 Summary")
        st.write(translated_summary)

        st.subheader("⚠️ Clause-by-Clause Analysis")
        risk_color = {"High": "#833939", "Medium": "#d17387", "Low": "#7ec191"}
        for i, c in enumerate(concise_clauses):
            with st.expander(f"Clause {i+1} (Risk: {c.get('risk','Medium')})", expanded=False):
                st.markdown(
                    f"<div style='background:{risk_color.get(c.get('risk','Medium'),'#fff')};"
                    f"padding:10px;border-radius:6px;'>"
                    f"<b>Original:</b> {c['clause']}<br>"
                    f"<b>Explanation:</b> {c['explanation_translated']}<br>"
                    f"<b>Example:</b> {c.get('example','')}<br>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.audio(c["audio_path"], format="audio/mp3")

        st.subheader("❓ Suggested Questions for Lawyers")
        st.write(questions)

        st.subheader("📝 Next Steps Checklist")
        steps = translated_next_steps.split("\n")
        for i, step in enumerate(steps):
            if step.strip():
                st.checkbox(step.strip(), key=f"step_{i}")

        st.subheader("🔊 Overall Audio")
        st.audio(overall_audio_path, format="audio/mp3")
