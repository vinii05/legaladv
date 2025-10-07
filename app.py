import os
import re
import json
import uuid
import fitz
import subprocess
import streamlit as st
from gtts import gTTS

# ---------------- CONFIG ----------------
MODEL_NAME = "llama3"   # Make sure: ollama pull llama3
SYSTEM_PROMPT = """
You are a legal assistant AI.
Translate legal language into plain English.
For each clause:
- Max 5 sentences explanation
- Assign a Risk level (Low / Medium / High)
- Give 1 short example
- Add related law reference
Overall:
- Provide a short summary
- Give up to 3 next steps
Always concise. Avoid repetition.
Always add: "Disclaimer: This is not legal advice."
"""

# ---------------- HELPERS ----------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def call_llama(messages):
    prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    cmd = ["ollama", "run", MODEL_NAME]
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return result.stdout.strip()

def clean_text_for_tts(text: str) -> str:
    return re.sub(r"[*_#`~>-]+", "", text).strip()

def make_tts(text, lang="en"):
    uid = str(uuid.uuid4())[:6]
    path = f"tts_{uid}.mp3"
    safe_text = clean_text_for_tts(text)
    gTTS(text=safe_text, lang=lang).save(path)
    return path

def split_into_clauses(text):
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return paras if paras else [text]

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("⚖️ AI Legal Assistant")
st.caption("Summarizes contracts clause by clause with risks, examples, law references, audio, and lets you ask questions.\n**Disclaimer: This is not legal advice.**")

# Sidebar mode switch
mode = st.sidebar.radio("Choose Mode:", ["📑 Clause Analyzer", "💬 Q&A Chatbot"])

uploaded_file = st.file_uploader("📂 Upload a contract (PDF)", type=["pdf"])
text_input = st.text_area("Or paste contract text", height=200)

# ---------------- CLAUSE ANALYZER ----------------
if mode == "📑 Clause Analyzer":
    if st.button("🔍 Analyze Contract"):
        if uploaded_file:
            contract_text = extract_text_from_pdf(uploaded_file)
        else:
            contract_text = text_input

        if not contract_text.strip():
            st.warning("Please upload or paste contract text.")
        else:
            with st.spinner("Analyzing contract..."):
                clauses_raw = split_into_clauses(contract_text)
                clause_outputs = []

                for i, clause in enumerate(clauses_raw):
                    clause_prompt = f"""
Analyze this clause:
{clause}

Return JSON with:
- explanation
- risk (Low / Medium / High)
- example
- law_reference
"""
                    raw = call_llama([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": clause_prompt}
                    ])

                    try:
                        cleaned = raw.strip().replace("```json", "").replace("```", "")
                        clause_json = json.loads(cleaned)
                    except:
                        clause_json = {
                            "explanation": "(AI summary unavailable)",
                            "risk": "Medium",
                            "example": "",
                            "law_reference": ""
                        }

                    clause_json["clause"] = clause
                    clause_json["audio_path"] = make_tts(clause_json["explanation"])
                    clause_outputs.append(clause_json)

                # ---- Overall summary & next steps ----
                summary = call_llama([
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":f"Give a short overall summary of this contract:\n{contract_text}"}
                ])
                next_steps = call_llama([
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":f"Suggest up to 3 next steps for this contract:\n{contract_text}"}
                ])
                overall_audio_path = make_tts(summary + "\nNext Steps:\n" + next_steps)

            # ---------------- DISPLAY ----------------
            st.subheader("📑 Overall Summary")
            st.write(summary)

            st.subheader("⚠️ Clause-by-Clause Analysis")
            risk_color = {"High": "#ffcccc", "Medium": "#fff0b3", "Low": "#ccffcc"}
            for i, c in enumerate(clause_outputs):
                with st.expander(f"Clause {i+1} (Risk: {c.get('risk','Medium')})", expanded=False):
                    st.markdown(
                        f"<div style='background:{risk_color.get(c.get('risk','Medium'),'#fff')};"
                        f"padding:10px;border-radius:6px;'>"
                        f"<b>Original:</b> {c['clause']}<br><br>"
                        f"<b>Explanation:</b> {c['explanation']}<br><br>"
                        f"<b>Law Reference:</b> {c['law_reference']}<br>"
                        f"<b>Example:</b> {c['example']}<br>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.audio(c["audio_path"], format="audio/mp3")

            st.subheader("📝 Next Steps")
            st.write(next_steps)

            st.subheader("🔊 Overall Audio")
            st.audio(overall_audio_path, format="audio/mp3")

# ---------------- Q&A CHATBOT ----------------
elif mode == "💬 Q&A Chatbot":
    user_query = st.text_area("Type your legal question here:")
    if st.button("Ask Question"):
        if uploaded_file:
            contract_text = extract_text_from_pdf(uploaded_file)
        else:
            contract_text = text_input

        if not contract_text.strip():
            st.warning("Please upload or paste contract text first.")
        elif not user_query.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Thinking..."):
                answer = call_llama([
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":f"Context:\n{contract_text}\n\nQuestion: {user_query}"}
                ])
                st.subheader("Answer")
                st.write(answer)

                audio_path = make_tts(answer, lang="en")
                st.audio(audio_path, format="audio/mp3")
