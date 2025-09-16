
---

# 📄 `app.py`

```python
import os, re, json, uuid, fitz, requests
import streamlit as st
from gtts import gTTS

# ---------------- CONFIG ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # set in Streamlit Secrets
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemma-3-4b-it"

LANG_OPTIONS = {
    "English":"en",
    "Hindi":"hi",
    "Tamil":"ta",
    "Malayalam":"ml",
    "Kannada":"kn",
    "Bengali":"bn"
}

SYSTEM_PROMPT = """
You are a legal assistant AI.
Translate complex legal language into plain words.
For each clause: produce a short explanation (1-2 sentences), a risk level (Low / Medium / High), and one short example.
Also generate 2-3 questions a person should ask a lawyer, and a short checklist of next steps.
Always add: "Disclaimer: This is not legal advice."
"""

# ---------------- HELPERS ----------------
def extract_text_from_pdf(file):
    """Extract text from uploaded PDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def call_gemma(messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": MODEL_NAME, "messages": messages}
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def make_tts(text, lang_code):
    uid = str(uuid.uuid4())[:6]
    path = f"tts_{uid}.mp3"
    gTTS(text=text, lang=lang_code).save(path)
    return path

def split_into_clauses(text):
    """Split into clauses (simple heuristic)."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return paras if paras else [text]

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("⚖️ AI Legal Assistant")
st.caption("Summarizes contracts, highlights risks, suggests questions, and narrates in multiple languages. \n**Disclaimer: This is not legal advice.**")

uploaded_file = st.file_uploader("📂 Upload a contract (PDF)", type=["pdf"])
text_input = st.text_area("Or paste contract text", height=200)
language = st.selectbox("🎧 Narration Language", list(LANG_OPTIONS.keys()))

if st.button("🔍 Analyze Contract"):
    if uploaded_file:
        contract_text = extract_text_from_pdf(uploaded_file)
    else:
        contract_text = text_input

    if not contract_text.strip():
        st.warning("Please upload or paste contract text.")
    else:
        with st.spinner("Analyzing with Gemma..."):
            # Clause analysis
            clause_prompt = f"""
            Break this contract into clauses.
            For each clause return JSON with keys: clause, explanation, risk (Low/Medium/High), example.
            Text:\n{contract_text}
            """
            raw = call_gemma([
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":clause_prompt}
            ])
            try:
                cleaned = re.sub(r"```json|```", "", raw).strip()
                clauses = json.loads(cleaned)
            except:
                clauses = [
                    {"clause":c, "explanation":"(AI summary unavailable)", "risk":"Medium", "example":""}
                    for c in split_into_clauses(contract_text)
                ]

            # Suggested questions
            questions = call_gemma([
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":"Suggest 3 simple questions to ask a lawyer:\n"+contract_text}
            ])
            
            # Next steps
            next_steps = call_gemma([
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":"List 5 next actions (plain language):\n"+contract_text}
            ])
            
            # Summary
            summary = call_gemma([
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":"Give one short summary paragraph:\n"+contract_text}
            ])

        # ---- Display ----
        st.subheader("📑 Summary")
        st.write(summary)

        st.subheader("⚠️ Clause-by-Clause Analysis")
        risk_color = {"High":"#ffcccc","Medium":"#fff2cc","Low":"#e6ffed"}
        for i, c in enumerate(clauses):
            with st.expander(f"Clause {i+1} (Risk: {c.get('risk','Medium')})", expanded=False):
                st.markdown(
                    f"<div style='background:{risk_color.get(c.get('risk','Medium'),'#fff')};"
                    f"padding:10px;border-radius:6px;'>"
                    f"<b>Original:</b> {c['clause']}<br><br>"
                    f"<b>Explanation:</b> {c['explanation']}<br>"
                    f"<b>Example:</b> {c.get('example','')}"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.subheader("❓ Suggested Questions for Lawyers")
        st.write(questions)

        st.subheader("📝 Next Steps Checklist")
        st.write(next_steps)

        # TTS
        lang_code = LANG_OPTIONS.get(language,"en")
        audio_path = make_tts(summary + "\n\nNext Steps:\n" + next_steps, lang_code)
        st.audio(audio_path, format="audio/mp3")
