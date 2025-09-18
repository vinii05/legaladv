import os, PyPDF2, chromadb
from sentence_transformers import SentenceTransformer

# Setup Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("indian_laws")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf_text(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Load all PDFs in laws/ folder
folder = "laws"
for fname in os.listdir(folder):
    if fname.endswith(".pdf"):
        path = os.path.join(folder, fname)
        text = load_pdf_text(path)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        embeddings = embedder.encode(chunks)

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                embeddings=[embeddings[i]],
                ids=[f"{fname}_{i}"]
            )
        print(f"✅ Indexed {fname}")

print("🎉 All Indian law PDFs indexed into ChromaDB!")
