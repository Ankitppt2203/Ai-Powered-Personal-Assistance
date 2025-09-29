
# CareBot - Medical Chatbot (Frontend + Backend + RAG scaffold)

This project contains a static **frontend (index.html)** and a **FastAPI backend** (`backend.py`) that:
- Serves the frontend.
- Provides `/api/chat` which performs a simple TF-IDF retrieval over a small FAQ dataset and calls OpenAI ChatCompletion (if `OPENAI_API_KEY` is set).
- Provides `/api/upload_report` to save uploaded files to `uploaded_reports/`.
- Provides `/api/faqs` to list built-in FAQs.

## Quick start (local)

1. Create a Python virtual environment and install packages:
```bash
python -m venv venv
source venv/bin/activate   # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Set your OpenAI API key (optional but recommended):
```bash
export OPENAI_API_KEY="sk-..."    # Linux / macOS
setx OPENAI_API_KEY "sk-..."     # Windows (restart required)
```

3. Run the server:
```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

4. Open `http://localhost:8000/index.html` in your browser.

## Notes & Next steps
- The project uses a simple TF-IDF + cosine similarity for retrieval (so you don't need heavy vector DBs).
- For production, add authentication, HTTPS, logging, and stricter safety filters.
- Consider using a managed vector DB (Pinecone, Weaviate) and better embeddings for improved RAG.
- Add PDF parsing (e.g., `pdfminer.six`) to index uploaded reports into the knowledge base.
- Ensure legal and regulatory compliance for medical applications before deploying to users.

