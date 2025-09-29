import os
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn
import json
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables (expects OPENAI_API_KEY)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY == "":
    print("WARNING: OPENAI_API_KEY is not set. The /api/chat endpoint will return example replies unless you set the key.")

openai.api_key = OPENAI_API_KEY

app = FastAPI(title="CareBot Backend")

# serve static frontend
# serve static frontend properly
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("index.html")


# Simple in-project FAQ knowledge base (faqs.json will be read)
FAQ_PATH = "faqs.json"
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faqs = json.load(f)

# Build TF-IDF vectorizer over FAQ "text" (question + answer)
faq_texts = [ (q["question"] + " " + q.get("answer","")) for q in faqs ]
vectorizer = TfidfVectorizer().fit(faq_texts)
faq_vectors = vectorizer.transform(faq_texts)

class Message(BaseModel):
    text: str

def retrieve_related(query, top_k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, faq_vectors)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = [{"score": float(sims[i]), "question": faqs[i]["question"], "answer": faqs[i].get("answer","")} for i in top_idx if sims[i] > 0]
    return results

@app.post("/api/chat")
async def chat(msg: Message):
    user_text = msg.text.strip()
    if not user_text:
        return JSONResponse({"reply":"Please type a message."})

    # 1) Retrieve relevant FAQ contexts
    retrieved = retrieve_related(user_text, top_k=3)
    context_text = ""
    if retrieved:
        context_text = "\n\nRelated FAQ excerpts:\n"
        for r in retrieved:
            context_text += f"- Q: {r['question']}\n  A: {r['answer']}\n"

    # 2) Create system prompt and call OpenAI (or return fallback if no API key)
    system_prompt = (
        "You are CareBot, a helpful and careful medical assistant. "
        "Provide general health information and non-diagnostic guidance. "
        "Always include a short safety disclaimer reminding users to consult a qualified healthcare professional for diagnosis or emergencies."
    )
    full_prompt = system_prompt + "\n\nUser question:\n" + user_text + ("\n\n" + context_text if context_text else "")

    if OPENAI_API_KEY == "":
        # Fallback reply for local testing without API key
        reply = "Demo reply: (OpenAI API key not set). Here are related FAQ excerpts:\n" + (context_text or "No related FAQ found.") + "\n\n⚠️ I am not a doctor. Please consult a healthcare professional for serious concerns."
        return JSONResponse({"reply": reply})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": user_text + ("\n\n" + context_text if context_text else "")}
            ],
            max_tokens=400,
            temperature=0.2,
        )
        reply_text = response.choices[0].message["content"].strip()
        # Ensure safety disclaimer
        if "consult" not in reply_text.lower():
            reply_text += "\n\n⚠️ I am not a doctor. Please consult a healthcare professional for serious concerns."
        return JSONResponse({"reply": reply_text})
    except Exception as e:
        return JSONResponse({"reply": f"Error contacting OpenAI: {str(e)}"})

@app.post("/api/upload_report")
async def upload_report(file: UploadFile = File(...)):
    save_dir = "uploaded_reports"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save file")
    return {"message": f"Saved {file.filename} to {file_path}"}

@app.get("/api/faqs")
def get_faqs():
    return faqs

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
