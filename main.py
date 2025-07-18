from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import generate_session_id, chat_with_session, process_pdf, create_collections

import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the upload directory exists
os.makedirs("uploaded_files", exist_ok=True)

class AskRequest(BaseModel):
    session_id: str
    question: str

# Endpoint to generate new session ID
@app.get("/get_session")
def get_session():
    return {"session_id": generate_session_id()}

# Endpoint to ask questions with session ID
@app.post("/ask")
def ask(request: AskRequest):
    response = chat_with_session(request.session_id, request.question)
    return {"answer": response}

# WebSocket chat endpoint
@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            reply = chat_with_session(session_id, question)
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        print(f"❌ Client {session_id} disconnected")

# ✅ New endpoint to upload PDF
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    file_path = f"uploaded_files/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process PDF (text extraction, image OCR, embeddings, etc.)
    process_pdf(file_path)

    return {"message": "PDF uploaded and processed successfully."}

@app.on_event("startup")
def startup_event():
    create_collections()
