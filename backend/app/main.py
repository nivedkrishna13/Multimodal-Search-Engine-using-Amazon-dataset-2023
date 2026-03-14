from fastapi import FastAPI
from pydantic import BaseModel
from backend.app.services.search_service import search_text,search_image 
from fastapi import UploadFile, File
import shutil
from backend.app.services.voice_service import transcribe_audio
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multimodal Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Define the structure of your incoming request
class SearchRequest(BaseModel):
    query: str

@app.get("/")
def health():
    return {"status": "Search engine running 🚀"}

@app.post("/search/text")
def search(request: SearchRequest): # Changed from 'query: str' to the Pydantic model
    # Access the string via request.query
    results = search_text(request.query)
    return {"results": results}


@app.post("/search/image")
def search_by_image(file: UploadFile = File(...)):
    results = search_image(file.file)
    return {"results": results}
@app.post("/search/voice")
def search_by_voice(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text_query = transcribe_audio(temp_path)

    os.remove(temp_path)

    results = search_text(text_query)

    return {
        "transcribed_text": text_query,
        "results": results
    }

#  uvicorn backend.app.main:app --reload
# http://127.0.0.1:8000/docs 