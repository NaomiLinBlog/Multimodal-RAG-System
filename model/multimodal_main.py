from fastapi import FastAPI, HTTPException, UploadFile, File, Form  # 添加 Form
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from multimodal_rag import MultiModalRAG
from document_processor import MultiModalDocument
import os
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
rag_instance = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentInput(BaseModel):
    text: str
    metadata: Optional[Dict] = None
    source_type: str
    page_number: Optional[int] = None
    timestamp: Optional[str] = None

class QueryInput(BaseModel):
    query: str
    top_k: Optional[int] = 1

@app.on_event("startup")
async def startup_event():
    global rag_instance
    rag_instance = MultiModalRAG(
        # model_name="yentinglin/Taiwan-LLM-7B-v2.0-base",
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        index_folder="./storage",
        device="cuda:0"  # 或 "cuda"
    )

@app.post("/add_documents")
async def add_documents(documents: List[DocumentInput]):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    docs = [
        MultiModalDocument(
            text=doc.text,
            metadata=doc.metadata or {},
            source_type=doc.source_type,
            page_number=doc.page_number,
            timestamp=doc.timestamp
        )
        for doc in documents
    ]
    
    rag_instance.add_documents(docs)
    return {"status": "success", "message": f"Added {len(docs)} documents"}

@app.post("/add_pdf")
async def add_pdf(file: UploadFile = File(...)):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # 保存上傳的文件
    file_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        rag_instance.add_pdf(file_path)
        return {"status": "success", "message": "PDF added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # finally:
    #     # 清理臨時文件
    #     if os.path.exists(file_path):
    #         os.remove(file_path)

@app.post("/add_video_transcript")
async def add_video_transcript(
    video_name: str = Form(...),  # 改用 Form
    transcript: UploadFile = File(...)
):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    transcript_path = f"./uploads/{transcript.filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    try:
        with open(transcript_path, "wb") as f:
            content = await transcript.read()
            f.write(content)
        
        rag_instance.add_video(video_name, transcript_path)
        return {"status": "success", "message": "Video transcript added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # finally:
    #     if os.path.exists(transcript_path):
    #         os.remove(transcript_path)

@app.post("/query")
async def query(query_input: QueryInput):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    result = rag_instance.query(
        query_text=query_input.query,
        top_k=query_input.top_k
    )
    return result

@app.post("/add_audio")
async def add_audio(audio_file: UploadFile = File(...)):
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Save uploaded audio file
    audio_path = f"./uploads/{audio_file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    try:
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Process audio file
        transcription = rag_instance.audio_processor.process_audio(audio_path)
        
        # Create documents from transcription
        documents = []
        for segment in transcription["segments"]:
            doc = MultiModalDocument(
                text=segment["text"],
                metadata={
                    "source_type": "audio",
                    "file_name": audio_file.filename,
                    "start_time": segment["timestamp"][0],
                    "end_time": segment["timestamp"][1]
                },
                source_type="audio",
                timestamp=f"{segment['timestamp'][0]:.2f}-{segment['timestamp'][1]:.2f}"
            )
            documents.append(doc)
        
        # Add documents to RAG system
        rag_instance.add_documents(documents)
        
        return {
            "status": "success",
            "message": "Audio processed and added successfully",
            "transcription": transcription["full_text"],
            "segments_count": len(documents)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # finally:
    #     if os.path.exists(audio_path):
    #         os.remove(audio_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

