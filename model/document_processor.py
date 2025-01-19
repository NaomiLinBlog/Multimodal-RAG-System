from typing import Dict, Any, Optional, List, Union, Tuple
import fitz  # PyMuPDF for PDF processing
import os
from dataclasses import dataclass
from pathlib import Path
from byaldi import RAGMultiModalModel

@dataclass
class MultiModalDocument:
    """多模態文檔的數據類"""
    text: str
    metadata: Dict[str, Any]
    source_type: str  # "pdf", "video", "audio", "text"
    page_number: Optional[int] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None

class DocumentProcessor:
    """處理不同類型文檔的類"""
    
    # Add this
    def initialize_image_retriever(self, pdf_path: str, index_name: str = "ds1") -> RAGMultiModalModel:
        """Initialize and build the index for the image retriever"""
        index_path = Path("./storage/.byaldi") / index_name
        if os.path.exists(index_path):
            print("Loading existing image index...")
            image_retriever = RAGMultiModalModel.from_index(index_name)
        else:
            print("Creating new image index...")
            image_retriever = RAGMultiModalModel.from_pretrained("vidore/colpali")
            image_retriever.index(
                input_path=Path(pdf_path),
                index_name=index_name,
                store_collection_with_index=True,
                overwrite=True
            )
        return image_retriever
    
    def process_pdf(self, pdf_path: str) -> List[MultiModalDocument]:
        """處理 PDF 文件"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        documents = []
        pdf_doc = fitz.open(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            text = page.get_text()
            
            doc = MultiModalDocument(
                text=text,
                metadata={
                    "file_name": file_name,
                    "page": page_num + 1,
                    "total_pages": len(pdf_doc)
                },
                source_type="pdf",
                page_number=page_num + 1
            )
            documents.append(doc)
            
        return documents

    def process_video_transcript(
        self,
        transcript_path: str,
        video_path: str
    ) -> List[MultiModalDocument]:
        """處理影片字幕文件"""
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
            
        documents = []
        video_name = os.path.basename(video_path)
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ' ' in line:
                    timestamp, text = line.strip().split(' ', 1)
                    
                    doc = MultiModalDocument(
                        text=text.strip(),
                        metadata={
                            "file_name": video_name,
                            "timestamp": timestamp,
                            "source_file": transcript_path
                        },
                        source_type="video",
                        timestamp=timestamp
                    )
                    documents.append(doc)
                    
        return documents

    def process_text(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> MultiModalDocument:
        """處理純文本"""
        return MultiModalDocument(
            text=text,
            metadata=metadata,
            source_type="text"
        )
