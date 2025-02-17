from typing import List, Dict, Any
import torch
from audio_processor import AudioProcessor
from document_processor import DocumentProcessor, MultiModalDocument
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import os
import time
from PIL import Image
import io
import base64

class MultiModalRAG:
    def __init__(
        self,
        index_folder: str = "./storage",
        files_folder: str = "./test_files",
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        embed_model_name: str = "BAAI/bge-large-zh-v1.5",
        audio_model_id: str = "andybi7676/cool-whisper-hf",
        device: str = None,
        load_in_8bit: bool = True
    ):
        # 自動選擇設備
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            
        self.index_folder = index_folder
        self.files_folder = files_folder
        self.doc_processor = DocumentProcessor()
        self.image_retriever = self.doc_processor.initialize_image_retriever(files_folder)
        self.audio_processor = AudioProcessor(model_id=audio_model_id)

        # 設定嵌入模型
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            device=device
        )
        
        # 設定 LLM
        self.setup_llm(model_name, load_in_8bit)
        
        # 設定提示模板
        self.setup_prompts()
        
        # 載入或創建索引
        self.load_or_create_index()
        
    def setup_llm(self, model_name: str, load_in_8bit: bool):
        """設定語言模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda:0" else torch.float32,
            "device_map": "cuda:0",
        }
        
        if self.device == "cuda:0" and load_in_8bit:
            model_kwargs.update({
                "load_in_8bit": True,
            })
            
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **model_kwargs
        )

        # For llama_index compatibility
        self.llm = HuggingFaceLLM(
            tokenizer=self.tokenizer,
            model=self.model,
            device_map="cuda:0",
            context_window=512,
            max_new_tokens=128,
            generate_kwargs={
                "temperature": 0.3,
                "top_p": 0.85,
                "do_sample": True,
                "num_beams": 3
            }
        )
        
        Settings.llm = self.llm
        
    def setup_prompts(self):
        """設定提示模板"""
        self.qa_template = PromptTemplate(
            """以下是一些相關的上下文信息：
            ----------------
            {context_str}
            ----------------
            
            根據上述上下文和圖片資訊，請回答問題：{query_str}
            
            請以繁體中文回答，並盡可能提供完整和準確的資訊。如果沒有相關信息，請誠實地說明無法回答。回答："""
        )
        
    def load_or_create_index(self):
        """載入或創建索引"""
        if os.path.exists(self.index_folder):
            self.storage_context = StorageContext.from_defaults(
                persist_dir=self.index_folder
            )
            self.index = load_index_from_storage(
                storage_context=self.storage_context
            )
        else:
            self.index = None
            
    def add_documents(self, documents: List[MultiModalDocument]):
        """添加文檔到索引"""
        parser = SimpleNodeParser.from_defaults(
            chunk_size=256,
            chunk_overlap=30,
            include_metadata=True
        )

        llama_docs = [
            Document(
                text=doc.text,
                metadata={
                    **doc.metadata,
                    "source_type": doc.source_type,
                    "page_number": doc.page_number,
                    "timestamp": doc.timestamp
                }
            )
            for doc in documents
        ]
        
        nodes = parser.get_nodes_from_documents(llama_docs)
        
        if self.index is None:
            self.index = VectorStoreIndex(
                nodes,
                show_progress=True
            )
        else:
            self.index.insert_nodes(nodes)
            
        if not os.path.exists(self.index_folder):
            os.makedirs(self.index_folder)
        self.index.storage_context.persist(persist_dir=self.index_folder)
    
    def add_pdf(self, pdf_path: str):
        """添加 PDF 文件"""
        documents = self.doc_processor.process_pdf(pdf_path)
        self.add_documents(documents)
        
    def add_video(self, video_path: str, transcript_path: str):
        """添加影片及其字幕"""
        documents = self.doc_processor.process_video_transcript(
            transcript_path,
            video_path
        )
        self.add_documents(documents)

    def extract_assistant_response(self, response: str) -> str:
        markers = [
            "assistant", "assistant:", "ASSISTANT", "ASSISTANT:"
        ]
        
        # response = response.split("human:")[-1].split("user")[-1]
        
        for marker in markers:
            if marker in response:
                return response.split(marker)[-1].strip()
        
        return response.strip()

    def query(
            self,
            query_text: str,
            top_k: int = 3,
            response_mode: str = "tree_summarize",
        ) -> Dict[str, Any]:
            """整合查詢系統（優化記憶體使用）"""
            if not self.index:
                raise ValueError("索引尚未建立，請先添加文件")
                
            responses = {}
            start = time.time()
            print("開始查詢流程...")
            
            # 1. 文本檢索
            print("執行文本檢索...")
            try:
                # 清理 CUDA 緩存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=top_k
                )
                
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    text_qa_template=self.qa_template,
                    response_mode=response_mode
                )
                
                text_response = query_engine.query(query_text)
                text_content = str(text_response)
                
                # 收集文本來源
                sources = [{
                    "text": node.text,
                    "score": node.score if hasattr(node, 'score') else None,
                    "metadata": node.metadata
                } for node in text_response.source_nodes]
                
                responses['sources'] = sources
                
            except Exception as e:
                print(f"文本檢索錯誤: {str(e)}")
                return {"response": "文本檢索過程發生錯誤", "sources": [], "image_sources": []}
            
            # 2. 圖片檢索
            print("執行圖片檢索...")
            try:
                # 再次清理 CUDA 緩存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                image_results = self.image_retriever.search(query_text, k=min(top_k, 2))  # 限制圖片數量
                print(f"檢索到的圖片數量: {len(image_results)}")
                for idx, result in enumerate(image_results):
                    print(f"圖片 {idx + 1} - 頁碼: {result.page_num}")
                    print(f"圖片 {idx + 1} - 相似度分数: {result.score if hasattr(result, 'score') else 'N/A'}")
    
                # 準備圖片
                images = []
                max_size = (224, 224)
                for result in image_results:
                    try:
                        image_bytes = base64.b64decode(result.base64)
                        image = Image.open(io.BytesIO(image_bytes))
                        # 調整圖片大小
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        images.append(image)
                    except Exception as img_error:
                        print(f"圖片處理錯誤: {str(img_error)}")
                        continue
                
                if not images:
                    raise ValueError("沒有可用的圖片")

                # 準備多模態輸入
                conversation = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{query_text}\n\n文本上下文:\n{text_content}\n\n相關頁面:{', '.join([str(r.page_num) for r in image_results])}"
                        }
                    ]
                }]
                
                # 分批處理圖片
                for img in images:
                    conversation[0]["content"].append({"type": "image", "image": img})
                
                print("處理多模態輸入...")
                
                with torch.cuda.amp.autocast():
                    inputs = self.processor(
                        images=images,
                        text=self.processor.apply_chat_template(
                            conversation,
                            tokenize=False,
                            add_generation_prompt=True
                        ),
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad(): 
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=256, 
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                    
                    final_response = self.processor.batch_decode(
                        output_ids,
                        skip_special_tokens=True
                    )[0]
                    
                    final_response = self.extract_assistant_response(final_response)
                    print(type(final_response))

                    del inputs
                    del output_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"圖片處理錯誤: {str(e)}")
                final_response = text_content
                image_results = []
            
            # 最終回應
            result = {
                "response": final_response,
                "sources": sources,
                "image_sources": [
                    {
                        "page_number": r.page_num,
                        "doc_id": getattr(r, 'doc_id', 'N/A'),
                        "filename": (
                            os.listdir(self.files_folder)[r.doc_id] 
                            if 0 <= r.doc_id < len(os.listdir(self.files_folder)) 
                            else 'N/A'
                        ),
                        "score": getattr(r, 'score', 'N/A')
                    } 
                    for r in image_results
                ] if 'image_results' in locals() else []
            }
            
            print(f"查詢完成，總耗時: {time.time() - start:.2f}秒")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result