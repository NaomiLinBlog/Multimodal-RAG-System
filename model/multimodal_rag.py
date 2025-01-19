from typing import List, Dict, Any, Union
import torch
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
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoProcessor, 
    LlavaForConditionalGeneration
)
import os
import time
from PIL import Image
import io
import base64

class MultiModalRAG:
    def __init__(
        self,
        index_folder: str = "./storage",
        model_name: str = "yentinglin/Taiwan-LLM-7B-v2.0-base",
        embed_model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = None,  # 自動選擇設備
        load_in_8bit: bool = True,  # 啟用 8-bit 量化
        llava_model_name: str = "llava-hf/llava-interleave-qwen-7b-hf" ## Add this
    ):
        # 自動選擇設備
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            # 清理 CUDA 緩存
            torch.cuda.empty_cache()

        """
        初始化多模態 RAG 系統
        """    
        self.index_folder = index_folder
        self.doc_processor = DocumentProcessor()
        #self.image_retriever = None
        self.image_retriever = self.doc_processor.initialize_image_retriever("./test_files")

        # 設定嵌入模型
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            device=device
        )
        
        # 設定文本 LLM
        self.setup_llm(model_name, load_in_8bit)
        
        # 設定圖像 LLM (LLaVA)
        print("--------Initialize LLaVA--------")
        self.setup_image_llm(llava_model_name)
        print("--------Finished loading LLaVA--------")
        #Settings.llm = self.image_model
        # 設定提示模板
        self.setup_prompts()
        
        # 載入或創建索引
        self.load_or_create_index()
        
    def setup_image_llm(self, model_name: str):
        """設定圖像語言模型 (LLaVA)"""
        self.image_model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True
        ).to(self.device)
        self.image_processor = AutoProcessor.from_pretrained(model_name)
        
    def setup_llm(self, model_name: str, load_in_8bit: bool):
        """設定文本語言模型"""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if self.device == "cuda" and load_in_8bit:
            model_kwargs.update({
                "device_map": "auto",
                "load_in_8bit": True,
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.llm = HuggingFaceLLM(
            tokenizer=tokenizer,
            model=model,
            device_map="auto" if self.device == "cuda" else None,
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
            
            根據上述上下文，請回答問題：{query_str}
            
            請以繁體中文回答，並盡可能提供完整和準確的資訊。如果上下文中沒有相關信息，請誠實地說明無法回答。回答："""
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
        
        # 將 MultiModalDocument 轉換為 llama_index Document
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
    
    def add_pdf(self, pdf_path: str, process_images: bool = True):
        """添加 PDF 文件"""
        documents = self.doc_processor.process_pdf(pdf_path)
        self.add_documents(documents)
        
        # Add this
        # if process_images --> initialize image retriever
        #if process_images:
        #    self.image_retriever = self.doc_processor.initialize_image_retriever(pdf_path)
        
        return documents
    
    def add_video(self, video_path: str, transcript_path: str):
        """添加影片及其字幕"""
        documents = self.doc_processor.process_video_transcript(
            transcript_path,
            video_path
        )
        self.add_documents(documents)
        
    def query(
        self,
        query_text: str,
        use_image: bool = True, # Add this
        top_k: int = 3,
        response_mode: str = "tree_summarize"
    ) -> Dict[str, Any]:
        """查詢系統"""
        if not self.index:
            raise ValueError("索引尚未建立，請先添加文件")
        
        # Collection of responses of different modalities
        responses = {} # Add this
        
        # 文本查詢
        start = time.time()
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        print(f"設定檢索器時間: {time.time() - start:.2f}秒")

        t1 = time.time()
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            text_qa_template=self.qa_template,
            response_mode=response_mode
        )
        print(f"建立查詢引擎時間: {time.time() - t1:.2f}秒")

        t2 = time.time()
        text_response = query_engine.query(query_text)
        responses['text'] = str(text_response) # Add this
        print(f"生成回答時間: {time.time() - t2:.2f}秒")
        
        sources = [{
            "text": node.text,
            "score": node.score if hasattr(node, 'score') else None,
            "metadata": node.metadata
        } for node in text_response.source_nodes]
        responses['source'] = sources


        if use_image:
            print('use image')
        if self.image_retriever:
            print('actually exist image retriever')
        # retrieve image 
        if use_image and self.image_retriever:
            try:
                image_results = self.image_retriever.search(query_text, k=top_k)
                
                # conversation with multiple image
                conversation = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{query_text}\n\nBased on PDF pages: {', '.join([str(r.page_num) for r in image_results])}"
                        },
                    ],
                }]
                
                for result in image_results:
                    conversation[0]["content"].append({"type": "image"})
                
                # use LLaVA for generation
                prompt = self.image_processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True
                )
                
                # prepare image
                images = []
                for r in image_results:
                    image_bytes = base64.b64decode(r.base64)
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)

                inputs = self.image_processor(
                    #images=[Image.open(io.BytesIO(base64.b64decode(r.base64))).resize((224, 224))
                    images=images,
                    text=prompt,
                    return_tensors='pt'
                    #padding=True
                ).to(self.device, torch.float16)
                
                output = self.image_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    #pad_token_id=self.image_processor.tokenizer.pad_token_id
                )
                
                responses['image'] = self.image_processor.decode(
                    output[0][2:],
                    skip_special_tokens=True
                )
                
            except Exception as e:
                print(f"image part error: {str(e)}")
                responses['image'] = ""
        
        return responses
