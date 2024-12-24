import requests
import json
import os

# API 網址
URL = "http://localhost:8000"

def test_add_documents():
    """測試添加不同類型的文檔"""
    documents = [
        # PDF 檔案來源
        {
            "text": "機器學習(Machine Learning)是人工智慧的一個分支，主要研究如何讓計算機系統從數據中學習和改進。監督式學習(Supervised Learning)是其中最常見的一種方法。",
            "metadata": {
                "source_type": "pdf",
                "file_name": "AI_course_notes.pdf",
                "page": 12,
                "section": "機器學習基礎概念",
                "course_name": "人工智慧導論",
                "semester": "112-1",
                "language": "zh-tw"
            },
            "source_type": "pdf",
            "page_number": 12
        },
        {
            "text": "深度學習(Deep Learning)中的神經網路是由多個層次組成，每一層都包含多個神經元。這種層次化的結構使得模型能夠學習更複雜的特徵表示。",
            "metadata": {
                "source_type": "pdf",
                "file_name": "AI_course_notes.pdf",
                "page": 25,
                "section": "神經網路架構",
                "course_name": "人工智慧導論",
                "semester": "112-1",
                "language": "zh-tw"
            },
            "source_type": "pdf",
            "page_number": 25
        },
        {
            "text": "卷積神經網路(CNN)的核心概念是使用卷積核進行特徵提取，這些特徵包括邊緣、紋理和更高階的視覺特徵。透過多層卷積和池化操作，模型可以學習到圖像的階層化表示。",
            "metadata": {
                "source_type": "pdf",
                "file_name": "AI_course_notes.pdf",
                "page": 45,
                "section": "卷積神經網路",
                "course_name": "人工智慧導論",
                "semester": "112-1",
                "language": "zh-tw"
            },
            "source_type": "pdf",
            "page_number": 45
        },
        
        # 影片講座來源
        {
            "text": "在進行資料預處理時，我們需要注意以下幾個步驟：首先是資料清理，處理缺失值和異常值；接著是特徵縮放，確保不同特徵的尺度一致；最後是特徵選擇，選擇最相關的特徵。",
            "metadata": {
                "source_type": "video",
                "file_name": "data_preprocessing_lecture.mp4",
                "start_time": "00:15:30",
                "end_time": "00:16:45",
                "chapter": "資料預處理基礎",
                "lecturer": "王教授",
                "course_name": "資料科學實務",
                "video_quality": "1080p",
                "transcript_confidence": 0.95
            },
            "source_type": "video",
            "timestamp": "00:15:30"
        },
        {
            "text": "特徵工程是資料預處理中的關鍵步驟。通過合適的特徵轉換和創建，我們可以幫助模型更好地學習數據中的模式。常見的方法包括標準化、正規化和編碼轉換。",
            "metadata": {
                "source_type": "video",
                "file_name": "data_preprocessing_lecture.mp4",
                "start_time": "00:25:00",
                "end_time": "00:26:30",
                "chapter": "特徵工程",
                "lecturer": "王教授",
                "course_name": "資料科學實務",
                "video_quality": "1080p",
                "transcript_confidence": 0.97
            },
            "source_type": "video",
            "timestamp": "00:25:00"
        }
    ]

    print("測試添加文件...")
    response = requests.post(
        f"{URL}/add_documents",
        json=documents
    )
    print("添加文件結果:", response.json())
    return response.status_code == 200

def test_add_pdf():
    """測試上傳 PDF 文件"""
    pdf_path = "./test_files/sample.pdf"  # 請確保此文件存在
    if not os.path.exists(pdf_path):
        print(f"找不到測試 PDF 文件: {pdf_path}")
        return False

    print("測試上傳 PDF...")
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{URL}/add_pdf",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")}
        )
    print("上傳 PDF 結果:", response.json())
    return response.status_code == 200

# test.py 中修改的部分
def test_add_video_transcript():
    """測試上傳影片字幕"""
    transcript_path = "test_files/sample_transcript.txt"  # 請確保此文件存在
    if not os.path.exists(transcript_path):
        print(f"找不到測試字幕文件: {transcript_path}")
        return False

    print("測試上傳影片字幕...")
    with open(transcript_path, "rb") as f:
        files = {"transcript": (os.path.basename(transcript_path), f, "text/plain")}
        data = {"video_name": "sample_video.mp4"}
        response = requests.post(
            f"{URL}/add_video_transcript",
            files=files,
            data=data  # 使用 data 而不是 json
        )
    print("上傳字幕結果:", response.json())
    return response.status_code == 200

def test_query():
    """測試查詢功能"""
    print("\n測試查詢功能...")
    
    queries = [
        {
            "query": "ECCV 2024 Corner Case Scene Understanding Challenge 的具體內容是什麼",
            "top_k": 2
        },
        {
            "query": "什麼是機器學習？",
            "top_k": 1
        }
    ]
    
    
    for query in queries:
        print(f"\n執行查詢: {query['query']}")
        response = requests.post(f"{URL}/query", json=query)
        
        if response.status_code == 200:
            result = response.json()
            print("\n回答:", result["response"])
            print("\n來源:")
            for source in result["sources"]:
                metadata = source["metadata"]
                if metadata["source_type"] == "pdf":
                    print(f"- [PDF] {metadata['file_name']} (第 {metadata['page']} 頁)")
                elif metadata["source_type"] == "video":
                    print(f"- [Video] {metadata['file_name']} (時間: {metadata.get('start_time', '')} - {metadata.get('end_time', '')})")
                print(f"  內容: {source['text']}\n")
        else:
            print("查詢失敗:", response.text)
    
    return response.status_code == 200

def run_all_tests():
    """運行所有測試"""
    print("開始運行測試...\n")
    
    # 創建測試文件目錄
    os.makedirs("test_files", exist_ok=True)
    
    # 創建測試用的字幕文件
    with open("test_files/sample_transcript.txt", "w", encoding="utf-8") as f:
        f.write("00:00:10 這是一個測試用的字幕文件。\n")
        f.write("00:00:20 用於測試多模態 RAG 系統的功能。\n")
    
    # 運行測試
    test_functions = [
        test_add_documents,
        test_add_pdf,
        test_add_video_transcript,
        test_query
    ]
    
    for test_func in test_functions:
        print(f"\n運行測試: {test_func.__name__}")
        try:
            success = test_func()
            if success:
                print(f"✓ {test_func.__name__} 測試通過")
            else:
                print(f"✗ {test_func.__name__} 測試失敗")
        except Exception as e:
            print(f"✗ {test_func.__name__} 測試出錯: {str(e)}")
    
    # 清理測試文件
    import shutil
    if os.path.exists("test_files"):
        shutil.rmtree("test_files")

if __name__ == "__main__":
    run_all_tests()
