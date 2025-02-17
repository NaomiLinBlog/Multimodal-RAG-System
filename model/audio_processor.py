from typing import Dict, List
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

class AudioProcessor:
    def __init__(self, model_id: str = "andybi7676/cool-whisper-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Initialize the speech recognition model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Create the pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=256,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
    
    def process_audio(self, audio_path: str) -> List[Dict]:
        """
        Process audio file and return transcription with timestamps
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of dictionaries containing transcribed text and timestamps
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Process the audio file
        result = self.pipe(audio_path)
        
        # Format the results
        transcription = {
            "full_text": result["text"],
            "segments": result["chunks"]
        }
        
        return transcription