from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from .ASR_interface import ASR_interface
import torch

class Whisper(ASR_interface):
    def __init__(self):
        super().__init__()
        
        model_name = "openai/whisper-large"  # You can choose different model sizes
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        
    def transcribe_audio(self,audio_file):
        # Load audio
        input_values = self.tokenizer(audio_file, return_tensors="pt").input_values

        # Perform speech recognition
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the model output
        transcription = self.tokenizer.batch_decode(predicted_ids)

        return transcription[0]
    
    
    def input(self, voice):
        text = self.transcribe_audio(voice)
        return text
    
    




