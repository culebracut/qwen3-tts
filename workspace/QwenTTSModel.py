import torch
from qwen_tts import Qwen3TTSModel

class QwenModelContainer:
    """Isolates the AI model loading and inference."""
    def __init__(self, model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, 
            device_map=self.device, 
            dtype=torch.bfloat16
        )

    def generate(self, text, language, prompt, instruct, temp=0.7):
        return self.model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            instruct=instruct,
            temperature=temp
        )
    
    def create_prompt(self, ref_audio, ref_text):
        return self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False
        )