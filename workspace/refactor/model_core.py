import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

# 1. Move the security allowlist here
torch.serialization.add_safe_globals([VoiceClonePromptItem])

class QwenModelContainer:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, 
            device_map=self.device, 
            dtype=torch.bfloat16
        )

    def apply_seed(self, seed):
        if seed is None or seed == -1:
            seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def create_prompt(self, ref_audio, ref_text):
        with torch.no_grad():
            return self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False
            )
    
    def generate(self, text, language, prompt, instruct, seed, temp):
        self.apply_seed(seed)
        with torch.no_grad():
            wav, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
                instruct=instruct,
                temperature=temp
            )
            
            # 2. Convert to Numpy HERE so main.py doesn't need Torch
            if torch.is_tensor(wav):
                wav = wav.cpu().float().numpy()
            
            # Ensure 1D array for soundfile
            wav = wav[0]
                
            return wav, sr
