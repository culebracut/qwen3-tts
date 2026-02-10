import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel

class QwenModelContainer:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, 
            device_map=self.device, 
            dtype=torch.bfloat16
        )

    def apply_seed(self, seed):
        """Resets RNG for bit-perfect reproducibility per task."""
        if seed is None or seed == -1:
            seed = random.randint(0, 1000000)
            
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
            return wav, sr
