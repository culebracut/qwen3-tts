import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel

class QwenModelContainer:
    """Isolates the AI model loading and inference."""
    def __init__(self, model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        # Check for CUDA; fallback to CPU if necessary
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load model with bfloat16 for high efficiency on modern GPUs (A100/H100/30/40 series)
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, 
            device_map=self.device, 
            dtype=torch.bfloat16
        )

    def apply_seed(self,seed):
        """Resets all random number generators to a fixed state."""
        if seed is not None and seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensures internal GPU algorithms are consistent
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"Set seed to: {seed}")
        else:
            # Handle random generation if seed is -1 or missing
            new_seed = random.randint(0, 10**6)
            random.seed(new_seed)
            torch.manual_seed(new_seed)
            print(f"Using random seed: {new_seed}")
        
    def create_prompt(self, ref_audio, ref_text):
        """Processes reference audio into a voice vector."""
        # Use no_grad to save memory during prompt creation
        with torch.no_grad():
            return self.model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=False
            )
    
    def generate(self, text, language, prompt, instruct, seed, temp=0.7):
        """Performs the actual TTS generation."""
        
        self.apply_seed(seed)

        # Wrap in no_grad to prevent memory leaks and speed up generation
        with torch.no_grad():
            wav, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
                instruct=instruct,
                temperature=temp
            )
            return wav, sr
