import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

# Allow the specific Qwen data type for security
torch.serialization.add_safe_globals([VoiceClonePromptItem])

class QwenModelContainer:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, 
            device_map=self.device, 
            dtype=torch.bfloat16
        )

    def save_persona(self, prompt, path):
        """Hides torch.save from other classes."""
        torch.save(prompt, path)

    def load_persona(self, path):
        """Hides torch.load and device mapping from other classes."""
        return torch.load(path, map_location=self.device, weights_only=True)

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
    
    def generate(self, persona):
        self.apply_seed(persona["seed"])
        with torch.no_grad():
            wav, sr = self.model.generate_voice_clone(
                text=persona["text"],
                language=persona["language"],
                ref_audio=persona["ref_audio"],
                ref_text=persona["ref_text"],
                x_vector_only_mode="False",
                voice_clone_prompt=persona["prompt"],
                non_streaming_mode="False",
                temperature=persona["temp"]
            )
            
            # Convert to Numpy and fix shape for Soundfile
            if torch.is_tensor(wav):
                wav = wav.cpu().float().numpy()
            
            # Squeeze and Reshape to (Samples, 1) for soundfile compatibility
            wav = np.squeeze(wav)
            if wav.ndim == 1:
                wav = wav.reshape(-1, 1)
                
            return wav, sr
