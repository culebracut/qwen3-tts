import torch
import os
import hashlib

class PersonaManager:
    def __init__(self, model_container, cache_dir="persona_cache"):
        self.engine = model_container
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache = {}

    def _get_cache_path(self, ref_audio):
        """Creates a unique filename based on the audio path hash."""
        # Use a hash to prevent issues with long paths or special characters
        path_hash = hashlib.md5(ref_audio.encode()).hexdigest()[:12]
        base_name = os.path.basename(ref_audio).split('.')[0]
        return os.path.join(self.cache_dir, f"{base_name}_{path_hash}.pt")

    def get_persona(self, ref_audio, ref_text):
        # 1. Check RAM first
        if ref_audio in self._memory_cache:
            return self._memory_cache[ref_audio]

        cache_path = self._get_cache_path(ref_audio)

        # 2. Check Disk
        if os.path.exists(cache_path):
            print(f"📦 Loading Persona from Disk: {os.path.basename(cache_path)}")
            # Load directly to the GPU/CPU device used by the model
            prompt = torch.load(cache_path, map_location=self.engine.device, weights_only=True)
        else:
            # 3. Generate New
            print(f"🎙️ Encoding New Persona: {os.path.basename(ref_audio)}")
            prompt = self.engine.create_prompt(ref_audio, ref_text)
            # Save for future use
            torch.save(prompt, cache_path)

        self._memory_cache[ref_audio] = prompt
        return prompt
