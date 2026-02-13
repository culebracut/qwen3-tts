import os
import hashlib
import time  # Only needed here

class PersonaManager:
    def __init__(self, model_container, cache_dir="persona_cache"):
        self.engine = model_container
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache = {}

    def _get_ts(self):
        """Helper for formatted timestamps."""
        return time.strftime("%H:%M:%S")

    def get_persona(self, id, ref_audio, ref_text):
        ts = self._get_ts()
        
        # 1. RAM Cache
        if ref_audio in self._memory_cache:
            print(f"[{ts}] 🧠 MEMORY HIT: {os.path.basename(ref_audio)}")
            return self._memory_cache[ref_audio]

        path_hash = hashlib.md5(ref_audio.encode()).hexdigest()[:8]
        cache_path = os.path.join(self.cache_dir, f"{id}_{path_hash}.pt")

        # 2. Disk Cache
        if os.path.exists(cache_path):
            print(f"[{ts}] 💿 DISK LOAD: {os.path.basename(cache_path)}")
            prompt = self.engine.load_persona(cache_path)
        else:
            # 3. GPU Encoding (with duration tracking)
            print(f"[{ts}] 🎙️  GPU ENCODING START: {os.path.basename(ref_audio)}")
            
            start_time = time.perf_counter()
            prompt = self.engine.create_prompt(ref_audio, ref_text)
            duration = time.perf_counter() - start_time
            
            self.engine.save_persona(prompt, cache_path)
            print(f"[{self._get_ts()}] ✅ ENCODING COMPLETE ({duration:.2f}s)")

        self._memory_cache[ref_audio] = prompt
        return prompt
