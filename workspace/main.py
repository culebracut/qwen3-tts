import os
import soundfile as sf
from model_core import QwenModelContainer
from config_manager import ConfigLoader
from persona_manager import PersonaManager
from voice_service import VoiceGenerationService

def main():
    loader = ConfigLoader("/data/Qwen3-TTS/workspace/data/config.json")
    
    # Initialization
    core = QwenModelContainer(loader.model_path)
    personas = PersonaManager(core, cache_dir=loader.cache_path)
    service = VoiceGenerationService(core, loader, personas)

    for result in service.generate_tasks():
        save_path = result["path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # result["wav"] is already a (Samples, 1) Numpy array
        sf.write(save_path, result["wav"], result["sr"])
        print(f"✅ Processed: {result['key']}")

if __name__ == "__main__":
    main()

