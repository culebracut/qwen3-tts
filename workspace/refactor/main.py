import os
import soundfile as sf
import torch
from model_core import QwenModelContainer
from config_manager import ConfigLoader
from voice_persona_manager import PersonaManager
from voice_persona_manager import VoiceGenerationService

def main():
    config_path = "/data/Qwen3-TTS/workspace/data/single_wav_config.json"
    loader = ConfigLoader(config_path)
    
    # Init components
    core = QwenModelContainer(loader.model_path)
    personas = PersonaManager(core)
    service = VoiceGenerationService(core, loader, personas)

    for result in service.generate_tasks():
        wav, sr, save_path = result["wav"], result["sr"], result["path"]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Convert Torch Tensor to Numpy safely
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().float().numpy()
        
        # Handle batch dimension [1, samples] -> [samples]
        #if wav.ndim > 1:
        wav = wav[0]

        sf.write(save_path, wav, sr)
        print(f"✅ Saved: {result['key']} (Path: {save_path})")

if __name__ == "__main__":
    main()
