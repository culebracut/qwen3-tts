import os
import soundfile as sf
import torch # Useful for squeezing tensors

from model_core import QwenModelContainer
from config_manager import ConfigLoader
from persona_manager import PersonaManager
from voice_service import VoiceGenerationService

def main():
    # 1. Configuration Setup
    # Hardcoding the node 'audio_clips' or making it a variable
    #configFile = "/data/Qwen3-TTS/workspace/configs.json"
    configFile = "/data/Qwen3-TTS/workspace/data/single_wav_config.json"
    config_mgr = ConfigLoader(configFile)
    
    # 2. Component Initialization
    # model_path is pulled directly from project_metadata in your JSON
    model_core = QwenModelContainer(config_mgr.model_path)
    persona_mgr = PersonaManager(model_core)
    service = VoiceGenerationService(model_core, config_mgr, persona_mgr)

    print(f"🚀 Starting Production for audio clips")

    # 3. Execution Loop
    for result in service.generate_tasks():
        if result is None:
            continue

        # Extract data from the Service's result dictionary
        wav = result["wav"]
        sr = result["sr"]
        save_path = result["path"]
        task_key = result["key"]

        # Ensure the directory exists (handles nested folders from JSON)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Logic to handle 2D (batched) or 1D audio arrays safely
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        
        # 4. Save to Disk
        sf.write(save_path, wav[0], sr)
        print(f"✅ [{task_key}] Saved to: {save_path}")

    print("\n✨ All tasks in this node are complete.")

if __name__ == "__main__":
    main()
