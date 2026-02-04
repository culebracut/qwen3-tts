import os
import json
import soundfile as sf

class VoiceGenerationService:
    """Handles the logic of processing configs and file management."""
    def __init__(self, model_container, config_path):
        self.engine = model_container
        with open(config_path, 'r', encoding='utf-8') as f:
            self.configs = json.load(f)

    def process_task(self, key):
        cfg = self.configs.get(key)
        if not cfg:
            print(f"Key {key} not found.")
            return

        # Prepare audio paths
        os.makedirs(os.path.dirname(cfg["output_file"]), exist_ok=True)

        # Logic flow: Create Prompt -> Generate Wav
        prompt = self.engine.create_prompt(cfg["ref_audio"], cfg["ref_text"])
        
        wav, sr = self.engine.generate(
            text=cfg["text"],
            language=cfg["language"],
            prompt=prompt,
            instruct=cfg["instruct"]
        )

        outputFile = cfg["output_file"]
        sf.write(outputFile, wav[0], sr)
        print(f"Finished: {key}")

    def run_all(self):
        for key in self.configs.keys():
            self.process_task(key)
