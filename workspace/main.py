import os
import soundfile as sf
from model_core import QwenModelContainer
from config_manager import ConfigLoader
from persona_manager import PromptCacheManager
from voice_service import VoiceGenerationService
from parse_script import parse_script


# 1. Load your config and your script
metadata = ConfigLoader("/data/Qwen3-TTS/workspace/data/config.json")

# Initialization 
core = QwenModelContainer(metadata.model_path)
personas = PromptCacheManager(core, cache_dir=metadata.cache_path)
service = VoiceGenerationService(core, metadata, personas)

# 2. Iterate through the lines in the script as name/value pairs
script = parse_script(metadata.script_path)

for line in script:
    speaker_id = line["speaker"]
    dialogue = line["text"]

    # 3. Lookup the Persona metadata
    persona = metadata.get_persona(speaker_id)

    if persona:
        # Insert the dialogue into the persona
        persona["text"] = dialogue
        
        print(f"\nActor: {speaker_id}")
        print(f"Dialog: {dialogue}.") # First 50 chars

        # generate_audio(task)
        # tts_engine.generate(task)
        result = service.process_task(persona)

        metadata.writer.write_chunk(result["wav"])
        print(f"✅ Processed dialogue line/n")
    else:
        print(f"Skipping {speaker_id}: No persona found in config.")
    
metadata.writer.close()
print(f"\n✅" + "="*30 + "\nScript Complete.")
