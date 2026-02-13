import os
import soundfile as sf
from model_core import QwenModelContainer
from config_manager import ConfigLoader
from persona_manager import PersonaManager
from voice_service import VoiceGenerationService
from parse_script import parse_script

# 1. Load your config and your script
config = ConfigLoader("/data/Qwen3-TTS/workspace/data/config.json")
script = parse_script(config.script_path)

# Initialization
core = QwenModelContainer(config.model_path)
personas = PersonaManager(core, cache_dir=config.cache_path)
service = VoiceGenerationService(core, config, personas)

# 2. Iterate through the name-value pairs
for entry in script:
    speaker_id = entry["speaker"]
    dialogue = entry["text"]

    # 3. Lookup the Persona metadata from your Config Manager
    task = config.get_task(speaker_id)

    if task:
        # Override the placeholder text with the actual script dialogue
        task["text"] = dialogue
        
        #print(f"Generating audio for {speaker_id}...")
        # tts_engine.generate(task)
        #print(f"\n--- Processing Persona: {speaker_id} ---")
        print(f"\nActor: {speaker_id}")
        print(f"Dialog: {task['text'][:200]}.") # First 50 chars
        #print(f"Instructions: {task['instruct']}\n")

        # generate_audio(task)
        result = service.process_task(task)

        # Check if file exists to determine mode
        # 'a' for append, 'w' for create new
        file_path = "/data/audio/output/earnest.wav"
        #mode = 'a' if os.path.exists(file_path) else 'w'
        #with sf.SoundFile(file_path, mode=mode, samplerate=result["sr"], channels=1) as f:
        #    f.write(result["wav"])
        if not os.path.exists(file_path):
            # First time? Create the file normally
            sf.write(file_path, result["wav"], result["sr"])
        else:
            # Append mode using r+
           with sf.SoundFile(file_path, mode='r+') as f:
            f.seek(0, sf.SEEK_END) # Go to the very end of the audio
            f.write(result["wav"])

        #sf.write(result["path"], result["wav"], result["sr"])
        #print(f"✅ Processed: {result['key']}")
    else:
        print(f"Skipping {speaker_id}: No persona found in config.")

print(f"\n" + "="*30 + "\nScript Complete.")
