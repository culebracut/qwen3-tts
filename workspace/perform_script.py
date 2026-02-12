from config_manager import ConfigLoader
from parse_script import parse_script

# 1. Load your config and your script
config = ConfigLoader("/data/Qwen3-TTS/workspace/data/config.json")
script = parse_script(config.script_path)

# 2. Iterate through the name-value pairs
for entry in script:
    speaker_id = entry["speaker"]
    dialogue = entry["text"]

    # 3. Lookup the Persona metadata from your Config Manager
    task = config.get_task(speaker_id)

    if task:
        # Override the placeholder text with the actual script dialogue
        task["text"] = dialogue
        
        print(f"Generating audio for {speaker_id}...")
        # tts_engine.generate(task)
    else:
        print(f"Skipping {speaker_id}: No persona found in config.")


print(f"\n" + "="*30 + "\nScript Complete.")
