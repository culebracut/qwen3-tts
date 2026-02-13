import soundfile as sf
from model_core import QwenModelContainer
from config_manager import ConfigLoader
from persona_manager import PersonaManager
from voice_service import VoiceGenerationService

def main():
    config = ConfigLoader("/data/Qwen3-TTS/workspace/data/config.json")
    
    # Initialization
    core = QwenModelContainer(config.model_path)
    personas = PersonaManager(core, cache_dir=config.cache_path)
    service = VoiceGenerationService(core, config, personas)

    # Loop through every persona ID found in the JSON
    for p_id in config.get_all_persona_ids():
        
        # get the task data
        task = config.get_persona(p_id)
        
        print(f"\n--- Processing Persona: {p_id} ---")
        print(f"Target Text: {task['text'][:50]}...") # First 50 chars
        print(f"Instructions: {task['instruct']}")
        print(f"Output Path: {task['full_output_path']}")
    
        # generate_audio(task)
        result = service.process_task(task)
        sf.write(result["path"], result["wav"], result["sr"])
        print(f"✅ Processed: {result['key']}")

if __name__ == "__main__":
    main()

