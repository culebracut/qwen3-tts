from QwenTTSModel import QwenModelContainer
from VoiceGenerationService import VoiceGenerationService

configPath = "/data/Qwen3-TTS/workspace/config.json"

def main():
    # 1. Instantiate the Model Class
    qwen_brain = QwenModelContainer()

    # 2. Instantiate the Logic Class (passing the model into it)
    tts_service = VoiceGenerationService(
        model_container=qwen_brain, 
        config_path=configPath
    )

    # 3. Execute
    tts_service.run_all()

if __name__ == "__main__":
    main()
