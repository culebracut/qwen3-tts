class VoiceGenerationService:
    def __init__(self, model_container, config_manager, prompt_cache_manager):
        self.engine = model_container
        self.configs = config_manager
        self.prompt = prompt_cache_manager

    def process_task(self, persona, dry_run=False):
        if not persona:
            return None

        # Call Qwen to apply instruct characteristics to voice
        prompt = self.prompt.get_persona(persona)
        persona["prompt"] = prompt

        # Call Qwen to generate audio
        wav, sr = self.engine.generate(persona)
        
        return {
            "wav": wav, 
            "sr": sr, 
        }

    def generate_tasks(self):
        for key in self.personas:
            yield self.process_task(key)
