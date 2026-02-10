class VoiceGenerationService:
    def __init__(self, model_container, config_manager, persona_manager):
        self.engine = model_container
        self.configs = config_manager
        self.personas = persona_manager

    def process_task(self, key, dry_run=False):
        cfg = self.configs.get_task(key)
        if not cfg:
            return None

        # Extract the pre-merged path from our ConfigLoader
        save_path = cfg.get("full_output_path")

        # Actual AI execution
        persona = self.personas.get_persona(cfg["ref_audio"], cfg["ref_text"])
        wav, sr = self.engine.generate(
            text=cfg["text"],
            language=cfg["language"],
            prompt=persona,
            instruct=cfg["instruct"],
            seed=cfg['seed'],
            temp=cfg['temp']
        )
        
        return {
            "wav": wav, 
            "sr": sr, 
            "path": save_path, 
            "key": key
        }

    def generate_tasks(self):
        for key in self.configs.get_all_keys():
            yield self.process_task(key)
