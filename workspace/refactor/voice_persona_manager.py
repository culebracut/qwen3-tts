class PersonaManager:
    def __init__(self, model_container):
        self.engine = model_container
        self._cache = {}

    def get_persona(self, ref_audio, ref_text):
        # Cache by audio path to avoid re-encoding the same voice
        if ref_audio not in self._cache:
            self._cache[ref_audio] = self.engine.create_prompt(ref_audio, ref_text)
        return self._cache[ref_audio]

class VoiceGenerationService:
    def __init__(self, model_container, config_manager, persona_manager):
        self.engine = model_container
        self.configs = config_manager
        self.personas = persona_manager

    def generate_tasks(self):
        for key in self.configs.get_all_keys():
            cfg = self.configs.get_task(key)
            persona = self.personas.get_persona(cfg["ref_audio"], cfg["ref_text"])
            
            wav, sr = self.engine.generate(
                text=cfg["text"],
                language=cfg["language"],
                prompt=persona,
                instruct=cfg["instruct"],
                seed=cfg["seed"],
                temp=cfg["temp"]
            )
            
            yield {
                "wav": wav, "sr": sr, "path": cfg["full_output_path"], "key": key
            }
