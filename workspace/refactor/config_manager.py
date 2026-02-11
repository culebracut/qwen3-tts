import json
import os
import copy

class ConfigLoader:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        metadata = json_data.get("project_metadata", {})
        self.base_output_dir = metadata.get("file_path", "")
        self.model_path = metadata.get("model_path")
        self.cache_path = metadata.get("cache_path")
        
        # Global defaults if missing in individual clips
        self.default_seed = metadata.get("seed", 42)
        self.default_temp = metadata.get("temperature", 0.7)
        self.audio_clips = json_data.get('audio_clips', {})

    def get_task(self, key):
        raw_task = self.audio_clips.get(key)
        if not raw_task:
            return None

        task = copy.deepcopy(raw_task)
        # Prioritize local clip values, fall back to global defaults
        task["seed"] = task.get("seed", self.default_seed)
        task["temp"] = task.get("temp", self.default_temp)
        
        filename = task.get("output_file", f"{key}.wav")
        task["full_output_path"] = os.path.join(self.base_output_dir, filename)
        return task

    def get_all_keys(self):
        return list(self.audio_clips.keys())
