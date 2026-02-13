from pathlib import Path
import re

def format_script_to_single_line(input_path):
    input_file = Path(input_path)
    output_file = input_file.with_name(f"{input_file.stem}_oneline.txt")

    raw_text = input_file.read_text(encoding='utf-8')

    # Regex logic: 
    # Find Name + Period + Newline + Text
    # Replace with Name + Colon + Space + Text
    pattern = re.compile(r'^([A-Z\s]{2,})\.\s*\n\s*(.*)', re.MULTILINE)
    formatted = pattern.sub(r'\1: \2', raw_text)

    # Clean up: Remove leading/trailing space from each line
    # but keep lines with content
    lines = [line.strip() for line in formatted.splitlines() if line.strip()]
    output_file.write_text("\n".join(lines), encoding='utf-8')
    
    print(f"✅ Formatted file created: {output_file.name}")
    return output_file

# --- Execution ---
#input_file_path = "/data/Qwen3-TTS/workspace/data/whos_on_first.txt"
input_file_path = "/data/Qwen3-TTS/workspace/data/importance_of_being_earnest.txt"
format_script_to_single_line(input_file_path)
