import re

def parse_script(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to separate character names from their dialogue
    # We look for Uppercase names followed by a period (e.g., MISS PRISM.)
    segments = re.split(r'\n\n+', content.strip())
    
    script_data = []
    for i in range(0, len(segments), 2):
        # Clean up the ID: "MISS PRISM." -> "miss_prism"
        char_id = segments[i].replace(".", "").strip().lower().replace(" ", "_")
        
        # Get the text block immediately following the name
        if i + 1 < len(segments):
            text = segments[i+1].strip()
            script_data.append((char_id, text))
            
    return script_data
