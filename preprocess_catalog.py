import json
import logging

logging.basicConfig(level=logging.INFO)

TYPE_MAPPING = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S"
}

def main():
    try:
        with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f, strict=False)
    except Exception as e:
        logging.error(f"Failed to load shl_product_catalog.json: {e}")
        return

    processed = []
    for item in raw_data:
        # Get link mapped to url
        url = item.get("link", "")
        # Try finding a valid test type
        keys = item.get("keys", [])
        test_type = "K" # Default
        test_type_full = "Knowledge & Skills"
        for k in keys:
            if k in TYPE_MAPPING:
                test_type = TYPE_MAPPING[k]
                test_type_full = k
                break # Just take the first valid one

        # extract duration
        dur_str = item.get("duration", "")
        try:
            dur = int(''.join(filter(str.isdigit, dur_str))) if dur_str else 0
        except:
            dur = 0
            
        processed_item = {
            "name": item.get("name", ""),
            "url": url,
            "test_type": test_type,
            "test_type_full": test_type_full,
            "remote_testing": item.get("remote", "no").lower() == "yes",
            "adaptive": item.get("adaptive", "no").lower() == "yes",
            "duration": dur,
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "job_families": item.get("job_families", []),
            "languages": item.get("languages", []),
            "competencies": item.get("keys", []) # We fall back keys to competencies or whatever the old code used
        }
        processed.append(processed_item)

    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)
    
    logging.info(f"Successfully processed {len(processed)} items into catalog.json")

if __name__ == "__main__":
    main()
