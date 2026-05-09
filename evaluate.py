import logging
import os
import re
import requests
import json
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_URL = "http://localhost:8000"

def extract_expected_urls(trace_text):
    """Extract all valid SHL catalog URLs listed in the expected markdown table."""
    # Find markdown links: [text](https://www.shl.com...) or raw <https://www.shl.com...>
    urls = set()
    links = re.findall(r'(https://www\.shl\.com/products/product-catalog/view/[^\s<>)]+)', trace_text)
    for l in links:
        urls.add(l)
    return urls

def extract_user_turns(trace_text):
    """Extract what the User says in the turn"""
    turns = []
    # User messages usually follow **User**\n\n> text
    matches = re.finditer(r'\*\*User\*\*\s*>\s*(.*?)(?=\n\n\*\*|\n\n###|\Z)', trace_text, re.DOTALL)
    for m in matches:
        turns.append(m.group(1).strip())
    return turns

def run_trace(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    user_turns = extract_user_turns(content)
    expected_urls = extract_expected_urls(content)

    if not expected_urls:
         logging.warning(f"{filename}: No expected URLs found in trace!")

    messages = []
    for i, turn in enumerate(user_turns):
        messages.append({"role": "user", "content": turn})
        
        # We should wait slightly between hit to not trigger rate limiting on free gemini
        time.sleep(5) 

        resp = requests.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=60)
        
        if resp.status_code == 429 or resp.status_code == 500:
             logging.warning(f"Rate limited or Server Error! Sleeping 30s")
             time.sleep(30)
             resp = requests.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=60)
             
        resp.raise_for_status()
        data = resp.json()
        
        messages.append({"role": "assistant", "content": data["reply"]})

        if data.get("end_of_conversation"):
            break

    # Re-evaluate
    predicted_urls = {r["url"] for r in data.get("recommendations", [])}
    
    if len(expected_urls) > 0:
        matches = predicted_urls.intersection(expected_urls)
        recall = len(matches) / len(expected_urls)
    else:
        recall = 1.0

    logging.info(f"{filename}: Turns={len(messages)//2}, Expected={len(expected_urls)}, Predicted={len(predicted_urls)}, Matches={len(matches) if expected_urls else 0}, Recall@10={recall:.2f}")
    return recall

def main():
    traces = []
    for i in range(1, 11):
        filename = f"C{i}.md"
        if os.path.exists(filename):
            traces.append(filename)

    if not traces:
        logging.error("No traces found!")
        return

    logging.info(f"Running evaluation on {len(traces)} traces...")
    recalls = []
    
    for t in traces:
        try:
            recall = run_trace(t)
            recalls.append(recall)
        except Exception as e:
            logging.error(f"Error on {t}: {e}")
            recalls.append(0.0)
            
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    logging.info(f"=== EVALUATION COMPLETE ===")
    logging.info(f"Mean Recall@10: {mean_recall:.2%}")

if __name__ == "__main__":
    main()
