# SHL Assessment Recommender

Conversational agent that recommends SHL Individual Test Solutions via a
stateless FastAPI service.

---

## Quick Start

### 1. Get a free Gemini API key
Go to https://aistudio.google.com/app/apikey → create a key (free tier).

### 2. Scrape the SHL catalog
```bash
pip install playwright beautifulsoup4 requests
playwright install chromium

python scraper.py
# → creates catalog.json  (commit this file!)
```

> **If the scraper fails** (bot-detection, structure change):
> Open https://www.shl.com/solutions/products/product-catalog/?type=1 in
> your browser, open DevTools → Network, find the XHR call that loads the
> table data, and replicate it directly.  Then map fields to the catalog schema
> below and write them to `catalog.json` manually.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run locally
```bash
export GEMINI_API_KEY=your_key_here
uvicorn main:app --reload
```

Test it:
```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I am hiring a Java developer"}]}'
```

---

## catalog.json schema

Each item must have at least `name`, `url`, and `test_type`:

```json
[
  {
    "name":          "Verify - Numerical Reasoning",
    "url":           "https://www.shl.com/solutions/products/product-catalog/...",
    "test_type":     "A",
    "test_type_full":"Ability & Aptitude",
    "remote_testing": true,
    "adaptive":       false,
    "duration":       17,
    "description":   "Measures numerical reasoning ...",
    "job_levels":    ["Graduate", "Mid"],
    "job_families":  [],
    "languages":     ["English", "French"],
    "competencies":  []
  }
]
```

**test_type codes**

| Code | Full name |
|------|-----------|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgement |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| P | Personality & Behaviour |
| S | Simulations |

---

## Deploy to Render (free)

1. Push your repo (with `catalog.json` committed) to GitHub.
2. Go to https://render.com → New Web Service → connect your repo.
3. Render auto-detects `render.yaml`.
4. Add environment variable `GEMINI_API_KEY` in the Render dashboard.
5. Deploy.  Your public URL will be:
   `https://shl-recommender.onrender.com`

> **Cold-start note**: Render free tier sleeps after 15 min of inactivity.
> The first `/health` call allows up to 2 minutes (per the assignment spec).
> FAISS index build + model load takes ~30–40 s on Render free tier.

---

## API Reference

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a mid-level Java dev"},
    {"role": "assistant", "content": "What level of seniority?"},
    {"role": "user",      "content": "Around 4 years of experience"}
  ]
}
```

**Response**
```json
{
  "reply": "Got it. Here are 5 assessments …",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/…", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is **empty** while gathering context or refusing.
- `end_of_conversation` is `true` only when the task is fully complete.
- Max 10 recommendations per response.
- Max 8 turns per conversation (agent recommends by turn 4 at the latest).

---

## Architecture

```
POST /chat
   │
   ▼
agent.respond(messages)
   │
   ├─ build_query() — concatenate last 4 user turns
   │
   ├─ retriever.retrieve(query, top_k=25)
   │     └─ FAISS cosine-similarity search over catalog embeddings
   │        (all-MiniLM-L6-v2, 384-dim, ~22 MB)
   │
   ├─ format catalog context (top-25 items injected into system prompt)
   │
   ├─ Gemini 1.5 Flash (response_mime_type=application/json)
   │
   └─ validate: drop any URL not in catalog → return ChatResponse
```

---

## Local testing

```bash
python test_api.py
```

Runs a quick multi-turn smoke test against the local server.
