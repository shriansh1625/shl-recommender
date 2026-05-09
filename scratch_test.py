import os
import openai

api_key = "sk-58e3f71098594149856e5528d3b6178a"
client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello"}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=1024,
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    import traceback
    traceback.print_exc()
