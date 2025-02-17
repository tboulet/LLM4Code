# Please install OpenAI SDK first: `pip3 install openai`

import os
from openai import OpenAI

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)