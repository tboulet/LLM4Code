import os
from openai import AzureOpenAI


client = AzureOpenAI(
  azure_endpoint="https://petunia-gpt4o-mini.openai.azure.com/",
  api_key=os.getenv("AZURE_API_KEY"),
  api_version="2024-02-15-preview"
)


message_text = [
	{"role":"system","content":"You are an AI assistant that helps people find information."},
	{"role":"user","content":"What is the capital of Denmark?"}
]

completion = client.chat.completions.create(
  # Use GPT4o
  model="gpt-4o", # model = "deployment_name"
  messages = message_text,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None
)

print(completion.choices[0].message.content)


# how to have access to flowers group ?
# openai.NotFoundError: Error code: 404 - {'error': {'code': 'DeploymentNotFound', 'message': 'The API deployment for this resource does not exist. 
# If you created the deployment within the last 5 minutes, please wait a moment and try again.'}}