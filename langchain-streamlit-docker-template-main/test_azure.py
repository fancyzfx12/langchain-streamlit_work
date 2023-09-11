import os
import requests
import json
import openai

openai.api_type = "azure"
openai.api_version = "2023-05-15" 
openai.api_key = "c33ce426568e41448a5f942ec58a4bda"
openai.api_base = "https://oh-ai-openai-scu.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
deployment_name='gpt-35-turbo' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
# Send a completion call to generate an answer
print('Sending a test completion job')
# start_phrase = 'Write a tagline for an ice cream shop. '
start_phrase = 'What would be a good company name for a company that makes colorful socks?'
# start_phrase = 'Tell me a joke.'
# start_phrase = "who are you?"


response = openai.Completion.create(engine=deployment_name, model="gpt-35-turbo", prompt=start_phrase, max_tokens=10)
text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
print(start_phrase+text)







# import os
# import openai
# openai.api_type = "azure"
# openai.api_version = "2023-05-15" 
# openai.api_base = os.getenv("https://oh-ai-openai-scu.openai.azure.com/")  # Your Azure OpenAI resource's endpoint value.
# openai.api_key = os.getenv("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")

# response = openai.ChatCompletion.create(
#     engine="gpt-35-turbo", # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
#     messages=[
#         {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
#         {"role": "user", "content": "Who were the founders of Microsoft?"}
#     ]
# )
# print(response)
# print(response['choices'][0]['message']['content'])













# # import os
# # os.environ["OPENAI_API_KEY"] = "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"

# import openai
# openai.api_type = "azure"
# openai.api_key = "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"
# openai.api_base = "https://example-endpoint.openai.azure.com"
# openai.api_version = "2023-05-15"

# # create a chat completion
# chat_completion = openai.ChatCompletion.create(deployment_id="deployment-name", model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])

# # print the completion
# print(chat_completion.choices[0].message.content)


# import os
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"]="2023-05-15"
# os.environ["OPENAI_API_BASE"]="https://example-endpoint.openai.azure.com"
# os.environ["OPENAI_API_KEY"]="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"
# # create a chat completion
# # chat_completion = openai.ChatCompletion.create(deployment_id="deployment-name", model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])

# # # print the completion
# # print(chat_completion.choices[0].message.content)

# import openai
 
# response = openai.Completion.create(
#     engine="text-davinci-002-prod",
#     prompt="This is a test",
#     max_tokens=5
# )
# print(response)