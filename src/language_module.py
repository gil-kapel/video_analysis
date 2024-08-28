import openai
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

openai.api_key = config.get('api_key', '')

def get_gpt_answer(context, question):
    try:
        messages = [
            {"role": "system", "content": config.get('agent_character', 'You are a helpfully assistant')},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {question}"}
        ]
        response = openai.ChatCompletion.create(
            model= config.get('model'),
            messages=messages,
            max_tokens=config.get('max_tokens'),
            temperature=config.get('temperature'),
            n=1
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except Exception as e:
        return f"An error occurred: {e}"
