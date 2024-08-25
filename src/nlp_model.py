import openai

# openai.api_key = input('Insert your OpenAI API key:\n')

def get_gpt_answer(context, question, agent_character:str = 'helpful assistant'):
    try:
        messages = [
            {"role": "system", "content": f"You are {agent_character}."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {question}"}
        ]
        response = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            n=1
        )
        answer = response.choices[0].message['content'].strip()
        return answer

    except Exception as e:
        return f"An error occurred: {e}"
