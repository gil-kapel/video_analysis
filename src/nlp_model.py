import openai

openai.api_key = input('Insert your OpenAI API key:\n')

def get_gpt_answer(context, question):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {question}"}
        ]

        # Make the API request to OpenAI
        response = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo",  # Use the latest model available
            messages=messages,
            max_tokens=150,         # Adjust token length as required
            temperature=0.7,        # Control the creativity of the response
            n=1                     # Return one response
        )

        # Extract the generated answer from the response
        answer = response.choices[0].message['content'].strip()
        return answer

    except Exception as e:
        return f"An error occurred: {e}"
