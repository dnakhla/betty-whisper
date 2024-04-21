# summary_generator.py

import requests
import json

async def generate_summary(transcript):
    print('Generating summary...')
    url = 'http://localhost:11434/api/chat'
    messages = [
        {
            'role': 'system',
            'content': 'You are an AI assistant that excels at summarizing transcripts. Often there are errors in the transcript so use your reasoning and context to first make sense of the topics and the transcript keeping in mind that words are often transcripted in correctly due to poor sound. Please follow the below steps to generate a summary:',
        },
        {
            'role': 'user',
            'content': f'Here is the transcript:\n\n{transcript}\n\nPlease generate a concise summary of the transcript with the following details:\n- Bullet-pointed summary of the main points\n- List of important details mentioned\n- Any follow-up actions or recommendations\n\nEnsure the response contains only the summary, without any additional explanations or text. at the bottom list "Questions Asked:" and put concise answers to any questions asked answered on the call and actually add answers and intelligence based on the transcript',
        },
    ]
    payload = {
        'model': 'llama3',
        'messages': messages,
        'stream': False,
        "options": {
            "num_keep": 5,
            "num_predict":10000,
            "seed": 42,
            "temperature": .2,
            "penalize_newline": False,
            "num_thread": 8
        }
    }
    headers = {
        'Content-Type': 'application/json',
    }
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response_data = response.json()
        summary = response_data['message']['content']  # Modify this line based on the actual response structure
        print('Summary generated.')
        return summary
    except requests.exceptions.RequestException as e:
        print('Error occurred while generating summary:', str(e))
        return None
    except (KeyError, ValueError) as e:
        print('Error occurred while parsing response:', str(e))
        return None