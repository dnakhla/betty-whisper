import pyaudio
import numpy as np
import whisper
from faster_whisper import WhisperModel
import webrtcvad
import queue
import threading
import os
from datetime import datetime
import requests
import json
import asyncio

# Set the sample rate and frame duration
sample_rate = 16000  # in Hz
frame_duration = 30  # in ms

# Set the VAD parameters
vad = webrtcvad.Vad()
vad_mode = 3  # Aggressive mode
min_speech_duration = 250  # in ms

# Create an instance of the Whisper model with default options
model = WhisperModel("base", device="cpu", compute_type="int8")

# Default values for options
default_task = "transcribe"
default_beam_size = 5
default_best_of = 5
default_patience = None
default_length_penalty = None
default_temperature = 0.0
default_compression_ratio_threshold = 2.4
default_log_prob_threshold = None
default_no_speech_threshold = 0.6
default_condition_on_previous_text = True
default_initial_prompt = None
default_word_timestamps = True
default_prepend_punctuations = "\\\"'\u00BF([{-"
default_append_punctuations = "\\\"'.\u3002,\uFF0C!\uFF01?\uFF1F:\uFF1A\")]}\u3001"
default_vad_filter = True

# Audio recording and transcription parameters
audio_queue = queue.Queue()
recording = True

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time as a string
datetime_string = current_datetime.strftime("%Y%m%d_%H%M%S")
output_file = f"transcriptions_{current_datetime.strftime('%m-%d-%Y__%A__%I:%M:%S%p')}.txt"

# Create the "transcriptions" directory if it doesn't exist
os.makedirs("transcriptions", exist_ok=True)

# Set the full path for the output file
output_file_path = os.path.join("transcriptions", output_file)

def record_audio():
    global recording
    chunk_size = int(sample_rate * frame_duration // 1000)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

    while recording:
        data = stream.read(chunk_size)
        audio_queue.put(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

def transcribe_audio():
    global recording
    frames = []
    speech_frames = []

    with open(output_file_path, "w") as file:
        file.write("Transcript:" + "\n")
        while recording or not audio_queue.empty():
            if not audio_queue.empty():
                data = audio_queue.get()
                frames.append(data)
                is_speech = vad.is_speech(data, sample_rate)

                if is_speech:
                    speech_frames.append(data)
                else:
                    if len(speech_frames) > 0:
                        speech_duration = len(speech_frames) * frame_duration
                        if speech_duration >= min_speech_duration:
                            audio_data = np.frombuffer(b''.join(speech_frames), dtype=np.int16)
                            audio_data = audio_data.astype(np.float32) / 32768.0

                            segments, info = model.transcribe(
                                audio_data,
                                beam_size=default_beam_size,
                                best_of=default_best_of,
                                temperature=default_temperature,
                                compression_ratio_threshold=default_compression_ratio_threshold,
                                log_prob_threshold=default_log_prob_threshold,
                                no_speech_threshold=default_no_speech_threshold,
                                condition_on_previous_text=default_condition_on_previous_text,
                                initial_prompt=default_initial_prompt,
                                word_timestamps=default_word_timestamps,
                                prepend_punctuations=default_prepend_punctuations,
                                append_punctuations=default_append_punctuations,
                                vad_filter=default_vad_filter
                            )
                            for segment in segments:
                                text = segment.text.strip()
                                if text:
                                    file.write(text + " ")
                                    file.flush()  # Flush the buffer to write immediately
                        speech_frames = []
                    frames = []
                    
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

async def main():
    global recording

    record_thread = threading.Thread(target=record_audio)
    transcribe_thread = threading.Thread(target=transcribe_audio)

    record_thread.start()
    transcribe_thread.start()

    input("Press Enter to stop recording...")
    recording = False

    record_thread.join()
    transcribe_thread.join()

    print("Recording stopped.")

    # Read the transcriptions from the file
    print("\nTranscriptions:")
    with open(output_file_path, "r") as file:
        transcript = file.read()

    # Generate summary using LLM
    summary = await generate_summary(transcript)
    print("\nSummary:")
    print(summary)

    # Write the summary to the file
    with open(output_file_path, "a") as file:
        file.write("\n\nSummary:\n")
        file.write(summary)

    # Write meta information to the file
    with open(output_file_path, "a") as file:
        file.write("\n\nMeta Information:\n")
        file.write(f"Duration: {abs(datetime.now() - current_datetime)}\n")
        file.write(f"Model Used: Mistral, Whisper\n")
        file.write(f"Number of Tokens: {len(transcript.split())}\n")

if __name__ == "__main__":
    asyncio.run(main())