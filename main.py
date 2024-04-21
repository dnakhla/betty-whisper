# main.py

import os
from datetime import datetime
import threading
import asyncio
from queue import Queue
from summary_generator import generate_summary
from audio_recording import record_audio, transcribe_audio

# Get the current date and time
current_datetime = datetime.now()

# Format the date and time as a string
datetime_string = current_datetime.strftime("%Y%m%d_%H%M%S")
output_file = f"transcriptions_{current_datetime.strftime('%m-%d-%Y__%A__%I:%M:%S%p')}.txt"

# Create the "transcriptions" directory if it doesn't exist
os.makedirs("transcriptions", exist_ok=True)

# Set the full path for the output file
output_file_path = os.path.join("transcriptions", output_file)

# Create a buffer queue for audio data
audio_buffer = Queue()

async def main():
    recording = True

    def stop_recording():
        nonlocal recording
        input("Press Enter to stop recording...")
        recording = False

    record_thread = threading.Thread(target=record_audio, args=(lambda: recording, audio_buffer))
    transcribe_thread = threading.Thread(target=transcribe_audio, args=(lambda: recording, audio_buffer, output_file_path))
    stop_thread = threading.Thread(target=stop_recording)

    record_thread.start()
    transcribe_thread.start()
    stop_thread.start()

    stop_thread.join()
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