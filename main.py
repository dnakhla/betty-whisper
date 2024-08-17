# main.py

import os
from datetime import datetime
import asyncio
from audio_recording import AudioProcessor
from summary_generator import generate_summary

async def main():
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as a string
    output_file = f"t_{current_datetime.strftime('%m-%d-%Y__%A__%I:%M:%S%p')}.txt"
    audio_file = f"audio_{current_datetime.strftime('%m-%d-%Y__%A__%I:%M:%S%p')}.wav"

    # Create the "transcriptions" directory if it doesn't exist
    os.makedirs("transcriptions", exist_ok=True)

    # Set the full path for the output file and audio file
    output_file_path = os.path.join("transcriptions", output_file)
    audio_file_path = os.path.join("transcriptions", audio_file)

    # Create an AudioProcessor instance
    processor = AudioProcessor(output_file_path, audio_file_path)

    print("Recording started. Press Enter to stop recording...")
    processor.start_recording()

    # Wait for user input to stop recording
    await asyncio.get_event_loop().run_in_executor(None, input)

    print("\nStopping recording...")
    processor.stop_recording()

    print("Recording stopped.")

    # Read the transcriptions from the file
    print("\nTranscriptions:")
    with open(output_file_path, "r") as file:
        transcript = file.read()
    print(transcript)

    # Generate summary using LLM
    summary = await generate_summary(transcript)
    print("\nSummary:")
    print(summary)

    # Write the summary to the file
    with open(output_file_path, "a") as file:
        file.write(f"\n\nSummary:\n {summary}")

    # Write meta information to the file
    with open(output_file_path, "a") as file:
        file.write("\n\nMeta Information:\n")
        file.write(f"Duration: {abs(datetime.now() - current_datetime)}\n")
        file.write(f"Model Used: Mistral, Whisper\n")
        file.write(f"Number of Tokens: {len(transcript.split())}\n")

if __name__ == "__main__":
    asyncio.run(main())