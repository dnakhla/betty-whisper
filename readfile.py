import numpy as np
import whisper
import subprocess
import os
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore")

# Default values for options
DEFAULT_TASK = "transcribe"
DEFAULT_BEAM_SIZE = 5
DEFAULT_BEST_OF = 5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4
DEFAULT_LOGPROB_THRESHOLD = -1.0
DEFAULT_NO_SPEECH_THRESHOLD = 0.6

def extract_audio(video_file, temp_audio_file):
    command = f"ffmpeg -i {video_file} -vn -acodec pcm_s16le -ar 16000 -ac 1 {temp_audio_file}"
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error extracting audio: {e.output.decode('utf-8')}")

def transcribe_audio(model, audio_file):
    try:
        audio_segment = AudioSegment.from_wav(audio_file)
        duration = len(audio_segment)
        chunk_size = 10 * 1000  # 10 seconds in milliseconds
        transcriptions = []

        for i in range(0, duration, chunk_size):
            chunk = audio_segment[i:i+chunk_size]
            chunk.export("temp_chunk.wav", format="wav")
            audio_data = whisper.load_audio("temp_chunk.wav")
            result = model.transcribe(
                audio_data,
                task=DEFAULT_TASK,
                beam_size=DEFAULT_BEAM_SIZE,
                best_of=DEFAULT_BEST_OF,
                temperature=DEFAULT_TEMPERATURE,
                compression_ratio_threshold=DEFAULT_COMPRESSION_RATIO_THRESHOLD,
                logprob_threshold=DEFAULT_LOGPROB_THRESHOLD,
                no_speech_threshold=DEFAULT_NO_SPEECH_THRESHOLD,
            )
            transcriptions.append(result["text"].strip())

        os.remove("temp_chunk.wav")
        return " ".join(transcriptions)
    except Exception as e:
        raise RuntimeError(f"Error transcribing audio: {str(e)}")

def main():
    # Set the path to the video file
    video_file = "Recon_demo.mov"
    temp_audio_file = "temp_audio.wav"

    try:
        # Create an instance of the Whisper model with default options
        model = whisper.load_model("medium")

        # Extract the audio from the video
        print("Extracting audio...")
        extract_audio(video_file, temp_audio_file)

        # Check if the temporary audio file was created
        if not os.path.exists(temp_audio_file):
            raise FileNotFoundError(f"The temporary audio file {temp_audio_file} was not created.")

        # Transcribe the audio
        print("Transcribing...")
        transcription = transcribe_audio(model, temp_audio_file)

        # Print the transcription
        if transcription:
            print("Transcription:", transcription)
        else:
            print("No speech detected.")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except RuntimeError as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up the temporary audio files
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

if __name__ == "__main__":
    main()