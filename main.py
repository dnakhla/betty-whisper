import pyaudio
import numpy as np
import whisper
import webrtcvad
import queue
import threading
import warnings
warnings.filterwarnings("ignore")

# Set the sample rate and frame duration
sample_rate = 16000  # in Hz
frame_duration = 30  # in ms

# Set the VAD parameters
vad = webrtcvad.Vad()
vad_mode = 3  # Aggressive mode
min_speech_duration = 100  # in ms

# Create an instance of the Whisper model with default options
model = whisper.load_model("base")

# Default values for options
default_task = "transcribe"
default_beam_size = 5
default_best_of = 5
default_temperature = 0.0
default_compression_ratio_threshold = 2.4
default_logprob_threshold = -1.0
default_no_speech_threshold = 0.6

# Audio recording and transcription parameters
audio_queue = queue.Queue()
recording = True

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
                        transcription = model.transcribe(
                            audio_data,
                            task=default_task,
                            beam_size=default_beam_size,
                            best_of=default_best_of,
                            temperature=default_temperature,
                            compression_ratio_threshold=default_compression_ratio_threshold,
                            logprob_threshold=default_logprob_threshold,
                            no_speech_threshold=default_no_speech_threshold,
                        )
                        text = transcription["text"].strip()
                        if text:
                            print(text)
                    speech_frames = []
                frames = []

def main():
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

if __name__ == "__main__":
    main()