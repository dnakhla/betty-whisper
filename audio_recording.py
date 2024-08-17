import pyaudio
import webrtcvad
import numpy as np
from faster_whisper import WhisperModel
import wave
import queue
import threading
import time
import os

# Set the sample rate and frame duration
SAMPLE_RATE = 16000  # in Hz
FRAME_DURATION = 30  # in ms

# Set the VAD parameters
VAD = webrtcvad.Vad(3)  # Aggressive mode
MIN_SPEECH_DURATION = 250  # in ms

# Create an instance of the Whisper model with default options
MODEL = WhisperModel("base.en", device="cpu", compute_type="int8")

# Whisper transcription parameters
TRANSCRIPTION_PARAMS = {
    "beam_size": 10,
    "best_of": 10,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": True,
    "word_timestamps": True,
    "prepend_punctuations": "\"'¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：)]};、",
    "vad_filter": True
}

class AudioProcessor:
    def __init__(self, output_file_path, audio_file_path):
        self.is_recording = False
        self.audio_buffer = queue.Queue()
        self.output_file_path = output_file_path
        self.audio_file_path = audio_file_path

    def start_recording(self):
        self.is_recording = True
        self.record_thread = threading.Thread(target=self.record_audio)
        self.transcribe_thread = threading.Thread(target=self.transcribe_audio)
        self.save_thread = threading.Thread(target=self.save_audio)

        self.record_thread.start()
        self.transcribe_thread.start()
        self.save_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.record_thread.join()
        self.transcribe_thread.join()
        self.save_thread.join()

    def record_audio(self):
        chunk_size = int(SAMPLE_RATE * FRAME_DURATION // 1000)
        audio = pyaudio.PyAudio()
        stream = None

        while self.is_recording:
            try:
                if stream is None:
                    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=chunk_size)
                data = stream.read(chunk_size)
                self.audio_buffer.put(data)
            except OSError as e:
                print(f"Error occurred during audio recording: {str(e)}")
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
                stream = None
                time.sleep(1)  # Wait before trying to reopen the stream

        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()

    def transcribe_audio(self):
        frames = []
        speech_frames = []

        with open(self.output_file_path, "w") as file:
            file.write("Transcript:\n")
            while self.is_recording or not self.audio_buffer.empty():
                if not self.audio_buffer.empty():
                    data = self.audio_buffer.get()
                    frames.append(data)
                    is_speech = VAD.is_speech(data, SAMPLE_RATE)

                    if is_speech:
                        speech_frames.append(data)
                    else:
                        if speech_frames:
                            speech_duration = len(speech_frames) * FRAME_DURATION
                            if speech_duration >= MIN_SPEECH_DURATION:
                                audio_data = np.frombuffer(b''.join(speech_frames), dtype=np.int16)
                                audio_data = audio_data.astype(np.float32) / 32768.0

                                segments, _ = MODEL.transcribe(audio_data, **TRANSCRIPTION_PARAMS)
                                for segment in segments:
                                    text = segment.text.strip()
                                    if text:
                                        file.write(text + " ")
                                        file.flush()
                            speech_frames = []
                        frames = []

    def save_audio(self):
        chunk_size = int(SAMPLE_RATE * FRAME_DURATION // 1000)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=chunk_size)

        frames = []
        while self.is_recording:
            data = stream.read(chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(self.audio_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))

def main():
    output_file_path = "transcript.txt"
    audio_file_path = "recorded_audio.wav"
    
    processor = AudioProcessor(output_file_path, audio_file_path)
    
    print("Recording started. Press Ctrl+C to stop.")
    processor.start_recording()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        processor.stop_recording()

if __name__ == "__main__":
    main()