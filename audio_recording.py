import pyaudio
import webrtcvad
import numpy as np
from faster_whisper import WhisperModel
import wave
import queue
import threading
import time

# Set the sample rate and frame duration
SAMPLE_RATE = 16000  # in Hz (webrtcvad supports 8000, 16000, 32000, or 48000)
FRAME_DURATION = 30  # in ms

# Set the VAD parameters
VAD = webrtcvad.Vad(3)  # Aggressive mode
MIN_SPEECH_DURATION = 250  # in ms

# Create an instance of the Whisper model with default options
MODEL = WhisperModel("base.en", device="cpu", compute_type="int8")

# Whisper transcription parameters
TRANSCRIPTION_PARAMS = {
    "beam_size": 5,
    "best_of": 5,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "no_speech_threshold": 0.3,
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
        self.frames = []  # To store audio data for saving to file

    def start_recording(self):
        self.is_recording = True
        self.record_thread = threading.Thread(target=self.record_audio)
        self.transcribe_thread = threading.Thread(target=self.transcribe_audio)

        self.record_thread.start()
        self.transcribe_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.record_thread.join()
        self.transcribe_thread.join()
        self.save_audio()

    def record_audio(self):
        chunk_size = int(SAMPLE_RATE * FRAME_DURATION // 1000)
        audio = pyaudio.PyAudio()
        stream = None

        try:
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=chunk_size)
            while self.is_recording:
                data = stream.read(chunk_size, exception_on_overflow=False)
                self.audio_buffer.put(data)
                self.frames.append(data)
        except Exception as e:
            print(f"Error occurred during audio recording: {str(e)}")
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()

    def transcribe_audio(self):
        speech_frames = []
        num_padding_frames = int(MIN_SPEECH_DURATION / FRAME_DURATION)
        padding_frames = [b''] * num_padding_frames

        with open(self.output_file_path, "w") as file:
            file.write("Transcript:\n")
            while self.is_recording or not self.audio_buffer.empty():
                if not self.audio_buffer.empty():
                    data = self.audio_buffer.get()
                    is_speech = VAD.is_speech(data, SAMPLE_RATE)

                    if is_speech:
                        speech_frames.append(data)
                    else:
                        if len(speech_frames) > 0:
                            # Add padding frames
                            speech_frames.extend(padding_frames)
                            audio_data = b''.join(speech_frames)
                            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                            segments, _ = MODEL.transcribe(audio_np, **TRANSCRIPTION_PARAMS)
                            for segment in segments:
                                text = segment.text.strip()
                                if text:
                                    file.write(text + " ")
                                    file.flush()
                            speech_frames = []
                else:
                    time.sleep(0.01)  # Sleep briefly to avoid busy waiting

    def save_audio(self):
        with wave.open(self.audio_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for paInt16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))