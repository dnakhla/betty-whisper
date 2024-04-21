# audio_recording.py

import pyaudio
import webrtcvad
import numpy as np
from faster_whisper import WhisperModel
import wave

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
default_beam_size = 5
default_best_of = 5
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

def record_audio(is_recording, audio_buffer):
    chunk_size = int(sample_rate * frame_duration // 1000)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

    while is_recording():
        data = stream.read(chunk_size)
        audio_buffer.put(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

def transcribe_audio(is_recording, audio_buffer, output_file_path):
    frames = []
    speech_frames = []

    with open(output_file_path, "w") as file:
        file.write("Transcript:" + "\n")
        while is_recording() or not audio_buffer.empty():
            if not audio_buffer.empty():
                data = audio_buffer.get()
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

def save_audio(is_recording, audio_file_path):
    chunk_size = int(sample_rate * frame_duration // 1000)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

    frames = []
    while is_recording():
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(audio_file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()