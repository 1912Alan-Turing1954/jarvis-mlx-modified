import pyaudio
import wave
import numpy as np
import time
import torch, torchaudio
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import os
import warnings

warnings.filterwarnings("ignore")


def record_audio(output_filename="recorded_output.wav", start_threshold=3500, silence_threshold=2500, silence_duration=1.5, warmup_samples=5):
    """Record audio from the microphone until silence is detected.

    Args:
        output_filename (str): Name of the output WAV file.
        start_threshold (int): Volume level to start recording.
        silence_threshold (int): Volume level to consider as silence.
        silence_duration (int): Duration in seconds to consider as silence.
        warmup_samples (int): Number of initial samples to ignore.
    """
    
    # Set up audio recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 4096

    audio = pyaudio.PyAudio()
    
    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Listening... Speak to start recording.")

    frames = []
    recording = False
    silence_start_time = None
    warmup_count = 0  # Counter for warm-up period

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume_level = np.abs(audio_data).mean()

        # Ignore initial readings during warm-up
        if warmup_count < warmup_samples:
            warmup_count += 1
            continue

        # Print the current volume level
        print(f"Current volume level: {volume_level}")

        if volume_level >= start_threshold:  # Start recording at a lower threshold
            if not recording:
                print("Recording started.")
                recording = True
            frames.append(data)  # Append audio data while recording
            silence_start_time = None  # Reset silence timer
        elif recording:  # Only check silence if currently recording
            frames.append(data)  # Continue appending audio data
            if volume_level < silence_threshold:
                if silence_start_time is None:
                    silence_start_time = time.time()  # Start silence timer
                elif time.time() - silence_start_time > silence_duration:
                    print(f"Silence detected for more than {silence_duration} seconds. Stopping recording.")
                    break
            else:
                silence_start_time = None  # Reset silence timer if volume is above threshold

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("Finished recording and saved to " + output_filename)
    return output_filename


def transcribe_audio(filename):
    """Transcribe audio using Wav2Vec 2.0."""
    
    # Load the pre-trained model and tokenizer
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("wav2vec2")
    model = Wav2Vec2ForCTC.from_pretrained("wav2vec2")

    # Load audio file
    waveform, sample_rate = torchaudio.load(filename)
    waveform = waveform.squeeze().numpy()

    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Prepare input for the model
    input_values = tokenizer(waveform, return_tensors="pt").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])

    os.remove(filename)

    return transcription

