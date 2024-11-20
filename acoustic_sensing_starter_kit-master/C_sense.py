import numpy as np
import os
import pickle
import librosa
import pyaudio
import threading
from matplotlib import pyplot
from matplotlib.widgets import Button
from scipy.signal import chirp

# Settings
SR = 48000
CHUNK = 1024
MODEL_PATH = r"C:\Users\Kehre\OneDrive\Desktop\WS25\RP\acoustic_sensing_starter_kit-master\lndw2022_sweep_1s\sensor_model.pkl"
AMPLIFICATION_FACTOR = 10**(47 / 20)  # Convert 47 dB to linear scale
SWEEP_SOUND = chirp(np.linspace(0, 1, SR), f0=100, f1=10000, t1=1, method='linear').astype('float32')

# Load model
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# Initialize audio
audio = pyaudio.PyAudio()

def setup_audio():
    """Find and set up input/output devices."""
    input_device_index = None
    output_device_index = None

    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        
        # Replace the conditions below with your specific device criteria
        if (not input_device_index and "MAYA44USB" in device_info['name'] and "Ch12" in device_info['name'] and 
            device_info['maxInputChannels'] == 4):
            input_device_index = i
        if (not output_device_index and "MAYA44USB" in device_info['name'] and "Ch12" in device_info['name'] and 
            device_info['maxOutputChannels'] == 4):
            output_device_index = i

    if input_device_index is None or output_device_index is None:
        raise ValueError("Suitable input or output device not found")

    return input_device_index, output_device_index

input_device_index, output_device_index = setup_audio()

stream = audio.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=SR,
    output=True,
    input=True,
    input_device_index=input_device_index,
    output_device_index=output_device_index
)

# Global variable for recorded data visualization
recorded_signal = np.zeros_like(SWEEP_SOUND)

def play_and_record():
    """Play sweep sound and record response with synchronization."""
    def playback():
        stream.write(SWEEP_SOUND.tobytes())

    def record():
        global recorded_signal
        frames = []
        for _ in range(int(len(SWEEP_SOUND) / CHUNK)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.float32))
        recorded_signal = np.hstack(frames)
        recorded_signal *= AMPLIFICATION_FACTOR  # Amplify the recorded signal

        # Ensure recorded signal matches the length of SWEEP_SOUND
        if len(recorded_signal) > len(SWEEP_SOUND):
            recorded_signal = recorded_signal[:len(SWEEP_SOUND)]
        elif len(recorded_signal) < len(SWEEP_SOUND):
            recorded_signal = np.pad(recorded_signal, (0, len(SWEEP_SOUND) - len(recorded_signal)), 'constant')

    # Start threads for playback and recording
    play_thread = threading.Thread(target=playback)
    record_thread = threading.Thread(target=record)

    play_thread.start()
    record_thread.start()
    play_thread.join()
    record_thread.join()

    return recorded_signal


def sound_to_spectrum(sound):
    """Convert sound to frequency spectrum."""
    spectrogram = np.abs(librosa.stft(sound, n_fft=CHUNK))
    return spectrogram.sum(axis=1)

def predict():
    """Record response, predict class, and visualize recorded signal."""
    response = play_and_record()
    spectrum = sound_to_spectrum(response).reshape(1, -1)

    # Update visualization
    line.set_ydata(response)
    ax.set_ylim(np.min(response) * 1.1, np.max(response) * 1.1)
    fig.canvas.draw_idle()

    # Prediction
    probabilities = clf.predict_proba(spectrum)[0]
    classes = clf.classes_
    prediction = clf.predict(spectrum)[0]

    print(f"Prediction: {prediction}")
    print("Class Probabilities:")
    for cls, prob in zip(classes, probabilities):
        print(f"  {cls}: {prob:.2f}")

# Setup Matplotlib UI
fig, ax = pyplot.subplots()
ax.set_title("Recorded Signal")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")
line, = ax.plot(np.zeros_like(SWEEP_SOUND))
ax_button = pyplot.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, "Predict")

def on_button_clicked(event):
    predict()

button.on_clicked(on_button_clicked)
pyplot.show()

# Cleanup on exit
stream.stop_stream()
stream.close()
audio.terminate()
