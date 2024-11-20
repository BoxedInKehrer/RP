import numpy as np
import random
import librosa
import os
import scipy.io.wavfile
from matplotlib import pyplot
from matplotlib.widgets import Button
import pyaudio
import wave
from glob import glob
from scipy.signal import chirp
import time
import threading

# ==================
# USER SETTINGS
# ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUND_NAME = "sweep"
CLASS_LABELS = ["metal", "desk"]
SAMPLES_PER_CLASS = 10
MODEL_NAME = "lndw2022_sweep_1s"
SHUFFLE_RECORDING_ORDER = False
APPEND_TO_EXISTING_FILES = True
# ==================

CHANNELS = 1
SR = 48000
CHUNK = 1024

AMPLIFICATION_FACTOR = 10**(47 / 20)  # Convert 47 dB to linear scale

#! the microphone has a range of 100Hz to 10kHz, so no point in going beyond that
SOUNDS = {
    "sweep": chirp(np.linspace(0, 1, SR), f0=100, f1=10000, t1=1, method='linear').astype('float32'),
    "white_noise": np.random.uniform(low=-0.999, high=1.0, size=(SR)).astype('float32'),
    "silence": np.zeros((SR,), dtype='float32'),
}

def main():
    print("Running for model '{}'".format(MODEL_NAME))
    print("Using sound: {}".format(SOUND_NAME))
    print("and classes: {}".format(CLASS_LABELS))

    global DATA_DIR
    DATA_DIR = mkpath(BASE_DIR, MODEL_NAME)

    setup_experiment()
    setup_audio(SOUND_NAME)
    setup_matplotlib()


def setup_experiment():
    global label_list, current_idx
    label_list = CLASS_LABELS * SAMPLES_PER_CLASS
    if SHUFFLE_RECORDING_ORDER:
        random.shuffle(label_list)
    current_idx = 0

    if APPEND_TO_EXISTING_FILES:
        wav_files = glob(os.path.join(DATA_DIR, "*.wav"))
        max_id = (
            max(
                [
                    int(os.path.basename(x).split("_")[0])
                    for x in wav_files
                ]
            )
            if wav_files
            else 0
        )
        label_list = [""] * max_id + label_list
        current_idx = max_id



def setup_audio(sound_name):
    global audio, stream, sound, Ains

    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    sound = SOUNDS[sound_name]

    # Ensure Ains matches the length of sound
    Ains = np.zeros_like(sound, dtype=np.float32)

    # Find the input and output device indices by matching criteria
    input_device_index = None
    output_device_index = None

    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        
        if (not input_device_index and "MAYA44USB" in device_info['name'] and "Ch12" in device_info['name'] and 
            device_info['maxInputChannels'] == 4):
            input_device_index = i
        if (not output_device_index and "MAYA44USB" in device_info['name'] and "Ch12" in device_info['name'] and 
            device_info['maxOutputChannels'] == 4):
            output_device_index = i

    if input_device_index is None or output_device_index is None:
        raise ValueError("Suitable input or output device not found")

    # print(f"Selected input device: {audio.get_device_info_by_index(input_device_index)['name']} (Index: {input_device_index})")
    # print(f"Selected output device: {audio.get_device_info_by_index(output_device_index)['name']} (Index: {output_device_index})")

    # Configuring input and output stream
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=SR,
        output=True,
        input=True,
        input_device_index=input_device_index,
        output_device_index=output_device_index
    )

    # Store active sound for reference
    sound_file = os.path.join(DATA_DIR, f"{0}_{sound_name}.wav")
    scipy.io.wavfile.write(sound_file, SR, sound)

def play_and_record():
    """
    Plays the sound and records simultaneously, synchronized to only capture the chirp duration.
    """
    global Ains

    def playback():
        stream.write(sound.tobytes())  # Play the chirp sound

    def recording():
        global Ains
        frames = []
        num_frames = int(len(sound) / CHUNK)  # Number of chunks matching the sound length
        for _ in range(num_frames):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.float32))
        Ains = np.hstack(frames)

        # Ensure Ains matches sound length exactly
        if len(Ains) > len(sound):
            Ains = Ains[:len(sound)]
        elif len(Ains) < len(sound):
            Ains = np.pad(Ains, (0, len(sound) - len(Ains)), 'constant')
   
    # Threads for simultaneous playback and recording
    play_thread = threading.Thread(target=playback)
    record_thread = threading.Thread(target=recording)

    play_thread.start()
    record_thread.start()

    # Wait for both threads to complete
    play_thread.join()
    record_thread.join()
    Ains *= AMPLIFICATION_FACTOR

def setup_matplotlib():
    global LINES, TITLE, b_rec
    fig, ax = pyplot.subplots(1)
    ax.set_ylim(-1, 1)  #? consider range
    pyplot.subplots_adjust(bottom=0.2)
    LINES, = ax.plot(Ains)
    ax_back = pyplot.axes([0.59, 0.05, 0.1, 0.075])
    b_back = Button(ax_back, '[B]ack')
    b_back.on_clicked(back)
    ax_rec = pyplot.axes([0.81, 0.05, 0.1, 0.075])
    b_rec = Button(ax_rec, '[R]ecord')
    b_rec.on_clicked(record)
    fig.canvas.mpl_connect('key_press_event', on_key)
    TITLE = ax.set_title(get_current_title())
    pyplot.show()


def on_key(event):
    if event.key == "r":
        record(event)
    elif event.key == "b":
        back(event)


def l(i):
    try:
        return label_list[i]
    except IndexError:
        return ""

def get_current_title():
    name = "Model: {}".format(MODEL_NAME.replace("_", " "))
    labels = "previous: {}   current: [{}]   next: {}".format(l(current_idx - 1), l(current_idx), l(current_idx + 1))
    number = "#{}/{}: {}".format(current_idx + 1, len(label_list), l(current_idx))
    if current_idx >= len(label_list):
        number += "DONE!"
    title = "{}\n{}\n{}".format(name, labels, number)
    return title

def back(event):
    global current_idx
    current_idx = max(0, current_idx - 1)
    update()


def record(event):
    global current_idx
    if current_idx >= len(label_list):
        return
    play_and_record()
    LINES.set_ydata(Ains)
    store()
    current_idx += 1
    update()


def store():
    sound_file = os.path.join(DATA_DIR, "{}_{}.wav".format(current_idx + 1, l(current_idx)))
    scipy.io.wavfile.write(sound_file, SR, Ains)


def mkpath(*args):
    path = os.path.join(*args)
    base_path = os.path.split(path)[0] if os.path.splitext(path)[1] else path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return path


def update():
    TITLE.set_text(get_current_title())
    pyplot.draw()


if __name__ == "__main__":
    main()
