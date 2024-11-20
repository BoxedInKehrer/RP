#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for the "Acoustic Sensing Starter Kit"
[Zöller, Gabriel, Vincent Wall, and Oliver Brock. “Active Acoustic Contact Sensing for Soft Pneumatic Actuators.” In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.]

This script _trains_ a KNN classifier to predict the previously recorded data samples.

In 'USER SETTINGS' define:
BASE_DIR - path where data is read from
MODEL_NAME - name of the sensor model. used as folder name.
TEST_SIZE - ratio of samples left out for testing. leave at '0' to use all samples for training.

@author: Vincent Wall, Gabriel Zöller
@copyright 2020 Robotics and Biology Lab, TU Berlin
@licence: BSD Licence
"""

import numpy
import librosa
import os
import pandas
import pickle

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from A_record import MODEL_NAME, BASE_DIR, SOUND_NAME

# ==================
# USER SETTINGS
# ==================
SENSORMODEL_FILENAME = "sensor_model.pkl"
TEST_SIZE = 0.2  # percentage of samples left out of training and used for reporting test score
SHOW_PLOTS = True
# ==================

SR = 48000
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
CHUNK = 1024

def get_num_and_label(filename):
    try:
        # remove file extension
        name = os.path.splitext(filename)[0]
        # remove initial number
        name = name.split("_")
        num = int(name[0])
        label = "_".join(name[1:])
        return num, label
    except ValueError:
        # filename with different formatting. ignore.
        return -1, None


def load_sounds(path):
    """Load soundfiles from disk"""
    filenames = sorted(os.listdir(path))
    sounds = []
    labels = []
    for fn in filenames:
        n, label = get_num_and_label(fn)
        if n < 0:
            # filename with different formatting. ignore.
            continue
        elif n == 0:
            # zero index contains active sound
            global SOUND_NAME
            SOUND_NAME = label
        else:
            sound = librosa.load(os.path.join(path, fn), sr=SR)[0]
            sounds.append(sound)
            labels.append(label)
    print(f"Loaded **{len(sounds)}** sounds with \nlabels: {sorted(set(labels))}")
    return sounds, labels


def sound_to_spectrum(sound):
    """Convert sounds to frequency spectra"""
    spectrum = numpy.fft.rfft(sound)
    amplitude_spectrum = numpy.abs(spectrum)
    d = 1.0/SR
    freqs = numpy.fft.rfftfreq(len(sound), d)
    index = pandas.Index(freqs)
    series = pandas.Series(amplitude_spectrum, index=index)
    return series

def sound_to_spectrum_stft(sound, n_fft=CHUNK, in_dB=False):
    spectrogram = numpy.abs(librosa.stft(sound, n_fft=n_fft))
    spectrum = spectrogram.sum(axis=1)
    if in_dB:
        # convert to decibel scale
        spectrum = librosa.amplitude_to_db(spectrum, ref=numpy.max)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    index = pandas.Index(freqs)
    series = pandas.Series(spectrum, index=index)
    return series


def save_sensor_model(path, clf, filename):
    """Saves sensor model to disk"""
    with open(os.path.join(path,filename), 'wb') as f:
        pickle.dump(clf, f, protocol=PICKLE_PROTOCOL)


def plot_spectra(spectra, labels):
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(1)
    color_list = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
    cdict = dict(zip(sorted(list(set(labels))), color_list))
    for i, (s, l) in enumerate(zip(spectra, labels)):
        ax.plot(s, c=cdict[l])

    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color=col, lw=4) for col in cdict.values()]
    legend_labels = list(cdict.keys())
    ax.legend(legend_lines, legend_labels)

    fig.show()


def main():
    print("Running for model '{}'".format(MODEL_NAME))
    global DATA_DIR
    DATA_DIR = os.path.join(BASE_DIR, MODEL_NAME)

    sounds, labels = load_sounds(DATA_DIR)
    spectra = [sound_to_spectrum_stft(sound) for sound in sounds]

    if SHOW_PLOTS:
        plot_spectra(spectra, labels)

    X_train, X_test, y_train, y_test = train_test_split(spectra, labels, test_size=TEST_SIZE)

    classifiers = {
        "KNN": KNeighborsClassifier(),
        # "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1337),
        "SVM": SVC(kernel='linear', C=1.0, probability=True, random_state=1337),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=1337),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=1337),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=1337),
        "Naive Bayes": GaussianNB(),
    }

    results = []
    best_model = None
    best_score = 0

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        results.append({
            "Model": name,
            "Train Accuracy": train_score,
            "Test Accuracy": test_score
        })

        if test_score > best_score:
            best_score = test_score
            best_model = (name, clf)

        print(f"{name} - Train Accuracy: {train_score:.2f}, Test Accuracy: {test_score:.2f}")

    # Save the best model
    if best_model:
        best_model_name, best_model_instance = best_model
        save_sensor_model(DATA_DIR, best_model_instance, SENSORMODEL_FILENAME)
        print(f"\nBest model '{best_model_name}' saved to '{os.path.join(DATA_DIR, SENSORMODEL_FILENAME)}' with Test Accuracy: {best_score:.2f}")

    # Display comparison
    import pandas as pd
    results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
    print("\nModel Performance Comparison:")
    print(results_df)

    if SHOW_PLOTS:
        pyplot.pause(0.1)
        pyplot.show()


if __name__ == "__main__":
    main()
