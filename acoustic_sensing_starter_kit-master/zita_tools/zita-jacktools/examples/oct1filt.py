#!/usr/bin/python

# ----------------------------------------------------------------------------
#
#  Copyright (C) 2012-2018 Fons Adriaensen <fons@linuxaudio.org>
#    
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http:#www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from math import *
from time import sleep
from jacktools.jackiecfilt import JackIECfilt
from jacktools.jacksignal import JackSignal


# Measure frequency response of octave band filters in JackIECfilt.
#

# Create and connect objects.
#
J = JackSignal("JackSignal")
if J.get_state() < 0:
    print ("Failed to create JackSignal -- is Jack running ?")
    exit(1)
J.create_output (0, "out")
J.create_input (0, "in")
J.silence()

F = JackIECfilt (1, 1, "JackIECfilt")

J.connect_output (0, "JackIECfilt:in_0")
J.connect_input (0, "JackIECfilt:out_0")

# Get Jack info.
#
name, fsamp, period = J.get_jack_info ()

# Parameters.
#
impval = 1.0
fftlen = 64 * 1024

# Generate data.
#
Aout = np.zeros ([100,], dtype = np.float32)
Aout [0] = impval

Ain1 = np.empty ([fftlen,], dtype = np.float32)
Freq = np.linspace (0, fsamp / 2, num = fftlen // 2 + 1)

J.set_output_data (0, Aout)
J.set_input_data (0, Ain1, nskip = period)

# Set up plot.
#
fig = plt.figure (figsize=(8,5), facecolor='white')
ax1 = fig.add_axes ([0.05, 0.04, 0.90, 0.90])
ax1.set_xlim (1e1, 24e3)
ax1.set_ylim (-50, 5)
ax1.set_xscale ('log')
ax1.grid ()

# Measure and plot each band.
#
for i in range (10):
    print ("measuring band", i)
    F.set_filter (0, 0, 1, i)
    J.process()
    J.wait()
    Spec = np.fft.rfft (Ain1)
    ax1.plot (Freq, 20 * np.log10 (np.abs (Spec) + 1e-10), color='b', lw=1)

# Show result.
#    
plt.show()
