#!/usr/bin/python

# ----------------------------------------------------------------------------
#
#  Copyright (C) 2008-2014 Fons Adriaensen <fons@linuxaudio.org>
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
from jacktools.jacksignal import JackSignal


# ------- Frequency response measurement using impulse and FFT  ---------
#
# Make sure jnoisemeter is running, using
# input 1, and select the A, C or ITU filter.

# Create a JackSignal object and connect.
#
J = JackSignal("JackSignal")
if J.get_state() < 0:
    print ("Failed to create JackSignal -- is Jack running ?")
    exit(1)
J.create_output (0, "out-1")
J.create_input (0, "in-1")

J.silence()
J.connect_output (0, "jnoisemeter:in_1")
J.connect_input (0, "jnoisemeter:out")

# Get Jack info.
#
name, fsamp, period = J.get_jack_info ()

# Parameters
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
J.set_input_data (0, Ain1, nskip = period) # Skip one period.

# Run the test
#
J.process()
J.wait ()

# Process the result
#
Spec = np.fft.rfft (Ain1)

# Display impulse and magnitude response.
#
fig = plt.figure (figsize=(8,6), facecolor='white')
ax1 = fig.add_axes ([0.05, 0.04, 0.90, 0.44])
ax1.set_ylim (-1.5, 1.5)
ax1.plot (Ain1, color='b', lw=1)
ax1.grid ()
ax2 = fig.add_axes ([0.05, 0.54, 0.90, 0.44])
ax2.set_xlim (1e1, 24e3)
ax2.set_ylim (-60, 15)
ax2.set_xscale ('log')
ax2.plot (Freq, 20 * np.log10 (np.abs (Spec) + 1e-10), color='b', lw=1)
ax2.grid ()
plt.show()
