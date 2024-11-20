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


# ----------- Oldfashioned frequency response using sweep and measure  -------------
#
# Make sure jnoisemeter is running, using
# input 1, and select the A, C or ITU filter.


# Generate a logarithmic sweep signal.
#
def logsweep (fmin, fmax, amp, dur, fsamp):

    # Create arrays for input, output and frequency.
    # Padding is added for the filtering in detect().
    len = int (dur * fsamp + 0.1)
    pad = int (0.1 * fsamp + 0.1)
    A = np.zeros ([len + 2 * pad], dtype = np.float32)
    B = np.zeros ([len + 2 * pad], dtype = np.float32)
    F = np.logspace (log10 (fmin), log10 (fmax), len)
    p = 0
    for i in range (len):
        A [i + pad] = amp * sin (2 * pi * p)
        p += F [i] / fsamp
        if p > 1.0: p -= 1.0
    return (A, B, F, pad)    


# Detect RMS and convert to dB.
#
def detect (A, fsamp, tfilt):

    # Two times square for RMS.
    R = 2 * A * A
    # Filter the result. We use a first order
    # lowpass filter, first in forward direction
    # and then in reverse. The result is a linear
    # phase 2nd order filter (i.e. with no delay).
    n = A.shape [0]
    w = 1.0 / (tfilt * fsamp)
    p = 0
    for i in range (n):
        p += w * (R [i] - p)
        R [i] = p
    p = 0    
    for i in range (n):
        j = n - 1 - i
        p += w * (R [j] - p)
        R [j] = 10 * log10 (p + 1e-60)
    return R    


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

# Set parameters.
#
# The amount of detail in the FR depends on the ratio of
# frequency range to sweep time, and on the time constant
# of the detector filter. Smaller values for the latter
# will produce ripples in the LF range.
#
fmin = 15.0
fmax = 22e3
level = 1.0
tsweep = 15.0
tfilt = 0.02

# Generate data.
#
print ("Preparing...")
Aout, Ainp, Freq, pad = logsweep (fmin, fmax, level, tsweep, fsamp)
J.set_output_data (0, Aout)
J.set_input_data (0, Ainp, nskip = period) # Skip one period.

# Run the test
#
print ("Measuring...")
J.process()
J.wait()

# Process the result.
#
print ("Processing...")
Resp = detect (Ainp, fsamp, tfilt)

# Display the FR.
#
print ("Displaying...")
fig = plt.figure (figsize=(8,6), facecolor='white')
ax = fig.add_axes ([0.05, 0.05, 0.9, 0.9])
ax.set_xlim (2e1, 2e4)
ax.set_ylim (-40, 20)
ax.set_xscale ('log')
# Remove padding, and decimate for faster display.
n = int (Freq.shape [0] / 5000)
ax.plot (Freq [0:-1:n], Resp [pad:-pad:n], color='b', lw=1)
ax.grid ()
plt.show()

