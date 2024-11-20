#!/usr/bin/python

# ----------------------------------------------------------------------------
#
#  Copyright (C) 2008-2015 Fons Adriaensen <fons@linuxaudio.org>
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
from math import *
from time import sleep
from jacktools.jacksignal import JackSignal
from jacktools.jackpeaklim import JackPeaklim
import matplotlib.pyplot as plt

# ------------------------------ Test JackPeaklim ----------------------------
#
# We test the response to
#
#  - a single sample,
#  - a 5 kHz burst,
#  - a 100 Hz burst,
#
# all with a peak amplitude of +6 dB, on top of a
# continuous 1 kHz signal with amplitude -6 dB.
# Zoom in on the plot to see the lookahead at work.
# Note automatically increased release time when
# limiting low frequency signals.

# Create a JackSignal object.
#
J = JackSignal("JackSignal")
if J.get_state() < 0:
    print ("Failed to create JackSignal -- is Jack running ?")
    exit(1)

L = JackPeaklim (1, "Limiter")
#L.set_release (0.001)    


# Create ports and connect.
#
J.create_output (0, "out-1")
J.create_input (0, "in-1")
J.silence()
J.connect_output (0, "Limiter:in_0")
J.connect_input (0, "Limiter:out_0")

# Get Jack info.
#
name, fsamp, period = J.get_jack_info ()

# Create input and output buffers, 0.3 seconds.
#
L1 = int (0.30 * fsamp + 0.1)
L2 = int (0.04 * fsamp + 0.1)
Aout = np.empty(L1, dtype = np.float32)
Ainp = np.empty(L1, dtype = np.float32)

# Generate output signal.
#

# 1 kHz sine wave at -6 dB during entire buffer.
# Used to see the actual gain. 
#
w = 1e3 * 2 * pi / fsamp
for i in range (L1):
    Aout [i] = 0.5 * sin (w * i)

amp = 2.0 # peak amplitude

# Add singe sample.
Aout [1000] = amp

# Add 5 Khz burst.
w = 5e3 * 2 * pi / fsamp
for i in range (L2):
    Aout [i + 4000] += amp * sin (w * i)

# Add 100 Hz burst.
w = 1e2 * 2 * pi / fsamp
for i in range (L2):
    Aout [i + 9000] += amp * sin (w * i)


# Assign buffers to ports. We skip the first period
# on input to compensate for the loop, and 64 samples
# to compensate for zita-dpl1 latency. The single
# sample peak should end up at sample 1000 in the
# captured signal (assuming Fs = 48 kHz).
#
J.set_output_data (0, Aout)
J.set_input_data (0, Ainp, nskip = period + 64)

# Run the test.
#
J.process()
J.wait()
del J
del L


# Display the result.
#
fig = plt.figure (figsize=(9,6), facecolor='white')
ax1 = fig.add_axes ([0.05, 0.05, 0.9, 0.42])
ax1.set_ylim (-1.2, 1.2)
ax1.plot (Ainp, color='b', lw=1)
ax1.grid ()
ax2 = fig.add_axes ([0.05, 0.55, 0.9, 0.42])
ax2.set_ylim (-3.0, 3.0)
ax2.plot (Aout, color='r', lw=1)
ax2.grid ()
plt.show()
