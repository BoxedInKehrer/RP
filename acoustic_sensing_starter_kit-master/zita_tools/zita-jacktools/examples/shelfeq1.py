#!/usr/bin/python

# ----------------------------------------------------------------------------
#
#  Copyright (C) 2013-2018 Fons Adriaensen <fons@linuxaudio.org>
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
from jacktools.jackparameq import JackParameq
from jacktools.jacksignal import JackSignal


# Measure low frequency shelf filter in JackParameq.
#

# Create and connect objects.
#
F = JackParameq (1, "L", "EQ")
if F.get_state() < 0:
    print ("Failed to create Jack client -- is Jack running ?")
    exit(1)

J = JackSignal("JackSignal")
J.create_output (0, "out-1")
J.create_input (0, "in-1")
J.silence()
J.connect_output (0, "EQ:in_0")
J.connect_input (0, "EQ:out_0")

# Get Jack info.
#
name, fsamp, period = J.get_jack_info ()

# Parameters.
#
fftlen = 16 * 1024

# Generate data.
#
Aout = np.zeros ([100,], dtype = np.float32)
Aout [0] = 1.0

Ain1 = np.empty ([fftlen,], dtype = np.float32)
Freq = np.linspace (0, fsamp / 2, num = fftlen // 2 + 1)

J.set_output_data (0, Aout)
J.set_input_data (0, Ain1, nskip = period)

# Set up plot.
#
fig = plt.figure (figsize=(10,6), facecolor='white')
xx = [0.04, 0.54, 0.04, 0.54]
yy = [0.54, 0.54, 0.04, 0.04]

freq = 100.0
F.set_bypass (False)

# Loop over gain range, shape = 0.
for i in range (3):
    ax = fig.add_axes ([xx[i], yy[i], 0.45, 0.45])
    ax.set_xlim (20, 20e3)
    ax.set_ylim (-25, 25)
    ax.set_xscale ('log')
    ax.grid ()
    s = 0.5 * i
    print ("s = %3.1f" % (s,))
    for g in range (-20,21,5):
        F.set_filter (0, freq,  g, s)
        # Wait for smooth change to end.
        if g == -20: sleep (0.5)
        else:        sleep (0.2)
        print ("   g = %5.1f" % (g,))
        # Measure.
        J.process()
        J.wait()
        # Plot result.
        Spec = np.fft.rfft (Ain1)
        if   g < 0: col = 'b'
        elif g > 0: col = 'r'
        else:       col = 'k'
        ax.plot (Freq, 20 * np.log10 (np.abs (Spec) + 1e-10), color = col, lw = 1)
    ax.text (500, 21, "freq = 100 Hz, shape = %3.1f" % (s,))
        
# Loop over shape range.
ax = fig.add_axes ([xx[3], yy[3], 0.45, 0.45])
ax.set_xlim (20, 20e3)
ax.set_ylim (-10, 25)
ax.set_xscale ('log')
ax.grid ()
g = 20.0
print ("g = %5.1f" % (g,))
for i in range (0, 11, 2):
    s = 0.1 * i
    F.set_filter (0, freq,  g, s)
    # Wait for smooth change to end.
    if i == 0: sleep (0.5)
    else:      sleep (0.2)
    print ("   s = %3.1f" % (s,))
    # Measure.
    J.process()
    J.wait()
    # Plot result.
    Spec = np.fft.rfft (Ain1)
    ax.plot (Freq, 20 * np.log10 (np.abs (Spec) + 1e-10), color= 'r', lw = 1)
ax.plot ([20, 20e3], [0, 0], color= 'k', lw = 1)
ax.text (125, 22, "freq = 100 Hz, gain = +20 dB, shape = 0...1")
        
# Show result.
#    
plt.show()
