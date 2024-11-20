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


# Measure frequency response of parametric filter in JackParameq.
#

# Create and connect objects.
#
F = JackParameq (1, "123", "JackParameq")
if F.get_state() < 0:
    print ("Failed to create JackSignal -- is Jack running ?")
    exit(1)

J = JackSignal("JackSignal")
J.create_output (0, "out-1")
J.create_input (0, "in-1")
J.silence()
J.connect_output (0, "JackParameq:in_0")
J.connect_input (0, "JackParameq:out_0")

# Get Jack info.
#
name, fsamp, period = J.get_jack_info ()

# Parameters.
#
impval = 1.0
fftlen = 4 * 1024

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
fig = plt.figure (figsize=(10,6), facecolor='white')
xx = [0.04, 0.54, 0.54]
yy = [0.54, 0.54, 0.04]

# Measurements.
#
F.set_bypass (False)
for i in range (3):
    ax = fig.add_axes ([xx[i], yy[i], 0.45, 0.45])
    ax.set_xlim (20, 20e3)
    ax.set_ylim (-25, 25)
    ax.set_xscale ('log')
    ax.text (30, 22, "Type %d" % (i + 1,))
    ax.grid ()

    print ("Type %d" % (i + 1,))
    # Loop over gain range.
    for g in range (-20,21,5):
        # Set section 'i' parameters.
        F.set_filter (i, 1e3, g, 0.5)
        # Wait for smooth change to end.
        if g == -20: sleep (0.5)
        else:        sleep (0.2)
        print ("  g = %5.1f" % (g,))
        # Measure.
        J.process()
        J.wait()
        # Plot result.
        Spec = np.fft.rfft (Ain1)
        ax.plot (Freq, 20 * np.log10 (np.abs (Spec) + 1e-10), color='b', lw=1)
    # Set section to bypass.    
    F.set_filter (i, 1e3, 0, 0.5)
        
fig.text (0.02, 0.40, "Type 1: Constant bandwidth resonance added or subtracted.")    
fig.text (0.02, 0.36, "Type 2: As mode 0, but adjusted for symmetry if gain is negative.")    
fig.text (0.02, 0.32, "Type 3: Bandwidth depends on gain, same as mode 1 at +/- 10 dB.")    
# Show result.
#    
plt.show()
