#!/usr/bin/python

# ----------------------------------------------------------------------------
#
#  Copyright (C) 2013-2015 Fons Adriaensen <fons@linuxaudio.org>
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


import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from utils.sinewave import *
from math import hypot, log10
from jacktools.jacksignal import JackSignal


# ------------------ Harmonic distortion measurement ------------------------


# Test levels (dB) 
#
Levels = [ -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0 ]

# Test frequency (Hz) 
#
Freq = 1003

# Time for each test (s)
#
Mtime = 1.0



# Create a Jacksignal object.
#
J = JackSignal("IMtest")
if J.get_state() < 0:
    print ("Failed to create JackSignal -- is the server running ?")
    exit(1)

# Get Jack info.
#
name, Fsamp, period = J.get_jack_info()

# Create ports and connect
#
J.create_output (0, "out")
J.create_input (0, "in")
J.silence()
J.connect_output (0, "jack_nonlin:in")
J.connect_input (0, "jack_nonlin:out")
#J.connect_output (0, "system:playback_1")
#J.connect_input (0, "system:capture_1")

# Length of test signals, half a second.
#
siglen = int (Mtime * Fsamp + 0.5)
margin = 1000
buflen = siglen + margin

# Generate test signal, will be scaled for output.
# Cast from double to float for Jacksignal.
#
Aout = gen_sinewave (1.00, Freq, Fsamp, buflen).astype (np.float32)

# Create input buffer.
#
Ainp = np.zeros ((buflen,), dtype = np.float32)

# Generate reference signals for selective measurement.
#
M1 = gen_complex (Freq, Fsamp, siglen)
M2 = gen_complex (2 * Freq, Fsamp, siglen)
M3 = gen_complex (3 * Freq, Fsamp, siglen)


# Run test.
#
N = len (Levels)
HD2 = [ 0 for i in range (N) ]
HD3 = [ 0 for i in range (N) ]

for i in range (N):

    # Define signal buffers and run test. 
    amp = pow (10.0, Levels [i] / 20.0) 
    J.set_output_data (0, amp * Aout)
    J.set_input_data (0, Ainp) 
    J.process()
    J.wait()

    # Skip margin samples and measure.
    # We are not interested in the phase here.
    #
    T = Ainp [margin:]
    L1,p = sigdetect (T, M1)
    L2,p = sigdetect (T, M2)
    L3,p = sigdetect (T, M3)

    # Print results.
    #
    L1dB = 20 * log10 (L1)
    HD2 [i] = hd2 = 100 * L2 / L1
    HD3 [i] = hd3 = 100 * L3 / L1
    print ("Level = %5.1f    A1 = %5.1f   hd2 = %6.3f%%  hd3 = %6.3f%%" % (Levels [i], L1dB, hd2, hd3))


# Create a nice graph to present the results.
#
Levels = np.array (Levels) # Allows to add offsets.
fig = plt.figure (figsize=(8,5), facecolor='white')
ax = fig.add_axes ([0.07, 0.05, 0.86, 0.90])
ax.set_title ("%% Harmonic distortion, %3.1f Hz"  % (Freq,))
ax.set_xlim (-55, 5)
ax.set_ylim (1e-3, 1e1)
ax.set_yscale ('log')
ax.bar (Levels - 0.9, HD2, 0.7, color = 'orange')
ax.bar (Levels + 0.1, HD3, 0.7, color = 'red')
ax.text (-51, 5.0, 'HD2', size = 17, color = 'orange')
ax.text (-46, 5.0, 'HD3', size = 17, color = 'red')
ax.grid ()
plt.show ()
    
