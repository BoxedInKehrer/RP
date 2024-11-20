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


# ---------------- SMPTE/DIN Intermodulation measurement --------------------


# SMPTE frequencies
#
#Flo = 60
#Fhi = 7000

# DIN frequencies
#
Flo = 250
Fhi = 8000


# Test levels (dB, level of low frequency) 
#
Levels = [ -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0 ]
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

# Length of test signals in samples.
#
siglen = int (Mtime * Fsamp + 0.5)
margin = 1000
buflen = siglen + margin

# Generate test signal, will be scaled for output.
#
Aout = (  gen_sinewave (1.00, Flo, Fsamp, buflen)
        + gen_sinewave (0.25, Fhi, Fsamp, buflen)).astype (np.float32)

# Input buffer.
#
Ainp = np.zeros ((buflen,), dtype = np.float32)

# Generate reference signals for selective measurement.
#
Mlo = gen_complex (Flo, Fsamp, siglen)
Mhi = gen_complex (Fhi, Fsamp, siglen)
M3a = gen_complex (Fhi - 2 * Flo, Fsamp, siglen)
M2a = gen_complex (Fhi - 1 * Flo, Fsamp, siglen)
M2b = gen_complex (Fhi + 1 * Flo, Fsamp, siglen)
M3b = gen_complex (Fhi + 2 * Flo, Fsamp, siglen)


# Run test.
#
N = len (Levels)
D2 = [ 0 for i in range (N) ]
D3 = [ 0 for i in range (N) ]

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
    Llo,p = sigdetect (T, Mlo)
    Lhi,p = sigdetect (T, Mhi)
    L3a,p = sigdetect (T, M3a)
    L2a,p = sigdetect (T, M2a)
    L2b,p = sigdetect (T, M2b)
    L3b,p = sigdetect (T, M3b)

    # Print results.
    #
    dblo = 20 * log10 (Llo + 1e-30)
    dbhi = 20 * log10 (Lhi + 1e-30)
    D2 [i] = im2 = 100 * hypot (L2a, L2b) / Lhi
    D3 [i] = im3 = 100 * hypot (L3b, L3b) / Lhi
    print ("Level = %5.1f    Al = %5.1f  Ah = %5.1f   im2 = %6.3f%%  im3 = %6.3f%%" % (Levels [i], dblo, dbhi, im2, im3))


# Create a nice graph to present the results.
#
Levels = np.array (Levels)
fig = plt.figure (figsize=(8,5), facecolor='white')
ax = fig.add_axes ([0.07, 0.05, 0.86, 0.90])
ax.set_title ("% Intermodulation distortion (DIN)")
ax.set_xlim (-55, 5)
ax.set_ylim (1e-3, 1e1)
ax.set_yscale ('log')
ax.bar (Levels - 0.9, D2, 0.8, color = 'b')
ax.bar (Levels + 0.1, D3, 0.8, color = 'g')
ax.text (-51, 5.0, 'IM2', size = 17, color = 'b')
ax.text (-46, 5.0, 'IM3', size = 17, color = 'g')
ax.grid ()
plt.show ()
    
