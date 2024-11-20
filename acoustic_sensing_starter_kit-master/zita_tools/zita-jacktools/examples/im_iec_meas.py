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
from math import hypot, log10
from utils.sinewave import *
from jacktools.jacksignal import JackSignal


# --------------------- IEC Intermodulation measurement ----------------------

# We need a linear X axis for the bargraph.
#
LogF = np.linspace (2.7, 4.3, 17, endpoint = True)
Freq = np.power (10.0, LogF) # 500 Hz to 20 kHz, 1/3 oct.
Fdiff = 60
Level = -20
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

# Length of test signals in samples.
#
siglen = int (Mtime * Fsamp + 0.5)
margin = 1000
buflen = siglen + margin

# Input buffer.
#
Ainp = np.zeros ((buflen,), dtype = np.float32)


# Run test.
#
N = Freq.shape [0]
D2 = N * [0]
D3 = N * [0]
for i in range (N):

    # Generate test signal, will be scaled for output.
    #
    Fcent = Freq [i]
    Flo = Fcent - Fdiff / 2
    Fhi = Fcent + Fdiff / 2
    amp = pow (10.0, Level / 20.0)
    Aout = (  gen_sinewave (amp, Flo, Fsamp, buflen)
            + gen_sinewave (amp, Fhi, Fsamp, buflen)).astype (np.float32)

    # Generate reference signals for selective measurement.
    #
    Mlo = gen_complex (Flo, Fsamp, siglen)
    Mhi = gen_complex (Fhi, Fsamp, siglen)
    M3a = gen_complex (2 * Flo - Fhi, Fsamp, siglen)
    M3b = gen_complex (2 * Fhi - Flo, Fsamp, siglen)
    M2  = gen_complex (Fdiff, Fsamp, siglen)

    # Define signal buffers and run test. 
    #
    J.set_output_data (0, Aout)
    J.set_input_data (0, Ainp) 
    J.process()
    J.wait()
    
    # Skip margin samples and measure.
    # We are not interested in the phase here.
    #
    T = Ainp [margin:]
    Llo,p = sigdetect (T, Mlo)
    Lhi,p = sigdetect (T, Mhi)
    L2,p  = sigdetect (T, M2)
    L3a,p = sigdetect (T, M3a)
    L3b,p = sigdetect (T, M3b)

    # Print results.
    #
    dblo = 20 * log10 (Llo + 1e-20)
    dbhi = 20 * log10 (Lhi + 1e-20)
    D2 [i] = im2 = 100 * L2 / (Llo + Lhi)
    D3 [i] = im3 = 100 * hypot (L3b, L3b) / (Llo + Lhi)
    print ("Freq = %7.1f    Al = %5.1f  Ah = %5.1f   im2 = %6.3f%%  im3 = %6.3f%%" % (Fcent, dblo, dbhi, im2, im3))


# Create a nice graph to present the results.
#
fig = plt.figure (figsize=(8,6), facecolor='white')
ax = fig.add_axes ([0.07, 0.05, 0.86, 0.90])
ax.set_title ("% Intermodulation distortion (IEC)")
ax.set_xlim (2.6, 4.4)
ax.set_xticks ((2.7, 3.0, 3.3, 3.7, 4.0, 4.3))
ax.set_xticklabels (('500', '1k', '2k', '5k', '10k', '20k'))
ax.set_ylim (1e-3, 1e1)
ax.set_yscale ('log')
ax.bar (LogF - 0.022, D2, 0.02, color = 'b')
ax.bar (LogF + 0.002, D3, 0.02, color = 'g')
ax.text (2.75, 5.0, 'IM2', size = 17, color = 'b')
ax.text (2.95, 5.0, 'IM3', size = 17, color = 'g')
ax.grid ()
plt.show ()
    
    
