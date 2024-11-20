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
from cmath import exp
from jacktools.jacksignal import JackSignal


# ------------------ Harmonic distortion measurement ------------------------


# Test level (dB) 
#
Level = -1.0

# Test frequency (Hz) 
#
Freq = 1000


# Create a Jacksignal object.
#
J = JackSignal("THDtest")
if J.get_state() < 0:
    print ("Failed to create JackSignal -- is the server running ?")
    exit(1)

# Get Jack info.
#
name, Fsamp, period = J.get_jack_info()

# Create ports and connect
#
J.create_output (0, "out_1")
J.create_output (1, "out_2")
J.create_input (0, "in_1")
J.silence()
J.connect_output (1, "jaaa:in_1")
J.connect_output (1, "jnoisemeter:in_1")

J.connect_output (0, "jack_nonlin:in")
J.connect_input (0, "jack_nonlin:out")

# Length of test signals, half a second.
#
siglen = int (Fsamp + 0.5) # One second
margin = siglen
buflen = siglen + margin

# Generate test signal, will be scaled for output.
#
Freq = adjust_freq (Freq, Fsamp, siglen)
Aout = gen_sinewave (1.00, Freq, Fsamp, buflen).astype (np.float32)

# Create input buffer.
#
Ainp = np.zeros ((siglen,), dtype = np.float32)

# Generate reference signal for selective measurement.
#
M1 = gen_complex (Freq, Fsamp, siglen, False)

# Define signal buffers and run test. 
# We skip the first second to avoid DC offsets.
#
amp = pow (10.0, Level / 20.0) 
J.set_output_data (0, amp * Aout)
J.set_input_data (0, Ainp, nskip = margin) 
J.process()
J.wait()

# Subtract detected fundamental.    
#
M1 = gen_complex (Freq, Fsamp, siglen, False)
r, p = sigdetect (Ainp, M1)
a = r * exp ((0 -1j) * p)
Ainp -= (a * M1).real

# Output residual.
#
J.set_output_data (0, None)
J.set_output_data (1, Ainp, nloop = 1000)
J.set_input_data (0, None)
J.process()
J.wait()
