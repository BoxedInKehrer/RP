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


import numpy as np
from time import sleep
from utils.sinewave import *
from jacktools.jacksignal import JackSignal


# ------------------- IEC Intermodulation test signal -----------------------

Fcent = 3000
Fdiff = 60

# Level in dB (of both frequencies)
#
Level = -10.0


# Create a Jacksignal object.
#
J = JackSignal("IMtest")
if J.get_state() < 0:
    print ("Failed to create JackSignal -- is the server running ?")
    exit(1)

# Get Jack info.
#
name, Fsamp, period = J.get_jack_info()

# Create one output and connect.
#
J.create_output (0, "out")
J.silence()
J.connect_output (0, "jaaa:in_1")

# Generate test signal. Since we will loop this we need an
# exactly integer number of cycles in the buffer.
#
siglen = int (1.0 * Fsamp + 0.5) # 1 second
Flo = adjust_freq (Fcent - 0.5 * Fdiff, Fsamp, siglen)
Fhi = adjust_freq (Fcent + 0.5 * Fdiff, Fsamp, siglen)
Amp = pow (10.0, Level / 20.0)
Aout = (  gen_sinewave (Amp, Flo, Fsamp, siglen)
        + gen_sinewave (Amp, Fhi, Fsamp, siglen)).astype (np.float32)

# Output signal, loop 1000 times.
#
J.set_output_data (0, Aout, nloop = 1000)
J.process()
J.wait()

# Cleanup
#    
del A    
del J

