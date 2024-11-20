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
from jacktools.jacksignal import JackSignal
from utils.sinewave import gen_sinewave


# Create a Jacksignal object.
#
J = JackSignal("JS")
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

# Length of test signals in samples.
#
siglen = Fsamp

# Generate test signal.
#
Ainp = np.zeros ((siglen,), dtype = np.float32)
Aout = gen_sinewave (1.0, 1234.0, Fsamp, siglen).astype (np.float32)


J.set_output_data (0, Aout, nloop = -1)
J.set_input_data (0, Ainp, nloop = -1) 
J.process()

p1 = 12000
while True:
    s, p2 = J.get_position ()
    if p2 >= p1:
        print ("%3d %12d %12d" % (s, p1, p2 - p1))
        p1 += 12000
    sleep (0.02)
