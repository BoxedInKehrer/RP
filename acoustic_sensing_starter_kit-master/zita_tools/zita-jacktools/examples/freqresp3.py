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
import matplotlib.pyplot as plt
from scipy.signal import convolve # much faster than numpy.convolve
from jacktools.jacksignal import JackSignal
from utils.logsweep import genlogsweep


# ------- Frequency response using log sweep, deconvolution and fft ---------


# Create a JackSignal object and connect.
#
J = JackSignal("JackSignal")
if J.get_state() < 0:
    print ("Failed to create JackSignal -- is Jack running ?")
    exit(1)
J.create_output (0, "out")
J.create_input (0, "in")
J.silence()
J.connect_output (0, "jnoisemeter:in_1")
J.connect_input (0, "jnoisemeter:out")

# Get Jack info.
#
name, fsamp, period = J.get_jack_info ()

# Parameters.
#
amp = 0.1
fmin = 20
fmax = 22.0e3
tfade1 = 0.50
tsweep = 3.00
tfade2 = 0.01
implen = 5000

# Generate sweep.
#
Afwd, Ainv, p = genlogsweep (amp, fmin, fmax, fsamp, tfade1, tsweep, tfade2)

# Create input buffer.
#
k = implen + Afwd.shape [0] - 1
Ainp = np.zeros ([k,], dtype = np.float32)

# Connect buffer to ports.
#
J.set_output_data (0, Afwd)
J.set_input_data (0, Ainp [implen//2:], nskip = period)

# Run the test
#
print ("Measuring....")
J.process()
J.wait()
del J

# Process and store result
#
print ("Convolving....")
Imp = convolve (Ainp, Ainv, mode = 'valid')
Lfft = 65536
Nbin = Lfft // 2 + 1
Spec = np.fft.rfft (Imp, Lfft)

# Display impulse and magnitude response.
#
print ("Display...")
nsam = 200
Imp = Imp [implen//2 - nsam:implen//2 + nsam + 1]
Time = np.linspace (-nsam, nsam, 2 * nsam + 1)
Freq = np.linspace (0, fsamp / 2, Nbin)

fig = plt.figure (figsize=(8,6), facecolor='white')
k = Afwd.shape [0]
ax1 = fig.add_axes ([0.05, 0.04, 0.90, 0.45])
ax1.set_ylim (-1.5, 1.5)
ax1.plot (Time, Imp, color='b', lw=1)
ax1.grid ()

ax2 = fig.add_axes ([0.05, 0.53, 0.90, 0.45])
ax2.set_xlim (10, 24e3)
ax2.set_ylim (-60, 15)
ax2.set_xscale ('log')
ax2.plot (Freq, 20 * np.log10 (np.abs (Spec) + 1e-10), color='b', lw=1)
ax2.grid ()
plt.show()
