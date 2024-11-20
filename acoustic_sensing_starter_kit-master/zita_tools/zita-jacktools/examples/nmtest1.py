#!/usr/bin/python

# ----------------------------------------------------------------------------
#
#  Copyright (C) 2012-2018 Fons Adriaensen <fons@linuxaudio.org>
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
from jacktools.jacknmeter import JackNmeter
from jacktools.jacksignal import JackSignal
import matplotlib.pyplot as plt


# Test JackNmeter filters
# -----------------------


# Create a JackSignal object.
#
S = JackSignal ("JackSignal")
if S.get_state () < 0:
    print ("Failed to create JackSignal -- is Jack running ?")
    exit(1)

# Create a JackNmeter object.
#
M = JackNmeter (1, 1, "JackNmeter")
if M.get_state () < 0:
    print ("Failed to create JackNmeter -- is Jack running ?")
    exit(1)

# Create and connect ports.
#
S.create_output (0, "out-1")
S.create_input (0, "in-1")
S.silence()
S.connect_output (0, "JackNmeter:in_0")
S.connect_input (0, "JackNmeter:out_0")

# Get Jack info.
#
name, fsamp, period = S.get_jack_info ()

# Parameters
#
impval = 1.0
fftlen = 16384

# Create and assign buffers.
#
Ain1 = np.empty ([fftlen,], dtype = np.float32)
Aout = np.zeros ([100,], dtype = np.float32)
Aout [0] = impval
S.set_input_data (0, Ain1, nskip = period - 10) # Skip one period minus 10 samples.
S.set_output_data (0, Aout)
Freq = np.linspace (0, fsamp / 2, num = fftlen // 2 + 1)

# Run one test.
#
def run_test (ftype, dcfilt, title):
    M.set_input (0, 0);
    M.set_filter (0, ftype, dcfilt);
    S.process()
    S.wait()
    Spec = np.fft.rfft (Ain1)
    fig = plt.figure (figsize=(8,6), facecolor='white')
    ax1 = fig.add_axes ([0.06, 0.05, 0.90, 0.42])
    ax1.set_ylim (-1.5, 1.5)
    ax1.grid ()
    ax2 = fig.add_axes ([0.06, 0.53, 0.90, 0.42])
    ax2.set_xlim (5, fsamp / 2)
    ax2.set_ylim (-60, 15)
    ax2.set_xscale ('log')
    ax2.grid ()
    ax2.set_title (title)
    ax1.plot (Ain1 [0:500], color='b', lw=1)
    ax2.plot (Freq, 20 * np.log10 (np.abs (Spec) + 1e-10), color='b', lw=1)
    plt.show()
    
# Test all filters.    
#
print ("Close plot to start next test.")
run_test (M.FIL_NONE,   True,  "DC filter only")
run_test (M.FIL_ENB20K, True,  "DC filter and 20 kHz filter")
run_test (M.FIL_IEC_A,  False, "IEC A-filter")
run_test (M.FIL_IEC_C,  False, "IEC C-filter")
run_test (M.FIL_ITU468, False, "ITU468 filter")
run_test (M.FIL_DOLBY,  False, "Dolby filter")
print ("Done.")

