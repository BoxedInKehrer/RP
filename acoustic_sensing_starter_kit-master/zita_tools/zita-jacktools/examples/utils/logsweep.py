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
from math import floor, log, exp, sin, cos, pi


def genlogsweep (amp, fmin, fmax, fsamp, tfi, tsw, tfo):

    """
    Generate logarithmic sweep, inverse sweep and phase correction.
   
    Parameters:
       amp:           Amplitude of sweep
       fmin, fmax:    Frequency range of the constant amplitude part. 
       fsamp:         Sample frequency.
       tsw:           Duration of constant amplitude part, seconds.
       tfi, tfo:      Fade in and out times, seconds.
   
    During the fade in and out segments the frequency increases at
    the same logarithmic rate as during the constant amplitude part
    from fmin to fmax. The fade out time should be chosen so that
    the frequency never exceeds half the sample rate.
   
    The inverse sweep amplitude is dimensioned such that after
    convolution the result is a bandlimited Dirac impulse with
    unity spectral magnitude between fmin and fmax.
    """
    
    k0 = int (tfi * fsamp + 0.5)
    k1 = int (tsw * fsamp + 0.5)
    k2 = int (tfo * fsamp + 0.5)
    a = log (fmax / fmin) / k1
    b = fmin / (a * fsamp)
    r = 4.0 * a * a / amp
    n = k0 + k1 + k2
    Afwd = np.empty ([n,], dtype = np.float32)
    Ainv = np.empty ([n,], dtype = np.float32)
    for i in range (n):
        j = n - i - 1
        if    i < k0: g = sin (0.5 * pi * i / k0)
        elif  j < k2: g = sin (0.5 * pi * j / k2)
        else:         g = 1.0 
        d = b * exp (a * (i - k0))
        p = d - b
        p -= floor (p)
        x = g * sin (2 * pi * p)
        Afwd [i] = x * amp
        Ainv [j] = x * d * r
    return Afwd, Ainv, b - floor (b)


def genlogsweep_orig (amp, fmin, fmax, fsamp, tfi, tsw, tfo):

    """
    Generate logarithmic sweep and inverse.
   
    Parameters:
       amp:           Amplitude of sweep
       fmin, fmax:    Frequency range of the constant amplitude part. 
       fsamp:         Sample frequency.
       tsw:           Duration of constant amplitude part, seconds.
       tfi, tfo:      Fade in and out times, seconds.
   
    During the fade in and out segments the frequency increases at
    the same logarithmic rate as during the constant amplitude part
    from fmin to fmax. The fade out time should be chosen so that
    the frequency never exceeds half the sample rate.
   
    The inverse sweep amplitude is dimensioned such that after
    convolution the result is a bandlimited Dirac impulse with
    unity spectral magnitude between fmin and fmax.
    """
    
    k0 = int (tfi * fsamp + 0.5)
    k1 = int (tsw * fsamp + 0.5)
    k2 = int (tfo * fsamp + 0.5)
    a = log (fmax / fmin) / k1
    b = fmin / (a * fsamp)
    c = exp (a) - 1.0
    r = 4.0 * a * a / amp
    n = k0 + k1 + k2
    Afwd = np.empty ([n,], dtype = np.float32)
    Ainv = np.empty ([n,], dtype = np.float32)
    p = 0;
    for i in range (n):
        j = n - i - 1
        if    i < k0: g = sin (0.5 * pi * i / k0)
        elif  j < k2: g = sin (0.5 * pi * j / k2)
        else: g = 1.0 
        x = g * sin (2 * pi * p)
        d = b * exp (a * (i - k0))
        p += c * d;
        if p > 1.0: p -= 2.0
        Afwd [i] = x * amp
        Ainv [j] = x * d * r
    return Afwd, Ainv


