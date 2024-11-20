# ----------------------------------------------------------------------------
#
#  Copyright (C) 2014-2015 Fons Adriaensen <fons@linuxaudio.org>
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


from audiotools import vresampler_ext


class VResampler() :

    """
    Python bindings to zita-resampler::VResampler
    See the C++ library documentation for details.
    """

    def __init__(self, ratio, nchan, hlen, frel = None):
        """
        Create a new VResampler object.
        """
        if frel is None: frel = 1.0 - 2.6 / hlen
        self._capsule = vresampler_ext.create (self, ratio, nchan, hlen, frel)


    def reset (self):
        """
        Reset the VResampler.
        """
        return vresampler_ext.reset (self._capsule)


    def inpsize (self):
        """
        Return inpsize ();
        """
        return vresampler_ext.inpsize (self._capsule)


    def inpdist (self):
        """
        Return inpdist ();
        """
        return vresampler_ext.inpdist (self._capsule)


    def set_phase (self, v):
        """
        Set the resampling filter phase.
        """
        return vresampler_ext.set_phase (self._capsule, v)
    

    def set_rrfilt (self, v):
        """
        Set the rratio filter.
        """
        return vresampler_ext.set_rrfilt (self._capsule, v)
    

    def set_rratio (self, v):
        """
        Set the relative ratio.
        """
        return vresampler_ext.set_rratio (self._capsule, v)
    

    def process (self, inp, out):
        """
        Resample data.
        """
        return vresampler_ext.process (self._capsule, inp, out)
    

