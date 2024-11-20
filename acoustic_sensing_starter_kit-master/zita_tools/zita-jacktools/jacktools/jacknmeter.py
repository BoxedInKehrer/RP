# ----------------------------------------------------------------------------
#
#  Copyright (C) 2015 Fons Adriaensen <fons@linuxaudio.org>
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


from array import array
from jacktools.jackclient import JackClient
from jacktools import jacknmeter_ext


class JackNmeter(JackClient) :
    """
    Multichannel audio level meter with standard weighting
    filters and level detector ballistics. Each channel can
    have its own filter and/or detector type. Also provides
    the filtered input signals. Channels numbers start at 0.
    Levels are on a linear scale, not in dB.
    """

    FIL_NONE   = 0
    FIL_ENB20K = 1 # 20 kHz noise bandwidth
    FIL_IEC_A  = 2
    FIL_IEC_C  = 3
    FIL_ITU468 = 4
    FIL_DOLBY  = 5 # ITU468 with reduced gain
    
    DET_NONE      = 0
    DET_RMS       = 1
    DET_RMS_SLOW  = 2
    DET_VUM       = 3
    DET_VUM_SLOW  = 4
    DET_ITU468    = 5

    
    def __init__(self, ninp, nout, client_name, server_name = None):
        """
        Create a new JackNmeter instance. Initially no filter or
        detector is selected.

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_levels().
        """
        self._levels = array ('f', (0 for i in range (nout)))
        self._jnmeter, base = jacknmeter_ext.makecaps (self, client_name, server_name, ninp, nout, self._levels)
        super(JackNmeter, self).__init__(base)
        
        
    def set_input(self, inp, out) :
        """
        Select input for given output, or for all outputs if 'out' < 0.
        """
        return jacknmeter_ext.set_input (self._jnmeter, inp, out)


    def set_filter(self, out, ftype, dcfilt) :
        """
        Set weighting and DC filters for output, or for all if 'out' < 0.
        """
        return jacknmeter_ext.set_filter (self._jnmeter, out, ftype, dcfilt)


    def set_detect(self, out, dtype) :
        """
        Set detector type for given output, or for all if 'out' < 0.
        """
        return jacknmeter_ext.set_detect (self._jnmeter, out, dtype)


    def get_levels(self) :
        """
        Return state and updated array of levels.
        """
        return jacknmeter_ext.get_levels (self._jnmeter), self._levels


