# ----------------------------------------------------------------------------
#
#  Copyright (C) 2008-2018 Fons Adriaensen <fons@linuxaudio.org>
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


from jacktools.jackclient import JackClient
from jacktools import jackparameq_ext


class JackParameq(JackClient):

    """
    Multichannel equaliser. There can be up to eight sections, each of
    them can be a low or high shelf filter or a parametric one.
    The shelf filters have a variable shape. For the  parametric three
    variations are available, these differ only in the way the gain and
    bandwidth settings interact. See the 'shelfeq.py' and 'parameq.py'
    example programs for details. All controls are fully dezippered. 
    Also switching bypass on/off will be click-free.
    """

    def __init__(self, nchan, types, client_name, server_name = None):
        """
        Create a new JackParameq instance with 'nchan' channels.

        The 'types' argument is a string of section types. Valid types
        are 'L' or 'H' for a low or high shelf filter, '1','2' or '3'
        for a parametric.
        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        self._jparameq, base = jackparameq_ext.makecaps (self, client_name,
                                                         server_name,
                                                         nchan, types)
        super(JackParameq, self).__init__(base)


    def set_filter (self, sect, freq, gain, shape):
        """
        Set filter parameters for single section.

        For a parametric 'shape' controls the relative bandwidth
        (1/Q), in the range 0.1 to 10. For the shelf filters the
        range is 0 to 1. The 'shelfeq.py' examples programs shows
        how this affects the shape of the response. Frequencies
        are in Hz, gains in dB.
        """
        return jackparameq_ext.set_filter (self._jparameq, sect,
                                           freq, gain, shape)

    
    def set_gain (self, gain):
        """
        Set make up gain in dB.
        """
        return jackparameq_ext.set_gain (self._jparameq, gain)


    def set_bypass (self, onoff):
        """
        Set global bypass on (True) or off (False).
        """
        return jackparameq_ext.set_bypass (self._jparameq, onoff)


