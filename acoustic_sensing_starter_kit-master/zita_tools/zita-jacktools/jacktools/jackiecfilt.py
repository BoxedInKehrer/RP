# ----------------------------------------------------------------------------
#
#  Copyright (C) 2013-2018 Fons Adriaensen <fons@linuxaudio.org>
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
from jacktools import jackiecfilt_ext


class JackIECfilt(JackClient):

    """
    IEC Class 1 octave and third octave band filters.
    There can be up to 64 filters.
    """

    OFF  = 0
    OCT1 = 1
    OCT3 = 3

    
    def __init__(self, ninp, nout, client_name, server_name = None):
        """
        Create a new Jackiecfilt instance.

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        self._jiecfilt, base = jackiecfilt_ext.makecaps (self, client_name, server_name, ninp, nout)
        super(JackIECfilt, self).__init__(base)



    def set_filter (self, inp, out, filt, band) :
        """
        Set filter type and band.

        If filt is OFF the output is muted.
        If filt is OCT1, band = 0..9,  freq = 31.5, 63, 125 ... 8k, 16k Hz. 
        if filt is OCT3, band = 0..30, freq = 20, 25, 31.5, 40 ... 16k, 20k Hz.
        """
        return jackiecfilt_ext.set_filter (self._jiecfilt, inp, out, filt, band)


