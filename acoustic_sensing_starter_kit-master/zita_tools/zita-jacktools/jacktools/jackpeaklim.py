# ----------------------------------------------------------------------------
#
#  Copyright (C) 2010-2018 Fons Adriaensen <fons@linuxaudio.org>
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
from jacktools import jackpeaklim_ext


class JackPeaklim(JackClient):

    """
    Multichannel dezippered audio gain control.
    """

    def __init__(self, nchan, client_name, server_name = None):
        """
        Create a new JackPeaklim instance. The initial state is muted.

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        assert (nchan > 0)
        self._jpeaklim, base = jackpeaklim_ext.makecaps (self, client_name, server_name, nchan)
        super(JackPeaklim, self).__init__(base)


    def set_inpgain(self, inpgain) :
        """
        Set input gain in dB.
        """
        return jackpeaklim_ext.set_inpgain (self._jpeaklim, inpgain)


    def set_threshold(self, threshold) :
        """
        Set threshold in dB.
        """
        return jackpeaklim_ext.set_threshold (self._jpeaklim, threshold)


    def set_release(self, release) :
        """
        Set release time in seconds.
        """
        return jackpeaklim_ext.set_release (self._jpeaklim, release)


