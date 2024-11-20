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
from jacktools import jacknoise_ext


class JackNoise(JackClient) :
    """
    Generate white and pink noise.
    """

    OFF   = 0
    WHITE = 1
    PINK  = 2

    
    def __init__(self, nchan, client_name, server_name = None):
        """
        Create a new JackNoise instance with 'nchan' independent outputs.

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        self._jnoise, base = jacknoise_ext.makecaps (self, client_name, server_name, nchan)
        super(JackNoise).__init__(base)


    def set_output (self, chan, type, level):     
        """
        Set noise type and level in dB FS. Initially all channels are OFF.
        """
        return jacknoise_ext.set_output (self._jnoise, chan, type, level)


