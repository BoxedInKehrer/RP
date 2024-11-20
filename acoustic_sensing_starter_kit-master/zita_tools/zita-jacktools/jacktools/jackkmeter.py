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


from array import array
from jacktools.jackclient import JackClient
from jacktools import jackkmeter_ext


class JackKmeter(JackClient) :

    """
    """

    def __init__(self, nchan, client_name, server_name = None):
        """
        Create a new JackKmeter instance.

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_levels().
        """
        self._rms = array ('f', (0 for i in range (nchan)))
        self._peak = array ('f', (0 for i in range (nchan)))
        self._jkmeter, base = jackkmeter_ext.makecaps (self, client_name, server_name, nchan, self._rms, self._peak)
        super(JackKmeter, self).__init__(base)


    def get_levels(self) :
        """
        Return state and updated 'rms' and 'peak' arrays.
        """
        return jackkmeter_ext.get_levels (self._jkmeter), self._rms, self._peak


