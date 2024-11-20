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
from jacktools import jackambpan_ext


class JackAmbpan(JackClient):

    """
    Ambisonic panner.

    ACN, SN3D, up to degree 4.
    """

    def __init__(self, degree, client_name, server_name = None):
        """
        Create a new JackAmbpan instance. 

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        assert (degree >= 1)
        assert (degree <= 4)
        self._jambpan, base = jackambpan_ext.makecaps (self, client_name, server_name, degree)
        super(JackAmbpan, self).__init__(base)


    def set_direction(self, azim, elev, time = 0.1) :
        """
        Set panning direction.

        Angles are in degrees.
        Panning gains will be interpolated linearly over 'time' seconds. 
        """
        return jackambpan_ext.set_direction (self._jambpan, azim, elev, time)

