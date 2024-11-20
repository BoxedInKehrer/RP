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
from jacktools import jackambrot_ext


class JackAmbrot(JackClient):

    """
    Ambisonic rotation on arbitrary axis.

    ACN, SN3D or N3D, up to degree 4.
    """

    def __init__(self, degree, client_name, server_name = None):
        """
        Create a new JackAmbrot instance. 

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        assert (degree >= 1)
        assert (degree <= 4)
        self._jambrot, base = jackambrot_ext.makecaps (self, client_name, server_name, degree)
        super(JackAmbrot, self).__init__(base)


    def set_quaternion(self, w, x, y, z, t = 0.1) :
        """
        Set rotation quaternion.

        The rotation matrix will be interpolated linearly over 't' seconds. 
        """
        return jackambrot_ext.set_quaternion (self._jambrot, w, x, y, z, t)

