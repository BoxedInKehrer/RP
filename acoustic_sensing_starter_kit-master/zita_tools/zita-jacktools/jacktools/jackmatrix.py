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
from jacktools import jackmatrix_ext


def db2lin (gain):
    """
    Convert gain in dB to linear value.
    """
    return pow (10.0, 0.05 * gain)


class JackMatrix(JackClient):

    """
    Audio matrix with gain controls for each input, output, and
    input, output pair. Gains are linear and can be negative. Use
    db2lin() to set gains in dB.
    Gain changes are interpolated linearly over one Jack period.
    Absolute gain values lower than 1e-15 (-300 dB) are set to zero. 
    """

    def __init__(self, ninp, nout, client_name, server_name = None):
        """
        Create a new JackMatrix instance with given number of inputs
        and outputs. All initial gains are zero.

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        assert (ninp > 0)
        assert (nout > 0)
        self._jmatrix, base = jackmatrix_ext.makecaps (self, client_name, server_name, ninp, nout)
        super(JackMatrix, self).__init__(base)


    def set_matrix_gain(self, inp, out, gain) :
        """
        Set gain for matrix element.
        """
        return jackmatrix_ext.set_gain (self._jmatrix, inp, out, gain)


    def set_input_gain(self, inp, gain) :
        """
        Set input gain.
        """
        return jackmatrix_ext.set_gain (self._jmatrix, inp, -1, gain)


    def set_output_gain(self, out, gain) :
        """
        Set output gain.
        """
        return jackmatrix_ext.set_gain (self._jmatrix, -1, out, gain)


