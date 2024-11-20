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


from jacktools.jackclient import JackClient
from jacktools import jackmatconv_ext
from audiotools.audiofile import AudioFile
import numpy as np


class JackMatconv(JackClient) :

    """
    Jack client implementing a zero delay convolution matrix. 
    This version is optimized for a dense matrix of up to 64 x 64
    relative short convolutions. It uses partitioned convolution
    with a single partition size equal to the Jack period.
    The CPU load can be spread over multiple parallel threads.
    For longer convolutions or sparse (e.g diagonal) matrices
    the JackConvolv class will provide better performance.
    """

    def __init__(self, size, ninp, nout, nthr, client_name, server_name = None):
        """
        Create a new JackMatconv instance, with 'ninp' inputs and 'nout'
        outputs, and a maximum impulse length of 'size' samples. The CPU
        load will be spread over 'nthr' threads.
        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        self._ninp = ninp
        self._nout = nout
        self._jmatconv, base = jackmatconv_ext.makecaps (self, client_name, server_name, size, ninp, nout, nthr)
        super(JackMatconv, self).__init__(base)
        

    def silence(self):
        """
        Set the state to SILENCE.

        In this state the convolver outputs silence. This is
        also the initial state unless the constructor failed.
        """
        return jackmatconv_ext.set_state (self._jmatconv, JackClient.SILENCE)


    def process(self):
        """
        Set the state to PROCESS.

        In this state the convolver is operating normally.
        It is still possible to modify the impulse responses.
        """
        return jackmatconv_ext.set_state (self._jmatconv, JackClient.PROCESS)


    def load_impulse(self, data, inp, out, gain = 1.0):
        """
        Load the impulse response for a single input,output pair.

        The 'data' must be a single dimension float32 numpy array
        or array view, or 'None' to clear the current response.
        The 'gain' parameter is a linear gain factor.
        """
        return jackmatconv_ext.load_impulse (self._jmatconv, data, inp, out, gain)
        

    def load_matrix (self, file, size, gain = 1.0):
        """
        Convenience function, load the entire convolution
        matrix from an audio file.

        If 'ninp' and 'nout' are the matrix dimensions, the
        file must have 'ninp' channels and consist of 'nout'
        sections of 'size' frames each. 
        The 'gain' parameter is a linear gain factor.
        """
        jname, fsamp, psize = self.get_jack_info ()
        F = AudioFile ()
        try:
            F.open_read (file)
            assert F.channels () == self._ninp
            assert F.filesize () == self._nout * size
            assert F.sampfreq () == fsamp
            A = np.empty ((size, self._ninp), dtype = np.float32)
            for i in range (self._nout):
                F.read (A)
                for j in range (self._ninp):
                     self.load_impulse (A [:,j], j, i, gain)
        finally:             
            F.close ()
    

    def load_matrix_transposed (self, file, size, gain = 1.0):
        """
        Convenience function, load the entire convolution
        matrix from an audio file.

        If 'ninp' and 'nout' are the matrix dimensions, the
        file must have 'nout' channels and consist of 'ninp'
        sections of 'size' frames each. 
        The 'gain' parameter is a linear gain factor.
        """
        jname, fsamp, psize = self.get_jack_info ()
        F = AudioFile ()
        try:
            F.open_read (file)
            assert F.channels () == self._nout
            assert F.filesize () == self._ninp * size
            assert F.sampfreq () == fsamp
            A = np.empty ((size, self._nout), dtype = np.float32)
            for i in range (self._ninp):
                F.read (A)
                for j in range (self._nout):
                     self.load_impulse (A [:,j], i, j, gain)
        finally:             
            F.close ()
    
