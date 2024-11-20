# ----------------------------------------------------------------------------
#
#  Copyright (C) 2012-2018 Fons Adriaensen <fons@linuxaudio.org>
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
from jacktools import jackconvolv_ext


class JackConvolv(JackClient) :

    """
    """

    def __init__(self, ninp, nout, client_name, server_name = None):
        """
        Create a new JackConvolv instance, with 'ninp' inputs and 'nout'
        outputs. 

        The optional 'server_name' allows to select between runningA1, fs, ft = read_audio ("/home/fons/ladproj/jconvolver/config-files/weird1.wav");

        Jack servers. The result should be checked using get_state().
        """
        self._ninp = ninp
        self._nout = nout
        self._jconvolv, base = jackconvolv_ext.makecaps (self, client_name, server_name, ninp, nout)
        super(JackConvolv, self).__init__(base)
        

    def silence(self):
        """
        Set the state to SILENCE. 

        In this state the convolver outputs silence. This is
        also the initial state unless the constructor failed.
        """
        return jackconvolv_ext.set_state (self._jconvolv, JackClient.SILENCE)


    def process(self):
        """
        Set the state to PROCESS.

        In this state the convolver is operating normally.
        It is still possible to modify the impulse responses.
        """
        return jackconvolv_ext.set_state (self._jconvolv, JackClient.PROCESS)

    
    def configure(self, maxlen, density):
        """
        Configure or reconfigure the convolver. 
        The object must be in the SILENCE state.
        """
        return jackconvolv_ext.configure (self._jconvolv, maxlen, density)


    def cleanup(self):
        """
        Reset the concolver configuration to the initial empty state.
        The object must be in the SILENCE state.
        """
        return jackconvolv_ext.configure (self._jconvolv, 0, 0.0)

    
    def impdata_create(self, data, inp, out, offs = 0):
        """
        Add 'data' to the impulse response for the given input and
        output, starting at sample postion 'offs'.

        The object must be in SILENCE state and configured. The 'data'
        argument should be a single dimension float32 numpy array or
        array view. 
        """
        return  jackconvolv_ext.impdata_write (self._jconvolv, data, inp, out, offs, 1)
        

    def impdata_link(self, inp1, out1, inp2, out2):
        """
        Set the impulse response for (inp2, out2) to a copy of the
        one for (inp1, out1). 

        The object must be in SILENCE state and configured.
        Links can be used to conserve memory when a number of identical
        long impulse responses is required. The precomputed frequency
        domain form of the impulse response is shared between linked
        matrix elements.
        """
        return jackconvolv_ext.impdata_link (self._jconvolv, inp1, out1, inp2, out2)
        

    def impdata_clear(self, data, inp, out):
        """
        Clear the existing impulse response for the given input and
        output.

        This can be called while in PROCESS state and is typically
        used before impdata_update() to replace the existing impulse
        responses.
        """
        return jackconvolv_ext.impdata_write (self._jconvolv, None, inp, out, 0, 0)
        

    def impdata_update(self, data, inp, out, offs = 0):
        """
        Add 'data' to the impulse response for the given input and
        output, starting at sample postion 'offs'.

        This method is used to modify existing impulse response data
        while in PROCESS state. It will modify existing partitions only.
        The 'data' argument should be a single dimension float32 numpy
        array or array view. 
        """
        return jackconvolv_ext.impdata_write (self._jconvolv, data, inp, out, offs, 0)
        


