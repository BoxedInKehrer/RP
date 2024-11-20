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


from time import sleep 
from jacktools.jackclient import JackClient
from jacktools import jacksignal_ext


class JackSignal(JackClient):

    """
    Jack client using numpy arrays for input and output.

    The main use of a JackSignal object is to generate
    arbitrary test signals and capture the results.
    There can be up to 64 inputs and outputs.
    """

    TRIGGER = 9

    def __init__(self, client_name, server_name = None) :
        """
        Create a new JackSignal instance.

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        self._jsignal, base = jacksignal_ext.makecaps (self, client_name, server_name)
        super(JackSignal, self).__init__(base)
        

    def passive(self):
        """
        Set the state to PASSIVE.

        This is also the initial state after creation, unless it failed.
        In this state the object is a jack client, but the process callback
        does not access any ports. Use this state to create or delete ports.
        Having connected ports in PASSIVE state may output garbage.  
        """
        return jacksignal_ext.set_state (self._jsignal, JackSignal.PASSIVE)


    def silence(self) :
        """
        Set the state to SILENCE.

        In this state the process callback outputs silence
        and discards any input. Use this state to connect or
        disconnect ports and define signal buffers.
        """
        return jacksignal_ext.set_state (self._jsignal, JackSignal.SILENCE)


    def process(self):
        """
        Set the state to PROCESS.

        In this state the process callback outputs and records
        signals until the end of the largest buffer is reached.
        At that point it returns to the SILENCE state.
        """
        return jacksignal_ext.set_state (self._jsignal, JackSignal.PROCESS)


    def trigger(self, ind):
        """
        Select the trigger input and set the state to TRIGGER.

        In this state the object is waiting for an external start
        signal. The state will be set to PROCESS by the first sample
        on the trigger input that exceeds +0.5f. Playback and capture
        will be synchronised to the position of that sample.
        """
        return jacksignal_ext.set_state (self._jsignal, JackSignal.PROCESS)


    def get_position (self) :
        """
        Return state, frame count.
        """
        return jacksignal_ext.get_posit (self._jsignal)


    def wait(self, dt = 0.1):
        """
        Wait while the state is TRIGGER or PROCESS.
        """
        actset =  (JackSignal.TRIGGER, JackClient.PROCESS)
        while self.get_state () in actset: sleep (dt); 

            
    def set_input_data (self, ind, data, nloop = 1, nskip = 0):
        """
        Define an input signal.

        Set the capture buffer for the input port with index
        'ind' to 'data'. The 'data' must be a single dimension
        float32 array or array view. To reset, use 'None' as the
        'data' argument. The JackSignal object keeps a reference
        to the array until it is reset or replaced, or the object
        is deleted.

        The optional 'nskip' parameter allows to skip a number
        of frames before starting capture. 
        """
        return jacksignal_ext.set_input_data (self._jsignal, ind, data, nloop, nskip)

    
    def set_output_data (self, ind, data, nloop = 1, nskip = 0):
        """
        Define an output signal.

        Set the playback buffer for the output port with index
        'ind' to 'data'. The 'data' must be a single dimension
        float32 array or array view. To reset, use 'None' as the
        'data' argument. The JackSignal object keeps a reference
        to the array until it is reset or replaced, or the object
        is deleted.

        The optional 'nloop' parameter can be used to loop the
        buffer, e.g. for generating continuous signals.
        """
        return jacksignal_ext.set_output_data (self._jsignal, ind, data, nloop, nskip)


