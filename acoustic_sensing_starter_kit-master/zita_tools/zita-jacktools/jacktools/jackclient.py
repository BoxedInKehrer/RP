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


from jacktools import jackclient_ext


class JackClient(object) :

    """
    Base class for most jacktools classes.
    Do not instantiate directly.
    """

    INITIAL =  0
    PASSIVE =  1
    SILENCE =  2
    PROCESS = 10
    FAILED  = -1
    ZOMBIE  = -2

    
    def __init__(self, jclient) :
        """
        For use by derived classes only !!
        Store a pointer to the C++ base class.
        """
        self._jclient = jclient


    def get_state(self) :
        """
        Returns the current state. Possible states are:
        
        PASSIVE =  1  Running as Jack client, ports are not accessed.
        SILENCE =  2  Outputs are silent, inputs are ignored. 
        PROCESS = 10  Normal processing state. 
        FAILED  = -1  Failed to intialise or run-time error.
        ZOMBIE  = -2  Kicked out by Jack.

        Most jacktools classes start up in the PROCESS state, and don't 
        have the PASSIVE or SILENCE states. Some clients have additional
        states.

        The only option in the last two states is to delete the object.
        """
        return jackclient_ext.get_state (self._jclient)

    def get_jack_info(self) :
        """
        Return the jack client name, sample rate, period size.
        """
        return jackclient_ext.get_jack_info (self._jclient)

    def create_input(self, ind, name) :
        """
        Create an input port with index 'ind' and name 'name'.
        Indices start at zero. Possible in the PASSIVE state only.
        """
        return jackclient_ext.port_operation (self._jclient, 0, ind, name)

    def create_output(self, ind, name) :
        """
        Create an output port with index 'ind' and name 'name'.
        Indices start at zero. Possible in the PASSIVE state only.
        """
        return jackclient_ext.port_operation (self._jclient, 1, ind, name)
    
    def delete_input(self, ind) :
        """
        Delete the input port with index 'ind', or all input
        ports if 'ind' == -1. Possible in the PASSIVE state only.
        """
        return jackclient_ext.port_operation (self._jclient, 2, ind, 0)

    def delete_output(self, ind) :
        """
        Delete the input port with index 'ind', or all input
        ports if 'ind' == -1. Possible in the PASSIVE state only.
        """
        return jackclient_ext.port_operation (self._jclient, 3, ind, 0)

    def rename_input(self, ind, name) :
        """
        Rename input port with index 'ind' to 'name'.
        Indices start at zero.
        """
        return jackclient_ext.port_operation (self._jclient, 4, ind, name)

    def rename_output(self, ind, name) :
        """
        Rename output port with index 'ind' to 'name'.
        Indices start at zero.
        """
        return jackclient_ext.port_operation (self._jclient, 5, ind, name)

    def connect_input(self, ind, srce) :
        """
        Connect the input with index 'ind' to 'srce'.
        """
        return jackclient_ext.port_operation (self._jclient, 6, ind, srce)

    def connect_output(self, ind, dest) :
        """
        Connect the output with index 'ind' to 'dest'.
        """
        return jackclient_ext.port_operation (self._jclient, 7, ind, dest)

    def disconn_input (self, ind, srce = None) :
        """
        Disconnect an input port, or all input ports if 'ind' == -1.
        In the first case, if 'srce' is not None, only the named
        source is disconnected.
        """
        return jackclient_ext.port_operation (self._jclient, 8, ind, srce)

    def disconn_output (self, ind, dest = None) :
        """
        Disconnect an output port, or all output ports if 'ind' == -1.
        In the first case, if 'dest' is not None, only the named
        destination is disconnected.
        """
        return jackclient_ext.port_operation (self._jclient, 9, ind, dest)


