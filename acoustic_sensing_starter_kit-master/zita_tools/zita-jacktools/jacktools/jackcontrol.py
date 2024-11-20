# ----------------------------------------------------------------------------
#
#  Copyright (C) 2008-2014 Fons Adriaensen <fons@linuxaudio.org>
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


from jacktools import jackcontrol_ext


class JackControl(object):
    """
    Jack connection and transport control.
    """

    ACTIVE = 1
    FAILED = -1
    ZOMBIE = -2


    def __init__(self, client_name, server_name = None):
        """
        Create a new JackControl instance.

        The optional 'server_name' allows to select between running
        servers. The result should be checked using get_client_state()
        """
        self._capsule = jackcontrol_ext.create (self, client_name, server_name)


    def get_jack_info(self):
        """
        Return the tuple (jack client name, sample rate, period size).
        """
        return jackcontrol_ext.get_jack_info (self._capsule)


    def get_client_state(self):
        """
        Returns the current client state, which will be one of

        ACTIVE  = 1   The object is active.
        FAILED  = -1  Failed to become a Jack client.
        ZOMBIE  = -2  Kicked out by Jack.

        The only option in the last two states is to delete the object.
        """
        return jackcontrol_ext.get_client_state (self._capsule)


    def get_transport_state(self):
        """
        Returns the tuple (transport state, transport frame).
        
        Transport state is one of

        STOPPED = 0  Transport is stopped.
        PLAYING = 1  Transport is rolling.
        SYNCING = 2  Waiting for clients to locate.
        """
        return jackcontrol_ext.get_transport_state (self._capsule)


    def transport_start(self):
        """
        Start the jack transport.
        """  
        return jackcontrol_ext.transport_start(self._capsule)


    def transport_stop(self):
        """
        Stop the jack transport.
        """  
        return jackcontrol_ext.transport_stop((self._capsule,))


    def transport_locate(self, frame):
        """
        Locate jack transport to postion 'frame'.
        """  
        return jackcontrol_ext.transport_locate(self._capsule, frame)


    def connect_ports(self, srce, dest):
        """
        Connect readable jack port 'srce' to writeable jack port 'dest'
        """        
        return jackcontrol_ext.connect_ports(self._capsule, srce, dest)


    def disconn_ports(self, srce, dest):
        """
        Disconnect writeable jack port 'dest' from readable jack port 'srce'
        """        
        return jackcontrol_ext.disconn_ports(self._capsule, srce, dest)

