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
from jacktools import jackgainctl_ext


class JackGainctl(JackClient):

    """
    Multichannel dezippered audio gain control.
    """

    def __init__(self, nchan, client_name, server_name = None):
        """
        Create a new JackGainctl instance. The initial state is muted.

        The optional 'server_name' allows to select between running
        Jack servers. The result should be checked using get_state().
        """
        assert (nchan > 0)
        self._jgainctl, base = jackgainctl_ext.makecaps (self, client_name, server_name, nchan)
        super(JackGainctl, self).__init__(base)


    def set_gain(self, gain, rate = 500) :
        """
        Set gain in dB.

        Gain will change smoothly to the given value by 'rate'
        dB per second. A 'gain' less than -120 dB will result in
        a full mute. The mimium value for 'rate' is 1 dB/s.
        """
        return jackgainctl_ext.set_gain (self._jgainctl, gain, rate)


    def set_muted(self, muted) :
        """
        Set muted state.

        The current gain setting is preserved.
        """
        return jackgainctl_ext.set_muted (self._jgainctl, muted)


