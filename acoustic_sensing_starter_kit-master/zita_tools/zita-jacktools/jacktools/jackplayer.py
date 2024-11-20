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
from jacktools import jackplayer_ext


class JackPlayer(JackClient):

    """"
    Multichannel audio file player, with high quality resampling.
    Supports up to 64 channels. 
    """
    
    STOPPED = 10
    STOPLOC = 11
    PLAYING = 12
    PLAYLOC = 13
    ENDFILE = 14

    def __init__(self, nchan, client_name, server_name = None) :
        """
        Create a new JackPlayer instance with 'nchan' outputs.

        The optional 'server_name' allows to select between running
         servers. The result should be checked using get_state().
        """
        self._jplayer, base = jackplayer_ext.makecaps (self, client_name, server_name, nchan)
        super(JackPlayer, self).__init__(base)


    def get_position (self) :
        """
        Return state, file position.

        Additional states are:

        STOPPED = 10    Player is active but stopped.
        STOPLOC = 11    Waiting for locate() to finish.    
        PLAYING = 12    Player is playing the audiofile.
        PLAYLOC = 13    Waiting for locate(), play when ready.
        ENDFILE = 14    Player stopped at end of file.

        The file position is in frames at the audio file sample rate.
        """
        return jackplayer_ext.get_posit (self._jplayer)


    def silence(self) :
        """
        Goto the SILENCE state.

        In this state the player utputs silence and does not respond to any
        transport commands. Use this state to connect or disconnect ports,
        and to open or close audio files.
        """
        return jackplayer_ext.set_state (self._jplayer, JackClient.SILENCE)


    def stop(self) :
        """
        Stop playback with transport commands enabled.
        """
        return jackplayer_ext.set_state (self._jplayer, JackPlayer.STOPPED)


    def play(self) :
        """
        Start playback.
        """
        return jackplayer_ext.set_state (self._jplayer, JackPlayer.PLAYING)


    def set_gain(self, gain, time = 0) :
        """
        Set playback gain in dB.

        Gain will change smoothly to the given value in 'time'
        seconds. Values less than -150 dB will result in a complete
        mute. The initial gain is 0 dB.
        """
        return jackplayer_ext.set_gain (self._jplayer, gain, time)


    def locate(self, posit) :
        """
        Locate to position in frames at the audio file sample rate.

        This can be used while stopped or while playing. If used in
        PLAYING state, playback resumes when at least one second or
        all remaining audio has been buffered.
        """
        return jackplayer_ext.set_posit (self._jplayer, posit)


    def open_file(self, name) :  
        """
        Open an audio file. Player must be in SILENCE state.
        Returns zero on success, non-zero otherwise.
        """
        return jackplayer_ext.open_file (self._jplayer, name)


    def close_file(self) :  
        """
        Close the current audio file. Player must be in SILENCE state.
        Returns zero on success, non-zero otherwise.
        """
        return jackplayer_ext.close_file (self._jplayer)


    def get_file_info(self) :
        """
        Returns channels, sample rate, lenght in frames.
        """
        return jackplayer_ext.get_file_info (self._jplayer)


