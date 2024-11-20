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


import numpy as np
from audiotools import audiofile_ext


class AudioFile() :

    """
    Read/write audio file data to/from numpy arrays.

    Data type of the arrays must be 32-bit float,
    with sample values in the range (-1,+1). 
    """

    default_maxread  = 1000000
    default_sampfreq = 48000
    default_filetype = 'wav,24bit'

    MODE_RD = 1
    MODE_WR = 2

    def __init__(self):
        """
        Create a new AudioFile object.

        An AudioFile object has no public data members.
        All acces should be via the function members only.
        """
        self._capsule = audiofile_ext.create (self)
        self._channels = None
        self._sampfreq = None
        self._filesize = None
        self._filename = None
        self._filemode = None
        self._filetype = None


    def open_read (self, filename):
        """
        Open an existing audio file for reading.

        Any currently open file is closed first. Returns MODE_RD,
        or None if opening the file fails.
        """
        self.close ()
        audiofile_ext.open_read (self._capsule, filename)
        if self.info (): self._filename = filename
        return self._filemode


    def open_write (self, filename, channels, sampfreq, filetype = None):
        """
        Create and open a new audio file for writing.

        The filetype argument should be a string consisting of the
        words listed below, comma separated and without spaces.
        Dither is applied to 16-bit files only.

        File types:      caf, wav, amb, aiff, flac.
        Sample formats:  16bit, 24bit, 32bit, float. 
        Dither type:     none, rec, tri, lips.

        Any currently open file is closed first. Returns MODE_WR,
        or None if opening the file fails.
        """
        self.close ()
        audiofile_ext.open_write (self._capsule, filename, channels, sampfreq, filetype)
        if self.info (): self._filename = filename
        return self._filemode


    def info (self):
        """
        Read parameters of the current file to local copies.

        For internal use only, called by the open* members.
        Returns the current access mode or 0.
        """
        R = audiofile_ext.info (self._capsule)
        if R [0] > 0:
            self._filemode = R [0]
            self._channels = R [1]
            self._sampfreq = R [2]
            self._filesize = R [3]
            self._filetype = R [4] + "," + R [5]
        return R [0] 
    

    def close (self):
        """
        Close the current audio file.
        """
        self._channels = None
        self._sampfreq = None
        self._filesize = None
        self._filename = None
        self._filemode = None
        self._filetype = None
        return audiofile_ext.close(self._capsule)


    def seek (self, posit, mode = 0):
        """
        Seek to new position, in frames.
        """
        return audiofile_ext.seek(self._capsule, posit, mode)


    def read (self, data):
        """
        Read audio frames from file into a numpy array 'data'.

        The 'data' argument can be an array or an array view
        created by slicing or reshaping. The first dimension
        determines the number of frames, the second dimension
        must be equal to the number of channels in the audio
        file. For a single channel file a 1-dimensional array
        or view, or a python array is accepted as well.

        Note that if the slicing operation results in a copy,
        no data will be returned as the copy will be discarded
        after being filled in. Basic slicing and reshaping
        always create valid view.
        """
        return audiofile_ext.read(self._capsule, data)


    def write (self, data):
        """
        Write audio frames from numpy array 'data' to file.

        See read() for more info on the 'data' argument.
        """
        return audiofile_ext.write(self._capsule, data)


    def channels (self):
        """
        Return the number of channels in the current file.
        """
        return self._channels


    def sampfreq (self):
        """
        Return the sample frequency of the current file.
        """
        return self._sampfreq


    def filesize (self):
        """
        Return the size of the current file, in frames.
        """
        return self._filesize


    def filename (self):
        """
        Return the current filename.
        """
        return self._filename


    def filemode (self):
        """
        Return the current access mode.
        """
        return self._filemode


    def filetype (self):
        """
        Return a string describing the file type and format.
        """
        return self._filetype



def read_audio (filename, maxread = AudioFile.default_maxread):
    """
    Commodity function. Read an entire file and return a
    2-D array containing the audio data, the sample rate
    and the file type. The optional second parameter puts
    a limit on the number of frames read and returned.
    """
    F = AudioFile ()
    F.open_read (filename)
    ns = min (F.filesize (), maxread)
    sf = F.sampfreq ()
    ft = F.filetype ()
    A = np.empty ([ns, F.channels ()], dtype = np.float32)
    F.read (A)
    F.close ()
    return (A, sf, ft)


def write_audio (A, filename, sampfreq = AudioFile.default_sampfreq, filetype = None):
    """
    Commodity fuction. Writes the vector or 2-D array A to
    an audio file with the given sample rate and type. The
    type options are the same as for write().
    """
    F = AudioFile ()
    if A.ndim == 1: nc = 1
    elif A.ndim == 2: nc = A.shape [1]
    else: raise TypeError ("Array dimension must be 1 or 2")
    F.open_write (filename, nc, sampfreq, filetype)
    F.write (A)
    F.close ()
