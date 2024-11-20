#!/usr/bin/python

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


from distutils.core import setup, Extension


audiofile = Extension ('audiotools.audiofile_ext',
                       ['source/audiofile_ext.cc',
                        'source/audiofile.cc',
                        'source/dither.cc',],
                       libraries = ['sndfile'])

vresampler = Extension ('audiotools.vresampler_ext',
                        ['source/vresampler_ext.cc'],
                        libraries = ['zita-resampler'])


setup (name = 'audiotools',
    version = '1.0.0',
    description = 'Tools for audio processing',
    license = 'LGPL',
    author = 'Fons Adriaensen',
    author_email = 'fons@linuxaudio.org',
    url = 'http://kokkinizita.linuxaudio.org/linuxaudio',
    packages = ['audiotools'],
    py_modules = ['audiotools.audiofile',
                  'audiotools.vresampler'],
    ext_modules = [audiofile,
                   vresampler])


