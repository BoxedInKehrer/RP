#!/usr/bin/python

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


from distutils.core import setup, Extension


jackclient = Extension ('jacktools.jackclient_ext',
                         ['source/jackclient_ext.cc'],
                         libraries = ['zita-jclient', 'jack'])

jackcontrol = Extension ('jacktools.jackcontrol_ext',
                         ['source/jackcontrol_ext.cc',
                          'source/jcontrol.cc'],
                         libraries = ['zita-jclient', 'jack'])

jackambbin = Extension ('jacktools.jackambbin_ext',
                        ['source/jackambbin_ext.cc',
                         'source/jambbin.cc',
                         'source/ambbin4.cc',
                         'source/ambrot4.cc',
                         'source/nffilt.cc',
                         'source/binconv.cc'],
                        libraries = ['zita-jclient', 'jack',
                                     'fftw3f'])

jackambpan = Extension ('jacktools.jackambpan_ext',
                        ['source/jackambpan_ext.cc',
                         'source/jambpan.cc',
                         'source/ambpan4.cc'],
                        libraries = ['zita-jclient', 'jack'])

jackambrot = Extension ('jacktools.jackambrot_ext',
                        ['source/jackambrot_ext.cc',
                         'source/jambrot.cc',
                         'source/ambrot4.cc'],
                        libraries = ['zita-jclient', 'jack'])

jackmatrix = Extension ('jacktools.jackmatrix_ext',
                        ['source/jackmatrix_ext.cc',
                         'source/jmatrix.cc'],
                        libraries = ['zita-jclient', 'jack'])

jackiecfilt = Extension ('jacktools.jackiecfilt_ext',
                        ['source/jackiecfilt_ext.cc',
                         'source/jiecfilt.cc',
                         'source/bp6filter.cc',
                         'source/oct1param1.cc',
                         'source/oct3param1.cc'],
                        libraries = ['zita-jclient', 'jack'])

jackkmeter = Extension ('jacktools.jackkmeter_ext',
                        ['source/jackkmeter_ext.cc',
                         'source/jkmeter.cc',
                         'source/kmeterdsp.cc'],
                        libraries = ['zita-jclient', 'jack'])

jacknmeter = Extension ('jacktools.jacknmeter_ext',
                        ['source/jacknmeter_ext.cc',
                         'source/jnmeter.cc',
                         'source/nmeterdsp.cc'],
                        libraries = ['zita-jclient', 'jack'])

jacknoise = Extension ('jacktools.jacknoise_ext',
                       ['source/jacknoise_ext.cc',
                        'source/rngen.cc',
                        'source/jnoise.cc',
                        'source/noisegen.cc'],
                       libraries = ['zita-jclient', 'jack'])

jackplayer = Extension ('jacktools.jackplayer_ext',
                        ['source/jackplayer_ext.cc',
                         'source/jplayer.cc',
                         'source/afreader.cc',
                         'source/posixthr.cc'],
                        libraries = ['zita-jclient', 'jack',
                                     'sndfile', 'zita-resampler'])

jacksignal = Extension ('jacktools.jacksignal_ext',
                        ['source/jacksignal_ext.cc',
                         'source/jsignal.cc'],
                        libraries = ['zita-jclient', 'jack'])

jackmatconv = Extension ('jacktools.jackmatconv_ext',
                        ['source/jackmatconv_ext.cc',
                         'source/denseconv.cc',
                         'source/jmatconv.cc',
                         'source/posixthr.cc'],
                        libraries = ['zita-jclient', 'jack',
                                     'fftw3f'])

jackconvolv = Extension ('jacktools.jackconvolv_ext',
                        ['source/jackconvolv_ext.cc',
                         'source/jconvolv.cc'],
                        libraries = ['zita-jclient', 'jack',
                                     'zita-convolver', 'fftw3f'])

jackgainctl = Extension ('jacktools.jackgainctl_ext',
                        ['source/jackgainctl_ext.cc',
                         'source/jgainctl.cc'],
                        libraries = ['zita-jclient', 'jack'])

jackpeaklim = Extension ('jacktools.jackpeaklim_ext',
                        ['source/jackpeaklim_ext.cc',
                         'source/jpeaklim.cc',
                         'source/peaklim.cc'],
                        libraries = ['zita-jclient', 'jack'])

jackparameq = Extension ('jacktools.jackparameq_ext',
                        ['source/jackparameq_ext.cc',
                         'source/jparameq.cc',
                         'source/eqgain.cc',
                         'source/svparam2.cc'],
                        libraries = ['zita-jclient', 'jack'])


setup (name = 'jacktools',
    version = '1.0.1',
    description = 'Audio tools running as a Jack client',
    license = 'LGPL',
    author = 'Fons Adriaensen',
    author_email = 'fons@linuxaudio.org',
    url = 'http://kokkinizita.linuxaudio.org/linuxaudio',
    packages = ['jacktools'],
    py_modules = ['jacktools.jackclient',
                  'jacktools.jackcontrol',
                  'jacktools.jackambbin',
                  'jacktools.jackambpan',
                  'jacktools.jackambrot',
                  'jacktools.jackmatrix',
                  'jacktools.jackiecfilt',
                  'jacktools.jackkmeter',
                  'jacktools.jacknmeter',
                  'jacktools.jacknoise',
                  'jacktools.jackplayer',
                  'jacktools.jacksignal',
                  'jacktools.jackmatconv',
                  'jacktools.jackconvolv',
                  'jacktools.jackgainctl',
                  'jacktools.jackpeaklim',
                  'jacktools.jackparameq'],
    ext_modules = [jackclient,
                   jackcontrol,
                   jackambbin,
                   jackambpan,
                   jackambrot,
                   jackmatrix,
                   jackiecfilt,
                   jackkmeter,
                   jacknmeter,
                   jacknoise,
                   jackplayer,
                   jacksignal,
                   jackmatconv,
                   jackconvolv,
                   jackgainctl,
                   jackpeaklim,
                   jackparameq])
