// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2015 Fons Adriaensen <fons@linuxaudio.org>
//    
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// ----------------------------------------------------------------------------


#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include "jkmeter.h"


Jkmeter::Jkmeter (const char *client_name, const char *server_name, int nchan, float *rms, float *pks) :
    _rms (rms),
    _pks (pks)
{
    if (nchan < 0) nchan = 0;
    if (nchan > MAXINP) nchan = MAXINP;
    if (   open_jack (client_name, server_name, nchan, 0)
        || create_inp_ports ("in_%d"))
    {
	_state = FAILED;
	return;
    }
    Kmeterdsp::init (_jack_rate, _jack_size, 0.5f, 15.0f);
    _kproc = new Kmeterdsp [nchan];
    _state = PROCESS;
}


Jkmeter::~Jkmeter (void)
{
    _state = INITIAL;
    close_jack ();
    delete[] _kproc;
}


int Jkmeter::jack_process (int nframes)
{
    int    i;
    float  *p;

    if (_state != PROCESS) return 0;
    for (i = 0; i < _ninp; i++)
    {
        p = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
	_kproc [i].process (p, nframes);
    }
    return 0;
}


int Jkmeter::get_levels (void)
{
    int  i;

    for (i = 0; i < _ninp; i++) _kproc [i].read (_rms + i, _pks + i);
    return _state;
}


