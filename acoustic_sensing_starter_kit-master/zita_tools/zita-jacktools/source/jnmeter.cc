// ----------------------------------------------------------------------------
//
//  Copyright (C) 2011..2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include "jnmeter.h"


Jnmeter::Jnmeter (const char *client_name, const char *server_name,
		  int ninp, int nout, float *levels) :
    _levels (levels)
{
    int i;
    
    if (ninp < 1) ninp = 1;
    if (ninp > MAXINP) ninp = MAXINP;
    if (nout < 1) nout = 1;
    if (nout > MAXOUT) nout = MAXOUT;
    if (   open_jack (client_name, server_name, ninp, nout)
        || create_inp_ports ("in_%d")
        || create_out_ports ("out_%d"))
    {
	_state = FAILED;
	return;
    }
    _nprocs = new Nmeterdsp [nout];
    for (i = 0; i < nout; i++)
    {
        _nprocs [i].init (jack_rate ());
	_inpmap [i] = 0;
    }
    _state = PROCESS;
}


Jnmeter::~Jnmeter (void)
{
    _state = INITIAL;
    close_jack ();
    delete[] _nprocs;
}


int Jnmeter::jack_process (int nframes)
{
    int    i;
    float  *inp [MAXINP];
    float  *out;
    
    if (_state != PROCESS) return 0;
    for (i = 0; i < _ninp; i++)
    {
        inp [i] = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
    }
    for (i = 0; i < _nout; i++)
    {
        out = (float *) jack_port_get_buffer (_out_ports [i], nframes);
	_nprocs [i].process (inp [_inpmap [i]], out, nframes);
    }
    return 0;
}


int Jnmeter::set_input (int inp, int out)
{
    int i;
    
    if (_state != PROCESS) return 1;
    if ((inp <  0) || (inp >= _ninp)) return 1;
    if ((out < -1) || (out >= _nout)) return 1;
    if (out < 0)
    {
	for (i = 0; i < _nout; i++) _inpmap [i] = inp;
    }
    else _inpmap [out] = inp;
    return 0;
}


int Jnmeter::set_filter (int out, int ftype, int dcfilt)
{
    int i, rv = 0;

    if (_state != PROCESS) return 1;
    if ((out < -1) || (out >= _nout)) return 1;
    if (out < 0)
    {
	for (i = 0; i < _nout; i++)
	{
	    rv |= _nprocs [i].set_filter (ftype, dcfilt);
	}
	return rv;
    }
    return _nprocs [out].set_filter (ftype, dcfilt);
}
    

int Jnmeter::set_detect (int out, int dtype)
{
    int i, rv = 0;

    if (_state != PROCESS) return 1;
    if ((out < -1) || (out >= _nout)) return 1;
    if (out < 0)
    {
	for (i = 0; i < _nout; i++)
	{
	    rv |= _nprocs [i].set_detect (dtype);
	}
	return rv;
    }
    return _nprocs [out].set_detect (dtype);
}


int Jnmeter::get_levels (void)
{
    int i;

    for (i = 0; i < _ninp; i++) _levels [i] = _nprocs [i].level ();
    return _state;
}


