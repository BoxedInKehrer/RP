// ----------------------------------------------------------------------------
//
//  Copyright (C) 2013-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#include <math.h>
#include <stdio.h>
#include <string.h>
#include "jiecfilt.h"


Jiecfilt::Jiecfilt (const char *client_name, const char *server_name,
		    int ninp, int nout)
{
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
    init ();
}


Jiecfilt::~Jiecfilt (void)
{
    fini ();
}


void Jiecfilt::init (void)
{
    memset (_inpmap, 0, MAXOUT * sizeof (int));
    memset (_newpars, 0, MAXOUT * sizeof (void *));
    switch (_jack_rate)
    {
    case 44100:
	_oct1set = &Oct1filt44;
	_oct3set = &Oct3filt44;
	break;
    case 48000:
	_oct1set = &Oct1filt48;
	_oct3set = &Oct3filt48;
	break;
    case 88200:
	_oct1set = &Oct1filt88;
	_oct3set = &Oct3filt88;
	break;
    case 96000:
	_oct1set = &Oct1filt96;
	_oct3set = &Oct3filt96;
	break;
    case 19200:
	_oct1set = &Oct1filt192;
	_oct3set = &Oct3filt192;
	break;
    default:
	_state = FAILED;
	return;
    }
    _state = PROCESS;
}


void Jiecfilt::fini (void)
{
    _state = INITIAL;
    close_jack ();
}


void Jiecfilt::set_filter (int inp, int out, int type, int band)
{
    Bp6paramset *S;
    
    if ((inp < 0) || (inp >= _ninp)) return;
    if ((out < 0) || (out >= _nout)) return;
    if (type == 0)
    {
	_newpars [out] = 0;
	return;
    }
    if      (type == 1) S = _oct1set;
    else if (type == 3) S = _oct3set;
    else return;
    if ((band < 0) || (band >= S->_nfilt)) return;
    _inpmap [out] = inp;
    _newpars [out] = S->_param + band;
}


int Jiecfilt::jack_process (int nframes)
{
    float  *inp [MAXINP];
    float  *out;
    int    i;
    
    if (_state < PROCESS) return 0;
    for (i = 0; i < _ninp; i++)
    {
        inp [i] = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
    }
    for (i = 0; i < _nout; i++)
    {
        out = (float *) jack_port_get_buffer (_out_ports [i], nframes);
	_filters [i].setparam (_newpars [i]);
	_filters [i].process (nframes, inp [_inpmap [i]], out);
    }

    return 0;
}

