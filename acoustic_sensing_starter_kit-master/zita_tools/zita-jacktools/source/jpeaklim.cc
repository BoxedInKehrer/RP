// ----------------------------------------------------------------------------
//
//  Copyright (C) 2010-2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include <string.h>
#include "jpeaklim.h"


Jpeaklim::Jpeaklim (const char *client_name, const char *server_name, int nchan)
{
    if (nchan < 1) nchan = 1;
    if (nchan > MAXCHAN) nchan = MAXCHAN;
    if (   open_jack (client_name, server_name, nchan, nchan)
        || create_inp_ports ("in_%d")
        || create_out_ports ("out_%d"))	_state = FAILED;
    else
    {
	_peaklim.init (_jack_rate, _ninp);
	_state = PROCESS;
    }
}


Jpeaklim::~Jpeaklim (void)
{
    _state = INITIAL;
    close_jack ();
}


int Jpeaklim::jack_process (int nframes)
{
    float  *inp [MAXCHAN];
    float  *out [MAXCHAN];
    int    i;
    
    if (_state < PROCESS) return 0;
    for (i = 0; i < _ninp; i++)
    {
        inp [i] = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
        out [i] = (float *) jack_port_get_buffer (_out_ports [i], nframes);
    }
    _peaklim.process (nframes, inp, out);
    return 0;
}

