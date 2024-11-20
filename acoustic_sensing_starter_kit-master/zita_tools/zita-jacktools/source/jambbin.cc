// ----------------------------------------------------------------------------
//
//  Copyright (C) 2015-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#include "jambbin.h"


Jambbin::Jambbin (const char *client_name, const char *server_name, int maxlen, int degree) :
    _ambbin (0)
{
    int n;
    
    if (degree < 1) degree = 1;
    if (degree > 4) degree = 4;
    n = (degree + 1) * (degree + 1);
    if (   open_jack (client_name, server_name, n, 2)
        || create_inp_ports ("in.%d", 0)
        || create_out_port  (0, "out.L")
        || create_out_port  (1, "out.R"))
    {
	_state = FAILED;
	return;
    }
    _ambbin = new Ambbin4 (_jack_rate, degree, maxlen, _jack_size);
    _state = PROCESS;
}


Jambbin::~Jambbin (void)
{
    _state = INITIAL;
    close_jack ();
    delete _ambbin;
}


int Jambbin::jack_process (int nframes)
{
    float  *inp [25];
    float  *out [2];

    if (_state < PROCESS) return 0;
    for (int i = 0; i < _ninp; i++)
    {
        inp [i] = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
    }
    out [0] = (float *) jack_port_get_buffer (_out_ports [0], nframes);
    out [1] = (float *) jack_port_get_buffer (_out_ports [1], nframes);
    _ambbin->process (nframes, inp, out);
    return 0;
}

