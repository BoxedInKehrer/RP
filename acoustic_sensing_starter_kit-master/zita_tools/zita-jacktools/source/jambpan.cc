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


#include "jambpan.h"


Jambpan::Jambpan (const char *client_name, const char *server_name, int degree) :
    _ambpan (0)
{
    int n;
    
    if (degree < 1) degree = 1;
    if (degree > 4) degree = 4;
    n = (degree + 1) * (degree + 1);
    if (   open_jack (client_name, server_name, 1, n)
        || create_inp_port (0, "in")
        || create_out_ports ("out.%d"))
    {
	_state = FAILED;
	return;
    }
    init (degree);
}


Jambpan::~Jambpan (void)
{
    fini ();
}


void Jambpan::init (int degree)
{
    _ambpan = new Ambpan4 (_jack_rate, degree, true);
    _state = PROCESS;
}


void Jambpan::fini (void)
{
    _state = INITIAL;
    close_jack ();
    delete _ambpan;
}


void Jambpan::set_direction (float az, float el, float dt)
{
    if (!_ambpan) return;
    _ambpan->set_direction (az, el, dt);
}


int Jambpan::jack_process (int nframes)
{
    int    i;
    float  *inp;
    float  *out [MAXOUT];
    
    if (_state < PROCESS) return 0;
    inp = (float *) jack_port_get_buffer (_inp_ports [0], nframes);
    for (i = 0; i < _nout; i++)
    {
        out [i] = (float *) jack_port_get_buffer (_out_ports [i], nframes);
    }
    _ambpan->process (nframes, inp, out);
    return 0;
}

