// ----------------------------------------------------------------------------
//
//  Copyright (C) 2012-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#include "jambrot.h"


Jambrot::Jambrot (const char *client_name, const char *server_name, int degree) :
    _ambrot (0)
{
    int n;
    
    if (degree < 1) degree = 1;
    if (degree > 4) degree = 4;
    n = (degree + 1) * (degree + 1);
    if (   open_jack (client_name, server_name, n, n)
        || create_inp_ports ("in.%d")
        || create_out_ports ("out.%d"))
    {
	_state = FAILED;
	return;
    }
    init (degree);
}


Jambrot::~Jambrot (void)
{
    fini ();
}


void Jambrot::init (int degree)
{
    _ambrot = new Ambrot4 (_jack_rate, degree);
    _state = PROCESS;
}


void Jambrot::fini (void)
{
    _state = INITIAL;
    close_jack ();
    delete _ambrot;
}


void Jambrot::set_quaternion (float w, float x, float y, float z, float t)
{
    if (!_ambrot) return;
    _ambrot->set_quaternion (w, x, y, z, t);
}


int Jambrot::jack_process (int nframes)
{
    int    i;
    float  *inp [MAXINP];
    float  *out [MAXOUT];
    
    if (_state < PROCESS) return 0;
    for (i = 0; i < _nout; i++)
    {
        inp [i] = (float *) jack_port_get_buffer (_inp_ports [0], nframes);
        out [i] = (float *) jack_port_get_buffer (_out_ports [i], nframes);
    }
    _ambrot->process (nframes, inp, out);
    return 0;
}

