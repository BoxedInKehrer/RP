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
#include "jnoise.h"


Jnoise::Jnoise (const char *client_name, const char *server_name, int nchan) :
    _noisegen (0)
{
    if (nchan < 0) nchan = 0;
    if (nchan > MAXCHAN) nchan = MAXCHAN;
    if (   open_jack (client_name, server_name, 0, nchan)
        || create_out_ports ("out_%d"))
    {
	_state = FAILED;
	return;
    }
    init ();
}


Jnoise::~Jnoise (void)
{
    fini ();
}


void Jnoise::init (void)
{
    _noisegen = new Noisegen [_nout];
    _state = PROCESS;
}


void Jnoise::fini (void)
{
    _state = INITIAL;
    close_jack ();
    delete[] _noisegen;
}


void Jnoise::set_output (int chan, int type, float level)
{
    if ((chan < 0) || (chan >= _nout)) return;
    _noisegen [chan].setparam (type, level);
}


int Jnoise::jack_process (int nframes)
{
    float  *out;
    int    i;
    
    if (_state < PROCESS) return 0;
    for (i = 0; i < _nout; i++)
    {
        out = (float *) jack_port_get_buffer (_out_ports [i], nframes);
	_noisegen [i].process (nframes, out);
    }
    return 0;
}


