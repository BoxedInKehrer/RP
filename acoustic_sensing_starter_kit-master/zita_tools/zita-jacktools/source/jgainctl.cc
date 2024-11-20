// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include "jgainctl.h"


Jgainctl::Jgainctl (const char *client_name, const char *server_name, int nchan) :
    _touch0 (0),
    _touch1 (0),
    _gain0 (-121.0f),
    _gain1 (-121.0f),
    _vgain (500.0f),
    _dgain (0.0f),
    _muted (false),
    _nfrag (0),
    _g (0.0)
{
    if (nchan < 0) nchan = 0;
    if (nchan > MAXCHAN) nchan = MAXCHAN;
    if (   open_jack (client_name, server_name, nchan, nchan)
        || create_inp_ports ("in_%d")
        || create_out_ports ("out_%d"))	_state = FAILED;
    else _state = PROCESS;
}


Jgainctl::~Jgainctl (void)
{
    _state = INITIAL;
    close_jack ();
}


void Jgainctl::set_gain (float g, float v)
{
    if (g < -120.0f) g = -121.0f;
    if (v < 1.0f) v = 1.0f;
    _gain0 = g;
    _vgain = v;
    if (!_muted) _touch0++;
}


int Jgainctl::jack_process (int nframes)
{
    float  *inp [MAXCHAN];
    float  *out [MAXCHAN];
    float  *p, *q;
    float  g, g0, dg, v;
    int    i, j;
    
    if (_state < PROCESS) return 0;
    
    for (i = 0; i < _ninp; i++)
    {
        inp [i] = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
        out [i] = (float *) jack_port_get_buffer (_out_ports [i], nframes);
    }

    if (_touch1 != _touch0)
    {
	if (_muted) { g = -121.0f; v = 1e3f;   }
	else        { g = _gain0;  v = _vgain; }
	_dgain = g - _gain1;
        _nfrag = ceilf ((fabsf (_dgain) * _jack_rate) / (v * _jack_size) + 0.1f);
        _dgain /= _nfrag;
	_touch1 = _touch0;
    }

    g0 = _g;
    if (_nfrag)
    {
	_nfrag--;
	_gain1 += _dgain;
	_g = (_gain1 < -120.0f) ? 0.0f : powf (10.0f, 0.05f * _gain1);
	dg = (_g - g0) / nframes;
    }
    else
    {
	dg = 0;
	if (g0 < 1e-6f)
	{
            for (i = 0; i < _nout; i++)
	    {
	        memset (out [i], 0, nframes * sizeof (float));
	    }
   	    return 0;
	}
    }

    for (i = 0; i < _nout; i++)
    {
        p = inp [i];
        q = out [i];
        g = g0;
        for (j = 0; j < nframes; j++)
        {
            g += dg;
            q [j] = g * p [j];
        }
    }

    return 0;
}

