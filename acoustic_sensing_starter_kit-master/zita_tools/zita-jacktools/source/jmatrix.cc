// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2017 Fons Adriaensen <fons@linuxaudio.org>
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
#include "jmatrix.h"


Jmatrix::Jmatrix (const char *client_name, const char *server_name, int ninp, int nout)
{
    if (ninp > MAXINP) ninp = MAXINP;
    else if (ninp < 0) ninp = 0;
    if (nout > MAXOUT) nout = MAXOUT;
    else if (nout < 0) nout = 0;
    if (   open_jack (client_name, server_name, ninp, nout)
        || create_inp_ports ("in_%d")
        || create_out_ports ("out_%d"))
    {
        _state = FAILED;
        return;
    }
    init ();
}


Jmatrix::~Jmatrix (void)
{
    fini ();
}


void Jmatrix::init (void)
{
    int  i;

    _ginp = new float [_ninp];
    _gout = new float [_nout];
    _gmatr = new float [_ninp * _nout];
    _gcurr = new float [_ninp * _nout];
    memset (_gmatr, 0, _ninp * _nout * sizeof (float));
    memset (_gcurr, 0, _ninp * _nout * sizeof (float));
    for (i = 0; i < _ninp; i++)	_ginp [i] = 1.0f;
    for (i = 0; i < _nout; i++)	_gout [i] = 1.0f;
    _state = PROCESS;
}


void Jmatrix::fini (void)
{
    _state = INITIAL;
    close_jack ();
    delete[] _ginp;
    delete[] _gout;
    delete[] _gmatr;
    delete[] _gcurr;
}


void Jmatrix::set_gain (int inp, int out, float gain)
{
    if (inp >= _ninp) return;
    if (out >= _nout) return;
    if (inp < 0)
    {
	if (out >= 0) _gout [out] = gain;
	return;
    }
    if (out < 0)
    {
	if (inp >= 0) _ginp [inp] = gain;
	return;
    }
    _gmatr [_ninp * out + inp] = gain;
}


int Jmatrix::jack_process (int nframes)
{
    float  *inp [MAXINP];
    float  *out, *p;
    float  g0, g1, dg;
    int    i, j, k, m;
    
    if (_state < PROCESS) return 0;

    for (i = 0; i < _ninp; i++)
    {
        inp [i] = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
    }
    for (j = m = 0; j < _nout; j++, m += _ninp)
    {
        out = (float *) jack_port_get_buffer (_out_ports [j], nframes);
	memset (out, 0, nframes * sizeof (float));
	for (i = 0; i < _ninp; i++)
	{
	    p = inp [i];
	    g0 = _gcurr [m + i];
	    g1 = _gmatr [m + i] * _ginp [i] * _gout [j];
	    dg = g1 - g0;
	    if (fabsf (dg) < 1e-3f * (fabsf (g0) + fabsf (g1)))
	    {
		if (fabsf (g1) >= 1e-15f)
		{
		    for (k = 0; k < nframes; k++)
		    {
		        out [k] += g1 * p [k];
		    }
		}
	    }
	    else
	    {
		g1 = g0;
		dg /= nframes;
		for (k = 0; k < nframes; k++)
		{
		    g1 += dg;
		    out [k] += g1 * p [k];
		}
	    }
            _gcurr [m + i] = g1;
	}
    }
    return 0;
}

