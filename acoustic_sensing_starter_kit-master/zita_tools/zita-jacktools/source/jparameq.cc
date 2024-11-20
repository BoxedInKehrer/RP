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
#include <stdio.h>
#include <string.h>
#include "jparameq.h"


Jparameq::Jparameq (const char *client_name, const char *server_name,
		    int nchan, const char *types):
    _nsect (0)
{
    if (nchan < 0) nchan = 0;
    if (nchan > MAXCHAN) nchan = MAXCHAN;
    if (   open_jack (client_name, server_name, nchan, nchan)
        || create_inp_ports ("in_%d")
        || create_out_ports ("out_%d"))
    {
	_state = FAILED;
	return;
    }
    init (types);
}


Jparameq::~Jparameq (void)
{
    fini ();
}


void Jparameq::init (const char *types)
{
    int ft = 0;
    
    _nsect = strlen (types);
    if (_nsect > MAXSECT) _nsect = MAXSECT;
    for (int i = 0; i < _nsect; i++)
    {
        switch (types [i])
	{
	case 'L': ft = Svparam2::LS; break;    
	case 'H': ft = Svparam2::HS; break;    
	case '1': ft = Svparam2::P1; break;    
	case '2': ft = Svparam2::P2; break;    
	case '3': ft = Svparam2::P3; break;
	}
	_filters [i] = new Svparam2 (ft);
    }
    _frag = (int)(ceilf (0.01f * jack_rate ()));
    _todo = 0;
    _state = PROCESS;
}


void Jparameq::fini (void)
{
    _state = INITIAL;
    close_jack ();
    for (int i = 0; i < _nsect; i++) delete _filters [i];
}


void Jparameq::set_filter (int sect, float freq, float gain, float bandw)
{
    Svparam2 *F;

    if ((sect < 0) || (sect >= _nsect)) return;
    F = _filters [sect];
    if (F)
    {
        freq /= jack_rate ();
        gain = powf (10.0f, gain / 20.0f);
        F->setpars (freq, gain, bandw);
    }
}


void Jparameq::set_bypass (bool onoff)
{
    _eqgain.bypass (onoff);
    for (int i = 0; i < _nsect; i++)
    {
	_filters [i]->bypass (onoff);
    }
}


void Jparameq::set_gain (float gain)
{
    _eqgain.setgain (powf (10.0f, gain / 20.0f));
}


int Jparameq::jack_process (int nframes)
{
    int    i, k;
    float  *inp [MAXCHAN];
    float  *out [MAXCHAN];
    
    if (_state < PROCESS) return 0;
    for (i = 0; i < _ninp; i++)
    {
        inp [i] = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
        out [i] = (float *) jack_port_get_buffer (_out_ports [i], nframes);
    }

    while (nframes)
    {
	if (_todo == 0)
	{
	    _eqgain.prepare (_frag);
  	    for (i = 0; i < _nsect; i++) _filters [i]->prepare (_frag);
	    _todo = _frag;
	}
	k = nframes;
	if (k > _todo) k = _todo;
	_eqgain.process (k, _nout, inp, out);
        for (i = 0; i < _nsect; i++) _filters [i]->process (k, _nout, out);
	for (i = 0; i < _nout; i++)
	{
	    inp [i] += k;
	    out [i] += k;
        }
        _todo -= k;
	nframes -= k;
    }
    
    return 0;
}

