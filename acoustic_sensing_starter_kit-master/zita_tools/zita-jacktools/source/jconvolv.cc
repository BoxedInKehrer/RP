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


#include <unistd.h>
#include <string.h>
#include "jconvolv.h"

#if ZITA_CONVOLVER_MAJOR_VERSION < 4
#error "This program requires zita-convolver version 4 or higher."
#endif


Jconvolv::Jconvolv (const char *client_name, const char *server_name, int ninp, int nout) :
    _convproc (0),
    _convsync (false)
{
    if (ninp > MAXINP) ninp = MAXINP;
    else if (ninp < 1) ninp = 1;
    if (nout > MAXOUT) nout = MAXOUT;
    else if (nout < 1) nout = 1;
    if (zita_convolver_major_version () != ZITA_CONVOLVER_MAJOR_VERSION)
    {
	fprintf (stderr, "Zita-convolver does not match compile-time version.\n");
        _state = FAILED;
        return;
    }
    if (   open_jack (client_name, server_name, ninp, nout)
        || create_inp_ports ("in_%d")
        || create_out_ports ("out_%d"))
    {
        _state = FAILED;
        return;
    }
    _convproc = new Convproc ();
}


Jconvolv::~Jconvolv (void)
{
    delete _convproc;
}


void Jconvolv::set_state (int state)
{
    switch (state)
    {
    case SILENCE:
	_state = SILENCE;
	if (!_convproc->stop_process ())
	{
	     while (!_convproc->check_stop ()) usleep (100000);
	}
        break;
    case PROCESS:
	if (!_convproc->start_process (_priority, _schedpol))
	{
	    _state = PROCESS;
	}
	break;
    }
}


int Jconvolv::configure (uint32_t maxlen, float density)
{
    if (_state < 0) return Converror::BAD_STATE;
    return _convproc->configure (_ninp, _nout, maxlen, _jack_size,
				 _jack_size, Convproc::MAXPART, density);
}


void Jconvolv::jack_freewheel (int state)
{
    _convsync = state ? true : false;
}


int Jconvolv::jack_buffsize (int nframes)
{
    return 1;   
}


int Jconvolv::jack_process (int nframes)
{
    float *p;

    if (_state < SILENCE) return 0;
    if (_convproc->state () != Convproc::ST_PROC)
    {
	for (int i = 0; i < _nout; i++)
	{
            p = (float *) jack_port_get_buffer (_out_ports [i], nframes);
	    memset (p, 0, nframes * sizeof (float));
	}
	return 0;
    }
    for (int i = 0; i < _ninp; i++)
    {
        p = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
        memcpy (_convproc->inpdata (i), p, nframes * sizeof (float));
    }
    _convproc->process (_convsync);
    for (int i = 0; i < _nout; i++)
    {
        p = (float *) jack_port_get_buffer (_out_ports [i], nframes);
        memcpy (p, _convproc->outdata (i), nframes * sizeof (float));
    }
    return 0;
}

