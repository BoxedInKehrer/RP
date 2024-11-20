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


#include <unistd.h>
#include <string.h>
#include "jmatconv.h"


Jmatconv::Jmatconv (const char *client_name, const char *server_name, int size, int ninp, int nout, int nthr) :
    _state_seq1 (0),
    _state_seq2 (0),
    _dconv (0)
{
    if (ninp > MAXINP) ninp = MAXINP;
    else if (ninp < 1) ninp = 1;
    if (nout > MAXOUT) nout = MAXOUT;
    else if (nout < 1) nout = 1;
    if (size > MAXLEN) size = MAXLEN;
    else if (size < 16) size = 16;
    if (   open_jack (client_name, server_name, ninp, nout)
        || create_inp_ports ("in_%d")
        || create_out_ports ("out_%d"))
    {
        _state = FAILED;
        return;
    }
    init (size, nthr);
}


Jmatconv::~Jmatconv (void)
{
    fini ();
}


void Jmatconv::init (int size, int nthr)
{
    _dconv = new Denseconv (_ninp, _nout, size, _jack_size, nthr, _priority + 10);
    _state = _new_state = SILENCE;
}


void Jmatconv::fini (void)
{
    _state = INITIAL;
    usleep (100000);
    close_jack ();
    delete _dconv;
}


void Jmatconv::set_state (int state)
{
    if (_state < PASSIVE) return;
    _state_seq1++;
    _new_state = state;
    while (_state_seq2 != _state_seq1) _state_sync.wait ();
}


int Jmatconv::jack_process (int nframes)
{
    int     i;
    float   *inpp [MAXINP];
    float   *outp [MAXOUT];

    if (_state < PASSIVE) return 0;
    if (_state_seq1 != _state_seq2)
    {
	_state = _new_state;
	_state_seq2++;
	_state_sync.post ();
    }
    if (_state < SILENCE) return 0;
    for (i = 0; i < _nout; i++)
    {
        outp [i] = (float *) jack_port_get_buffer (_out_ports [i], nframes);
    }
    if (_state != PROCESS)
    {
        for (i = 0; i < _nout; i++)
        {
  	    memset (outp [i], 0, nframes * sizeof (float));
        }
	return 0;
    }
    for (i = 0; i < _ninp; i++)
    {
        inpp [i] = (float *) jack_port_get_buffer (_inp_ports [i], nframes);
    }
    _dconv->process (inpp, outp);
 
    return 0;
}

