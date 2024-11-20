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


#include <unistd.h>
#include <string.h>
#include "jsignal.h"


Jsignal::Jsignal (const char *client_name, const char *server_name) :
    _state_seq1 (0),
    _state_seq2 (0),
    _triginp (0),
    _offset (0),
    _fcount (0)
{
//    if (ninp < 0) ninp = 0;
//    if (ninp> MAXINP) ninp = MAXINP;
//    if (nout < 0) nout = 0;
//    if (nout> MAXOUT) nout = MAXOUT;
//    if (   open_jack (client_name, server_name, ninp, nout)
//        || create_inp_ports ("in_%d")
//        || create_out_ports ("out_%d"))
    if (open_jack (client_name, server_name, MAXINP, MAXOUT))
    {
        _state = FAILED;
        return;
    }
    init ();
}


Jsignal::~Jsignal (void)
{
    fini ();
}


void Jsignal::init (void)
{
    _state = _new_state = PASSIVE;
}


void Jsignal::fini (void)
{
    int i;

    set_state (PASSIVE);
    close_jack ();
    // Release Python owned buffers.
    for (i = 0; i < MAXINP; i++) set_buffer (_inp_data + i, 0, 0);
    for (i = 0; i < MAXOUT; i++) set_buffer (_out_data + i, 0, 0);
}


void Jsignal::set_state (int state)
{
    if (_state < PASSIVE) return;
    _state_seq1++;
    _new_state = state;
    while (_state_seq2 != _state_seq1) _state_sync.wait ();
}


void Jsignal::init_process (void)
{
    int     i;
    Jsdata  *D;

    _fcount = 0;
    _offset = 0;
    for (i = 0, D = _inp_data; i < MAXINP; i++, D++) D->_isamp = D->_iloop = D->_iskip = 0;
    for (i = 0, D = _out_data; i < MAXOUT; i++, D++) D->_isamp = D->_iloop = D->_iskip = 0;
}


int Jsignal::jack_process (int nframes)
{
    int    i, act;
    float  *p;

    if (_state < PASSIVE) return 0;
    if (_state_seq1 != _state_seq2)
    {
	_state = _new_state;
	_state_seq2++;
	_state_sync.post ();
	if (_state == PROCESS) init_process ();
    }
    if (_state < SILENCE) return 0;
    if (_state == TRIGGER && _inp_ports [_triginp])
    {
        p = (float *) jack_port_get_buffer (_inp_ports [_triginp], nframes);
	for (i = 0; i < nframes; i++)
	{
	    if (p [i] > 0.5f)
	    {
		_offset = i;
		_state = PROCESS;
		break;
	    }
	}
    }
    if (_state != PROCESS)
    {
        for (i = 0; i < MAXOUT; i++)
        {
	    if (_out_ports [i])
	    {
                p = (float *) jack_port_get_buffer (_out_ports [i], nframes);
		memset (p, 0, nframes * sizeof (float));
	    }
        }
	return 0;
    }

    act = 0;
    for (i = 0; i < MAXOUT; i++)
    {
        if (_out_ports [i]) act += output (i, nframes);
    }
    for (i = 0; i < MAXINP; i++)
    {
        if (_inp_ports [i]) act += input (i, nframes);
    }

    _fcount += nframes - _offset;
    _offset = 0;
    if (! act) _state = SILENCE;
 
    return 0;
}


int Jsignal::output (int iport, int nframes)
{
    int32_t  d, i, k;
    float    *p, *q;
    Jsdata   *D;

    k = 0;
    q = (float *) jack_port_get_buffer (_out_ports [iport], nframes);
    D = _out_data + iport;
    if (D->_data && ((D->_nloop == -1) || (D->_iloop < D->_nloop)))
    {
        if (_offset)
        {
            memset (q, 0, _offset * sizeof (float));
	    nframes -= _offset;
	    q += _offset;
	}
        k = D->_nskip - D->_iskip;
	if (nframes && (k > 0))
	{
            if (k > nframes) k = nframes;
	    memset (q, 0, k * sizeof (float));
    	    nframes -= k;
	    q += k;
	    D->_iskip += k;
        }
	while (nframes)
	{
	    k = D->_nsamp - D->_isamp;
	    if (k > nframes) k = nframes;
	    d = D->_step;
	    p = D->_data + d * D->_isamp;
	    for (i = 0; i < k; i++) q [i] = p [i * d];
	    nframes -= k;
	    q += k;
	    D->_isamp += k;
	    if (D->_isamp == D->_nsamp)
	    {
		D->_isamp = 0;
		D->_iloop += 1;
		if (D->_iloop == D->_nloop) break;
	    }
	}
	k = 1;
    }
    if (nframes) memset (q, 0, nframes * sizeof (float));
    return k;
}


int Jsignal::input (int iport, int nframes)
{
    int32_t  d, i, k;
    float    *p, *q;
    Jsdata   *D;

    k = 0;
    p = (float *) jack_port_get_buffer (_inp_ports [iport], nframes);
    D = _inp_data + iport;
    if (D->_data && ((D->_nloop == -1) || (D->_iloop < D->_nloop)))
    {
	if (_offset)
	{
	    nframes -= _offset;
	    p += _offset;
	}
        k = D->_nskip - D->_iskip;
	if (nframes && (k > 0))
	{
            if (k > nframes) k = nframes;
    	    nframes -= k;
	    p += k;
	    D->_iskip += k;
        }
	while (nframes)
        {
            k = D->_nsamp - D->_isamp;
            if (k > nframes) k = nframes;
   	    d = D->_step;
	    q = D->_data + d * D->_isamp;
  	    for (i = 0; i < k; i++) q [i * d] = p [i];
 	    nframes -= k;
	    p += k;
	    D->_isamp += k;
	    if (D->_isamp == D->_nsamp)
	    {
		D->_isamp = 0;
		D->_iloop += 1;
		if (D->_iloop == D->_nloop) break;
	    }
	}
        k = 1;
    }
    return k;
}



void Jsignal::set_trig_inp (int ind)
{
    if ((ind >= 0) && (ind < MAXINP)) _triginp = ind;
}


void Jsignal::set_out_data (int ind, PyObject *V, int bits, int32_t nloop, int32_t nskip)
{
    Jsdata *D;

    if ((_state != SILENCE) || (ind < 0) || (ind >= MAXOUT)) return;
    D = _out_data + ind;
    set_buffer (D, V, bits);
    D->_nloop = nloop;
    D->_nskip = nskip;
}


void Jsignal::set_inp_data (int ind, PyObject *V, int bits, int32_t nloop, int32_t nskip)
{
    Jsdata *D;

    if ((_state != SILENCE) || (ind < 0) || (ind >= MAXINP)) return;
    D = _inp_data + ind;
    set_buffer (D, V, bits);
    D->_nloop = nloop;
    D->_nskip = nskip;
}


void Jsignal::set_buffer (Jsdata *D, PyObject *V, int bits)
{
    if (D->_view.obj) PyBuffer_Release (&(D->_view));
    if (V)
    {
        PyObject_GetBuffer (V, &(D->_view), bits);
        D->_data = (float *) D->_view.buf;
        D->_step = D->_view.strides [0] / sizeof (float); 
        D->_nsamp = D->_view.shape [0];
    }
    else
    {
        D->_data = 0;
    }
}

