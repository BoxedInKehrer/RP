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


#ifndef __JSIGNAL_H
#define __JSIGNAL_H


#include <Python.h>
#include <stdint.h>
#include <zita-jclient.h>
#include "posixthr.h"


class Jsdata
{
public:

    Jsdata (void):
    _data (0),
    _step (0),
    _nsamp (0),
    _nloop (0),
    _nskip (0),
    _isamp (0),	
    _iloop (0),	
    _iskip (0)	
    {
	_view.obj = 0;
    }

    float       *_data;
    int32_t      _step;
    int32_t      _nsamp;
    int32_t      _nloop;
    int32_t      _nskip;
    int32_t      _isamp;
    int32_t      _iloop;
    int32_t      _iskip;
    Py_buffer    _view;
};


class Jsignal : public Jclient
{
public:

    Jsignal (const char *client_name, const char *server_name);
    virtual ~Jsignal (void);

    enum { MAXINP = 64, MAXOUT = 64, TRIGGER = 9 };

    void set_state (int state);
    void set_out_data (int ind, PyObject *V, int bits, int32_t nloop, int32_t nskip);
    void set_inp_data (int ind, PyObject *V, int bits, int32_t nloop, int32_t nskip);
    void set_trig_inp (int ind);
    int64_t get_posit (void) const { return _fcount; }

private:

    void init ();
    void fini ();
    int  jack_process (int nfram);
    void init_process (void);
    int  output (int iport, int nframes);
    int  input (int iport, int nframes);
    void set_buffer (Jsdata *D, PyObject *V, int bits);

    volatile int     _new_state;
    volatile int     _state_seq1;
    volatile int     _state_seq2;
    P_sema           _state_sync;
    int              _triginp;
    int              _offset;
    int64_t          _fcount;
    Jsdata           _out_data [MAXOUT];
    Jsdata           _inp_data [MAXINP];
};


#endif
