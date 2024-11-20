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


#ifndef __JMATCONV_H
#define __JMATCONV_H


#include <zita-jclient.h>
#include "denseconv.h"


class Jmatconv : public Jclient
{
public:

    enum
    {
	MAXINP = Denseconv::MAXINP,
	MAXOUT = Denseconv::MAXOUT,
	MAXLEN = Denseconv::MAXLEN
    };
    
    Jmatconv (const char *client_name, const char *server_name, int size, int ninp, int nout, int nthr);
    virtual ~Jmatconv (void);

    Denseconv *convproc (void) const { return (Denseconv *) _dconv; }
    void set_state (int state);

private:

    void init (int size, int nthr);
    void fini ();
    int  jack_process (int nframes);

    volatile int     _new_state;
    volatile int     _state_seq1;
    volatile int     _state_seq2;
    P_sema           _state_sync;
    Denseconv       *_dconv;
};


#endif
