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


#ifndef __JCONVOLV_H
#define __JCONVOLV_H


#include <zita-jclient.h>
#include <zita-convolver.h>


class Jconvolv : public Jclient
{
public:

    enum
    {
	MAXINP = Convproc::MAXINP,
	MAXOUT = Convproc::MAXOUT,
    };
    
    Jconvolv (const char *client_name, const char *server_name, int ninp, int nout);
    virtual ~Jconvolv (void);

    Convproc *convproc (void) const { return (Convproc *) _convproc; }
    void set_state (int state);
    int  configure (uint32_t maxlen, float density);
    
private:

    void jack_freewheel (int state); 
    int  jack_buffsize (int nframes);
    int  jack_process (int nframes);

    Convproc        *_convproc;
    bool             _convsync;
};


#endif
