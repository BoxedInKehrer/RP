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


#ifndef __JIECFILT_H
#define __JIECFILT_H


#include <zita-jclient.h>
#include "iecfilt.h"


class Jiecfilt : public Jclient
{
public:

    Jiecfilt (const char *client_name, const char *server_name,
	      int ninp, int nout);
    virtual ~Jiecfilt (void);

    enum {  MAXINP = 64, MAXOUT = 64 };

    void set_filter (int inp, int out, int type, int band);

private:

    void init ();
    void fini ();
    int  jack_process (int nframes);

    Bp6paramset     *_oct1set;
    Bp6paramset     *_oct3set;
    Bp6param        *_newpars [MAXOUT];
    Bp6filter        _filters [MAXOUT];
    int              _inpmap [MAXOUT];
};


#endif
