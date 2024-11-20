// ----------------------------------------------------------------------------
//
//  Copyright (C) 2011..2018 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __JNMETER_H
#define __JNMETER_H


#include <zita-jclient.h>
#include "nmeterdsp.h"


class Jnmeter : public Jclient
{
public:

    Jnmeter (const char *client_name, const char *server_name,
	     int ninp, int nout, float *levels);
    virtual ~Jnmeter (void);

    enum { MAXINP = 64, MAXOUT = 64 };

    int set_input (int inp, int out);
    int set_filter (int out, int ftype, int dcfilt);
    int set_detect (int out, int dtype);
    int get_levels (void);

private:

    int jack_process (int nframes);

    Nmeterdsp       *_nprocs;
    float           *_levels;
    int              _inpmap [MAXOUT];
};


#endif
