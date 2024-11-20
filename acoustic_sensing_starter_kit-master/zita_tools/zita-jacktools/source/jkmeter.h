// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2015 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __JKMETER_H
#define __JKMETER_H


#include <zita-jclient.h>
#include "kmeterdsp.h"


class Jkmeter : public Jclient
{
public:

    Jkmeter (const char *client_name, const char *server_name, int ninp, float *rms, float *pks);
    virtual ~Jkmeter (void);

    enum { MAXINP = 64 };

    int get_levels (void);

private:

    int jack_process (int nframes);

    Kmeterdsp       *_kproc;
    float           *_rms;
    float           *_pks;
};


#endif
