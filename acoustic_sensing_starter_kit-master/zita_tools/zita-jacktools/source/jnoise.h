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


#ifndef __JNOISE_H
#define __JNOISE_H


#include <zita-jclient.h>
#include "noisegen.h"


class Jnoise : public Jclient
{
public:

    Jnoise (const char *client_name, const char *server_name, int nchan);
    virtual ~Jnoise (void);

    enum {  MAXCHAN = 64 };

    void set_output (int chan, int type, float level);

private:

    void init ();
    void fini ();
    int  jack_process (int nframes);

    Noisegen  *_noisegen;
};


#endif
