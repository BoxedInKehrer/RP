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


#ifndef __JPARAMEQ_H
#define __JPARAMEQ_H


#include <zita-jclient.h>
#include "eqgain.h"
#include "svparam2.h"


class Jparameq : public Jclient
{
public:

    Jparameq (const char *client_name, const char *server_name,
	      int nchan, const char *types);
    virtual ~Jparameq (void);

    enum { MAXCHAN = Svparam2::MAXCH, MAXSECT = 8 };

    void set_filter (int sect, float freq, float gain, float shape);
    void set_bypass (bool act);
    void set_gain (float gain);

private:

    void init (const char *types);
    void fini ();
    int  jack_process (int nframes);

    int              _frag;  
    int              _todo;
    int              _nsect;
    Eqgain           _eqgain;
    Svparam2        *_filters [MAXSECT];
};


#endif
