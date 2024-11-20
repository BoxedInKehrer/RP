// ----------------------------------------------------------------------------
//
//  Copyright (C) 2010-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __JAMBPAN_H
#define __JAMBPAN_H


#include <zita-jclient.h>
#include "ambpan4.h"


class Jambpan : public Jclient
{
public:

    Jambpan (const char *client_name, const char *server_name, int degree);
    virtual ~Jambpan (void);

    void set_direction (float az, float el, float dt);
    
private:

    enum { MAXOUT = 25 };

    void init (int degree);
    void fini (void);
    int  jack_process (int nframes);

    Ambpan4  *_ambpan;
};


#endif
