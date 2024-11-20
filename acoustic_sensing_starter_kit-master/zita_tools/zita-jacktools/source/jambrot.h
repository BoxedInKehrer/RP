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


#ifndef __JAMBROT_H
#define __JAMBROT_H


#include <zita-jclient.h>
#include "ambrot4.h"


class Jambrot : public Jclient
{
public:

    Jambrot (const char *client_name, const char *server_name, int degree);
    virtual ~Jambrot (void);

    void set_quaternion (float w, float x, float y, float z, float t);
    
private:

    enum { MAXINP = 25, MAXOUT = 25 };

    void init (int degree);
    void fini (void);
    int  jack_process (int nframes);

    Ambrot4  *_ambrot;
};


#endif
