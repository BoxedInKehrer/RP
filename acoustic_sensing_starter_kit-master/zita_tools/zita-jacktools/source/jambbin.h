// ----------------------------------------------------------------------------
//
//  Copyright (C) 2015-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __JAMBBIN_H
#define __JAMBBIN_H


#include <zita-jclient.h>
#include "ambbin4.h"


class Jambbin : public Jclient
{
public:

    Jambbin (const char *client_name, const char *server_name, int maxlen, int degree);
    virtual ~Jambbin (void);

    Ambbin4 *ambbin (void) const { return _ambbin; }
    
private:

    int  jack_process (int nframes);

    Ambbin4  *_ambbin;
};


#endif
