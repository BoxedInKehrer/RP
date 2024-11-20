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


#ifndef __JPEAKLIM_H
#define __JPEAKLIM_H


#include <zita-jclient.h>
#include "peaklim.h"


class Jpeaklim : public Jclient
{
public:

    Jpeaklim (const char *client_name, const char *server_name, int nchan);
    virtual ~Jpeaklim (void);

    enum { MAXCHAN = Peaklim::MAXCHAN };

    Peaklim *peaklim (void) { return &_peaklim; }

private:

    int jack_process (int nframes);

    Peaklim   _peaklim;
};


#endif
