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


#include <string.h>
#include "jcontrol.h"


Jcontrol::Jcontrol (const char *client_name, const char *server_name) :
    _transport_state (0),
    _transport_frame (0)
{
    if (open_jack (client_name, server_name, 0, 0)) _state = FAILED;
    else _state = PROCESS;
}


Jcontrol::~Jcontrol (void)
{ 
    close_jack ();
}


void Jcontrol::jack_shutdown (void)
{
    _state = ZOMBIE;
}


int Jcontrol::jack_process (int nframes)
{
    jack_position_t         posit;
    jack_transport_state_t  state;

    state = jack_transport_query (_client, &posit);
    switch (state)
    {
    case JackTransportStopped: _transport_state = STOPPED; break;
    case JackTransportRolling: _transport_state = PLAYING; break;
    default: _transport_state = SYNCING;
    }
    _transport_frame = posit.frame;
    return 0;
}

