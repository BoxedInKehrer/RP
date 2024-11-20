// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2014 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __JCONTROL_H
#define __JCONTROL_H


#include <stdint.h>
#include <zita-jclient.h>


class Jcontrol : public Jclient
{
public:

    Jcontrol (const char *client_name, const char *server_name);
    virtual ~Jcontrol (void);

    int        transport_state (void) const { return _transport_state; }
    uint32_t   transport_frame (void) const { return _transport_frame; }
   
    void transport_stop (void)
    {
        if (_client) jack_transport_stop (_client);
    }

    void transport_start (void)
    {
        if (_client) jack_transport_start (_client);
    }

    void transport_locate (uint32_t frame)
    {
        if (_client) jack_transport_locate (_client, frame);
    }

    int connect_ports (const char *srce, const char *dest)
    {
        return _client ? jack_connect (_client, srce, dest) : -1;
    }

    int disconn_ports (const char *srce, const char *dest)
    {
        return _client ? jack_disconnect (_client, srce, dest) : -1;
    }

private:

    enum { STOPPED, PLAYING, SYNCING };

    void jack_shutdown (void);
    int  jack_process (int nframes);

    int              _transport_state;
    uint32_t         _transport_frame;
};


#endif
