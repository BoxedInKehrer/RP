// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __JCLIENT_H
#define __JCLIENT_H


#include <jack/jack.h>


class Jclient
{
public:

    Jclient (void);
    virtual ~Jclient (void);

    enum { INITIAL, PASSIVE, SILENCE, PROCESS = 10, FAILED = -1, ZOMBIE = -2 };

    int get_state (void) const { return _state; }
    
    const char *jack_name (void) const { return _jack_name; }
    int jack_rate (void) const { return _jack_rate; }
    int jack_size (void) const { return _jack_size; }

    int create_inp_ports (const char *form, int offs = 0);
    int create_out_ports (const char *form, int offs = 0);
    int port_operation (int opc, int ind, const char *name);

    int create_inp_port (int i, const char *name);
    int create_out_port (int i, const char *name);
    int rename_inp_port (int i, const char *name);
    int rename_out_port (int i, const char *name);
    int delete_inp_port (int i);
    int delete_out_port (int i);
    int connect_inp_port (int i, const char *srce);
    int connect_out_port (int i, const char *dest);
    int disconn_inp_port (int i, const char *srce);
    int disconn_out_port (int i, const char *dest);

    int ninp (void) const { return _ninp; }
    int nout (void) const { return _nout; }
    
protected:

    int open_jack (const char *client_name, const char *server_name, int max_inps, int max_outs);
    int close_jack (void);
    int delete_port (int i, int nport, jack_port_t **ports);
    int disconn_port (int i, int nport, jack_port_t **ports, const char *srce, const char *dest);
    
    virtual void jack_shutdown (void);
    virtual void jack_freewheel (int state); 
    virtual int  jack_buffsize (int nframes);
    virtual int  jack_process (int nframes) = 0;

    jack_client_t   *_client;
    int              _state;
    int              _ninp;
    int              _nout;
    const char      *_jack_name;
    int              _jack_rate;
    int              _jack_size;
    jack_port_t    **_inp_ports;
    jack_port_t    **_out_ports;
    int              _schedpol;
    int              _priority;

private:
    
    void cleanup (void);

    static void jack_static_shutdown (void *arg);
    static void jack_static_freewheel (int state, void *arg);
    static int  jack_static_buffsize (jack_nframes_t nframes , void *arg);
    static int  jack_static_process (jack_nframes_t nframes, void *arg);
};


#endif