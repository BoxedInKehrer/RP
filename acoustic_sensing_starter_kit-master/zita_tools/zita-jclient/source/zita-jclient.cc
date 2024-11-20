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


#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "zita-jclient.h"


Jclient::Jclient (void) :
    _inp_ports (0),
    _out_ports (0)
{
    cleanup ();
}


Jclient::~Jclient (void)
{
    close_jack ();
}


void Jclient::cleanup (void)
{
    _client = 0;
    _state = INITIAL;
    _ninp = 0;
    _nout = 0;
    _jack_name = 0;
    _jack_rate = 0;
    _jack_size = 0;
    delete[] _inp_ports;
    delete[] _out_ports;
    _inp_ports = 0;
    _out_ports = 0;
}


int Jclient::open_jack (const char *client_name, const char *server_name, int ninp, int nout)
{
    int                 jack_opts;
    jack_status_t       jack_stat;
    struct sched_param  sched_par;
    
    if (_client) return 1;

    jack_opts = JackNoStartServer;
    if (server_name) jack_opts |= JackServerName;
    if ((_client = jack_client_open (client_name, (jack_options_t)jack_opts, &jack_stat, server_name)) == 0)
    {
        return 1;
    }

    jack_set_process_callback (_client, jack_static_process, (void *) this);
    jack_on_shutdown (_client, jack_static_shutdown, (void *) this);

    if (jack_activate (_client)) 
    {
        jack_client_close (_client);
	_client = 0;
	return 1;
    }
    _jack_name = jack_get_client_name (_client);
    _jack_rate = jack_get_sample_rate (_client);
    _jack_size = jack_get_buffer_size (_client);

    pthread_getschedparam (jack_client_thread_id (_client), &_schedpol, &sched_par);
    _priority = sched_par.sched_priority;

    _ninp = ninp;
    if (ninp)
    {
        _inp_ports = new jack_port_t * [ninp];
        memset (_inp_ports, 0, ninp * sizeof (jack_port_t *));
    }
    _nout = nout;
    if (nout)
    {
        _out_ports = new jack_port_t * [nout];
        memset (_out_ports, 0, nout * sizeof (jack_port_t *));
    }

    _state = PASSIVE;
    return 0;
}


int Jclient::close_jack (void)
{
    if (_client)
    {
       jack_deactivate (_client);
       jack_client_close (_client);
    }
    cleanup ();
    return 0;
}


void Jclient::jack_shutdown (void)
{
    _state = ZOMBIE;
}


void Jclient::jack_freewheel (int state)
{
}


int Jclient::jack_buffsize (int nframes)
{
    return 0;
}


int Jclient::create_inp_ports (const char *form, int offs)
{
    int  i, rv;
    char s [64];

    rv = 0;
    for (i = 0; i < _ninp; i++)
    {
	snprintf (s, 64, form, i + offs);
	rv |= create_inp_port (i, s);
    }
    return rv;
}


int Jclient::create_out_ports (const char *form, int offs)
{
    int  i, rv;
    char s [64];

    rv = 0;
    for (i = 0; i < _nout; i++)
    {
	snprintf (s, 64, form, i + offs);
	rv |= create_out_port (i, s);
    }
    return rv;
}


int Jclient::port_operation (int opc, int ind, const char *name)
{
    switch (opc)
    {
    case 0:  return create_inp_port (ind, name);
    case 1:  return create_out_port (ind, name);
    case 2:  return delete_inp_port (ind);
    case 3:  return delete_out_port (ind);
    case 4:  return rename_inp_port (ind, name);
    case 5:  return rename_out_port (ind, name);
    case 6:  return connect_inp_port (ind, name);
    case 7:  return connect_out_port (ind, name);
    case 8:  return disconn_inp_port (ind, name);
    case 9:  return disconn_out_port (ind, name);
    }
    return 1;
}

    
int Jclient::create_inp_port (int i, const char *name)
{
    if ((_state != PASSIVE) || (i < 0) || (i >= _ninp) || _inp_ports [i]) return 1;
    _inp_ports [i] = jack_port_register (_client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    return _inp_ports [i] ? 0 : 1;
}


int Jclient::create_out_port (int i, const char *name)
{
    if ((_state != PASSIVE) || (i < 0) || (i >= _nout) || _out_ports [i]) return 1;
    _out_ports [i] = jack_port_register (_client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    return _out_ports [i] ? 0 : 1;
}


int Jclient::delete_inp_port (int i)
{
    return delete_port (i, _ninp, _inp_ports);
}


int Jclient::delete_out_port (int i)
{
    return delete_port (i, _nout, _out_ports);
}


int Jclient::rename_inp_port (int i, const char *name)
{
    if ((_state < 0) || (i < 0) || (i >= _ninp) || !_inp_ports [i]) return 1;
    return jack_port_rename (_client, _inp_ports [i], name);
}


int Jclient::rename_out_port (int i, const char *name)
{
    if ((_state < 0) || (i < 0) || (i >= _nout) || !_out_ports [i]) return 1;
    return jack_port_rename (_client, _out_ports [i], name);
}


int Jclient::delete_port (int i, int nport, jack_port_t **ports)
{
    if ((_state != PASSIVE) || (i < -1) || (i >= nport)) return 1;
    if (i == -1)
    {
	for (i = 0; i < nport; i++)
        {
            if (ports [i])
	    {
                jack_port_unregister (_client, ports [i]);
	        ports [i] = 0;
	    }
	}
    }
    else
    {
	if (! ports [i]) return 1;
        jack_port_unregister (_client, ports [i]);
        ports [i] = 0;
    }
    return 0;
}


int Jclient::connect_inp_port (int i, const char *srce)
{
    int rv;

    if ((i < 0) || (i >= _ninp) || !_inp_ports [i]) return -1;
    rv = jack_connect (_client, srce, jack_port_name (_inp_ports [i]));
    if (rv == EEXIST) rv = 0;
    return rv;
}


int Jclient::connect_out_port (int i, const char *dest)
{
    int rv;

    if ((i < 0) || (i >= _nout) || !_out_ports [i]) return -1;
    rv = jack_connect (_client, jack_port_name (_out_ports [i]), dest);
    if (rv == EEXIST) rv = 0;
    return rv;
}


int Jclient::disconn_inp_port (int i, const char *srce)
{
    return disconn_port (i, _ninp, _inp_ports, srce, 0);
}    


int Jclient::disconn_out_port (int i, const char *dest)
{
    return disconn_port (i, _nout, _out_ports, 0, dest);
}    


int Jclient::disconn_port (int i, int nport, jack_port_t **ports, const char *srce, const char *dest)
{
    if ((i < -1) || (i >= nport)) return -1;
    if (i == -1)
    {
        for (i = 0; i < nport; i++)
        {
            if (ports [i]) jack_port_disconnect (_client, ports [i]);
	}
    }
    else
    {
        if (! ports [i]) return -1;
        if      (srce) jack_disconnect (_client, srce, jack_port_name (ports [i]));
        else if (dest) jack_disconnect (_client, jack_port_name (ports [i]), dest);
        else jack_port_disconnect (_client, ports [i]);
    }
    return 0;
}


void Jclient::jack_static_shutdown (void *arg)
{
    ((Jclient *) arg)->jack_shutdown ();
}


void Jclient::jack_static_freewheel (int state, void *arg)
{
    ((Jclient *) arg)->jack_freewheel (state);
}


int Jclient::jack_static_buffsize (jack_nframes_t nframes, void *arg)
{
    return ((Jclient *) arg)->jack_buffsize (nframes);
}


int Jclient::jack_static_process (jack_nframes_t nframes, void *arg)
{
    return ((Jclient *) arg)->jack_process (nframes);
}


