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


#ifndef __JGAINCTL_H
#define __JGAINCTL_H


#include <zita-jclient.h>


class Jgainctl : public Jclient
{
public:

    Jgainctl (const char *client_name, const char *server_name, int nchan);
    virtual ~Jgainctl (void);

    enum { MAXCHAN = 64 };

    void set_gain (float g, float v);
    void set_muted (bool s)
    {
	if (_muted != s)
	{
	    _muted = s;
	    _touch0++;
	}
    }
    
private:

    int jack_process (int nframes);

    volatile int16_t  _touch0;   
    volatile int16_t  _touch1;
    volatile float    _gain0;  
    volatile float    _gain1;
    volatile float    _vgain;
    float             _dgain;
    bool              _muted;
    int               _nfrag;
    float             _g;
};


#endif
