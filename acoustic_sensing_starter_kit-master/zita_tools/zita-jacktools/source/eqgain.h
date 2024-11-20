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


#ifndef __EQGAIN_H
#define __EQGAIN_H


#include <stdint.h>


class Eqgain
{
public:

    Eqgain (void);
    ~Eqgain (void);
    
    void setgain (float g)
    {
	_g0 = g;
	_touch0++;
    }
    void bypass (bool s)
    {
	if (s != _bypass)
	{
	    _bypass = s;
	    _touch0++;
	}
    }
    void prepare (int nsamp);
    void process (int nsamp, int nchan, float *inp[], float *out[]);

private:

    enum { BYPASS, STATIC, SMOOTH };

    volatile int16_t  _touch0;
    volatile int16_t  _touch1;
    bool              _bypass;
    int               _state;
    float             _g0, _g1;
    float             _g, _dg;
};


#endif
