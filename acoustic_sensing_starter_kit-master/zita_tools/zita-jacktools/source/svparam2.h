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


#ifndef __SVPARAM2_H
#define __SVPARAM2_H


#include <stdint.h>


class Svparam2
{
public:

    enum { LS, HS, P1, P2, P3 };
    enum { MAXCH = 64 };
    
    Svparam2 (int type);
    ~Svparam2 (void);
    
    void setpars (float f, float g, float s);
    void reset (void);
    void bypass (bool s)
    {
	if (s != _bypass)
	{
	    _bypass = s;
	    _touch0++;
	}
    }
    void prepare (int nsamp);
    void process (int nsamp, int nchan, float *data[])
    {
	if (_state != BYPASS) process1 (nsamp, nchan, data); 
    }

private:

    enum { BYPASS, STATIC, SMOOTH };
    
    void calcpar1 (int nsamp, float f, float g, float s);
    void process1 (int nsamp, int nchan, float *data[]);

    volatile int16_t  _touch0;
    volatile int16_t  _touch1;
    bool              _bypass;
    int               _state;
    int               _type;
    float             _f0, _f1;
    float             _g0, _g1;
    float             _s0, _s1;
    float             _a0, _a1, _a2;
    float             _b1, _b2;
    float             _da0, _da1, _da2;
    float             _db1, _db2;
    double            _z1 [MAXCH];
    double            _z2 [MAXCH];
};


#endif
