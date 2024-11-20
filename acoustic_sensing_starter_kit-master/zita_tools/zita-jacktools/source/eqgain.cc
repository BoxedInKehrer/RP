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


#include <math.h>
#include <string.h>
#include "eqgain.h"


Eqgain::Eqgain (void) :
    _touch0 (0),
    _touch1 (0),
    _bypass (true),
    _state (BYPASS),
    _g0 (1),
    _g1 (1),
    _g  (1),
    _dg (0)
{
}


Eqgain::~Eqgain (void)
{
}


void Eqgain::prepare (int nsamp)
{
    float g;
    
    if (_touch1 != _touch0)
    {
        g = _bypass ? 1.0f : _g0;
        if (g != _g1)
	{
            if      (g > 2.5f * _g1) _g1 *= 2.0f;
            else if (_g1 > 2.5f * g) _g1 /= 2.0f;
	    else _g1 = g;
	    _dg = _g1 - _g;
	    if (fabsf (_dg) < 1e-6f)
	    {
		_g = _g1;
                _dg = 0;
		_state = STATIC;
	    }
	    else
	    {
		_dg /= nsamp;
		_state = SMOOTH;
	    }
	}
	else
	{
	    _touch1 = _touch0;
            _state = (fabs (_g - 1) < 0.001f) ? BYPASS : STATIC;
	}
    }
}


void Eqgain::process (int nsamp, int nchan, float *inp[], float *out[])
{
    int     i, j;
    float   g;
    float   *p, *q;

    g = _g;
    for (i = 0; i < nchan; i++)
    {
        p = inp [i];
        q = out [i];
	switch (_state)
	{
	case SMOOTH:
	    g = _g;
	    for (j = 0; j < nsamp; j++)
	    {
		g += _dg;
		*q++ = g * *p++;
	    }
	    break;
	case STATIC:
	    g = _g;
	    for (j = 0; j < nsamp; j++)
	    {
		*q++ = g * *p++;
	    }
	    break;
	case BYPASS:
	    if (p != q) memcpy (q, p, nsamp * sizeof (float));
	    break;
	}
    }
    if (_state == SMOOTH) _g = g;
}
