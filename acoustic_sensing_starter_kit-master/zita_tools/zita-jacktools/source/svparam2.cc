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
#include "svparam2.h"


#define FPI 3.141592f
#define EPS 1e-15f


Svparam2::Svparam2 (int mode) :
    _touch0 (0),
    _touch1 (0),
    _bypass (true),
    _state (BYPASS),
    _type (mode),
    _f0 (0.1f),
    _f1 (0.1f),
    _g0 (1.0f),
    _g1 (1.0f)
{
    if (_type >= P1) _s0 = _s1 = 1.0f;
    else             _s0 = _s1 = 0.0f;
    calcpar1 (0, _f0, _g0, _s0);
    reset ();
}


Svparam2::~Svparam2 (void)
{
}


void Svparam2::reset (void)
{
    memset (_z1, 0, sizeof (double) * MAXCH); 
    memset (_z2, 0, sizeof (double) * MAXCH); 
}


void Svparam2::setpars (float f, float g, float s)
{
    if (f < 1e-5f) f = 1e-5f;
    if (f > 0.49f) f = 0.49f;
    if (g > 10.0f) g = 10.0f;
    if (g < 0.10f) g = 0.10f;
    if (_type >= P1)
    {
        if (s > 10.0f) s = 10.0f;
        if (s < 0.10f) s = 0.10f;
    }
    else
    {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
    }
    _f0 = f;
    _g0 = g;
    _s0 = s;
    _touch0++;
}


void Svparam2::prepare (int nsamp)
{
    bool  upd = false;
    float g, f, s;

    if (nsamp)
    {
        if (_touch1 != _touch0)
        {
            f = _f0;
            s = _s0;
            g = _bypass ? 1.0f : _g0;
            if (f != _f1)
            {
                upd = true;
                if      (f > 1.4f * _f1) _f1 *= 1.25f;
                else if (_f1 > 1.4f * f) _f1 /= 1.25f;
                else _f1 = f;
            }
            if (g != _g1)
            {
                upd = true;
                if      (g > 2.5f * _g1) _g1 *= 2.0f;
                else if (_g1 > 2.5f * g) _g1 /= 2.0f;
                else _g1 = g;
	    }
            if (s != _s1)
            {
                upd = true;
		if (_type >= P1)
		{
		    if      (s > 1.4f * _s1) _s1 *= 1.25f;
		    else if (_s1 > 1.4f * s) _s1 /= 1.25f;
		    else _s1 = s;
		}
		else
		{
		    if      (s > _s1 + 0.25f) _s1 += 0.2f;
		    else if (s < _s1 - 0.25f) _s1 -= 0.2f;
		    else _s1 = s;
		}
            }
            if (upd) 
            {
                calcpar1 (nsamp, _f1, _g1, _s1);
                _state = SMOOTH;
            }
            else
            {
                _touch1 = _touch0;
                if (fabs (_g1 - 1) < 0.001f)
                {
                    reset ();
                    _state = BYPASS;
                }
                else if (_state == SMOOTH)   
                {
                    calcpar1 (0, _f1, _g1, _s1);
                    _state = STATIC;
                }
            }
        }
    }
    else
    {
        calcpar1 (0, _f0, _g0, _s0);
        _state = STATIC;
    }
}


void Svparam2::calcpar1 (int nsamp, float f, float g, float s)
{
    float a0, a1, a2, b1, b2, c1, c2, c3, c4;
    float w1, w2, d1, d2, r;
    
    if (_type >= P1) 
    {    
	// Parametric.
        if (_type == P2)
        {
	    if (g < 1) s /= g;
        }
        else if (_type == P3)
        {
	    s *= sqrtf (3.16f / g);
        }
	w1 = tanf (FPI * f);
	g -= 1;
	c1 = w1 * s;
	c2 = w1 * w1;
	d1 = 1 + c1 + c2;
	d2 = c1 + 2 * c2;
	b1 = 2 * d2 / d1;
	b2 = 2 * c2 / d2;
	a2 = 0;
        a1 = g * (1 - b2);
        a0 = a1 * d2 / d1;
    }
    else
    {
	// High or low shelf.
	r = (g < 1) ? 1 / g : g;
	s *= (r - 1) / r;
	if (_type == HS)
	{
	    w1 = tanf (0.63f * FPI * f);
	    w2 = w1 / sqrtf (r);
	}
	else
	{
	    w1 = tanf (1.60f * FPI * f);
	    w2 = w1 * sqrtf (r);
	}
	if (g < 1)
	{
	    c1 = w2 * (2.0f - 1.6f * s);
	    c2 = w2 * w2;
	    c3 = w1 * (2.0f - 0.9f * s);
	    c4 = w1 * w1;
	}
	else
	{
	    c1 = w1 * (2.0f - 0.9f * s);
	    c2 = w1 * w1;
	    c3 = w2 * (2.0f - 1.6f * s);
	    c4 = w2 * w2;
	}
	d1 = 1 + c1 + c2;
	d2 = c1 + 2 * c2;
	b1 = 2 * d2 / d1;
	b2 = 2 * c2 / d2;
	a0 = (1 + c3 + c4) / d1;
	a1 = (c3 + 2 * c4) / d2;
	if (_type == HS)
	{
	    a0 *= g;
	    a1 *= g;
	    a2 = 0;
	}
	else
	{
	    a2 = g - 1;
	}
	a0 -= 1;
	a1 -= 1;
    }

    if (nsamp)
    {
        _da0 = (a0 - _a0) / nsamp;
        _da1 = (a1 - _a1) / nsamp;
        _da2 = (a2 - _a2) / nsamp;
        _db1 = (b1 - _b1) / nsamp;
        _db2 = (b2 - _b2) / nsamp;
    }
    else
    {
        _a0 = a0;
        _a1 = a1;
        _a2 = a2;
        _b1 = b1;
        _b2 = b2;
    }
}


void Svparam2::process1 (int nsamp, int nchan, float *data[])
{
    int     i, j;
    double  a0, a1, a2, b1, b2;
    double  x, y, z1, z2;
    float   *p;

    a0 = _a0;
    a1 = _a1;
    a2 = _a2;
    b1 = _b1;
    b2 = _b2;
    if (_state == SMOOTH)
    {
        for (i = 0; i < nchan; i++)
        {
            p = data [i];
            z1 = _z1 [i];
            z2 = _z2 [i];
            a0 = _a0;
            a1 = _a1;
            a2 = _a2;
            b1 = _b1;
            b2 = _b2;
            for (j = 0; j < nsamp; j++)
            {
                a0 += _da0;
                a1 += _da1;
                a2 += _da2;
                b1 += _db1;
                b2 += _db2;
                x = *p;
                y = x - z1 - z2 + EPS;
                *p++ = x + a0 * y + a1 * z1 + a2 * z2;
                z2 += b2 * z1;
                z1 += b1 * y;
            }
            _z1 [i] = z1;
            _z2 [i] = z2;
        }
        _a0 = a0;
        _a1 = a1;
        _a2 = a2;
        _b1 = b1;
        _b2 = b2;
    }
    else
    {
        for (i = 0; i < nchan; i++)
        {
            p = data [i];
            z1 = _z1 [i];
            z2 = _z2 [i];
            for (j = 0; j < nsamp; j++)
            {
                x = *p;
                y = x - z1 - z2 + EPS;
                *p++ = x + a0 * y + a1 * z1 + a2 * z2;
                z2 += b2 * z1;
                z1 += b1 * y;
            }
            _z1 [i] = z1;
            _z2 [i] = z2;
        }
    }
}


