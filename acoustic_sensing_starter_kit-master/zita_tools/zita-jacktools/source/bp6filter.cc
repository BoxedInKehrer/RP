// ----------------------------------------------------------------------------
//
//  Copyright (C) 2013-2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include "bp6filter.h"


static const float EPS = 1e-40f;


Bp6filter::Bp6filter (void):
    _param (0)
{
    reset ();
}


Bp6filter::~Bp6filter (void)
{
}


void Bp6filter::reset (void)
{
    memset (_z, 0, 6 * sizeof (double));
}


void Bp6filter::setparam (const Bp6param *P)
{
    if (_param != P)
    {
	_param = P;
	reset ();
    }
}


void Bp6filter::process (int nsamp, float *inp, float *out)
{
    int   i;
    double z0, z1, z2, z3, z4, z5;
    double c0, c1, c2, c3, c4, c5;
    double g, h, x, y;

    if (!_param)
    {
	memset (out, 0, nsamp * sizeof (float));
	return;
    }
    
    z0 = _z [0];
    z1 = _z [1];
    z2 = _z [2];
    z3 = _z [3];
    g = _param->_gain;
    c0 = _param->_coeff [0];
    c1 = _param->_coeff [1];
    c2 = _param->_coeff [2];
    c3 = _param->_coeff [3];
    
    if (_param->_mode == Bp6param::HH)
    {
        for (i = 0; i < nsamp; i++)
        { 
            x = inp [i];
            x -= c0 * z0 + c1 * z1 + EPS;
            z1 += z0;
            z0 += x;
            x -= c2 * z2 + c3 * z3 + EPS;
            z3 += z2;
            z2 += x;
            out [i] = g * x;
        }
    }
    else
    {
        z4 = _z [4];
        z5 = _z [5];
        c4 = _param->_coeff [4];
        c5 = _param->_coeff [5];
        h = (_param->_mode == Bp6param::BBH) ? 0.0 : 2.0;

        for (i = 0; i < nsamp; i++)
        { 
            x = inp [i];
            x -= c0 * z0 + c1 * z1 + EPS;
            y = x + 2 * z0;
            z1 += z0;
            z0 += x;
            y -= c2 * z2 + c3 * z3 + EPS;
            x = y + 2 * z2;
            z3 += z2;
            z2 += y;
            x -= c4 * z4 + c5 * z5 + EPS;
            y = x + h * z4;
            z5 += z4;
            z4 += x;
            out [i] = g * y;
        }
        _z [4] = z4;
        _z [5] = z5;
    }
    _z [0] = z0;
    _z [1] = z1;
    _z [2] = z2;
    _z [3] = z3;
}


