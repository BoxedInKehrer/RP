// ----------------------------------------------------------------------------
//
//  Copyright (C) 2012..2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include <stdlib.h>
#include <string.h>
#include "ambpan4.h"


Ambpan4::Ambpan4 (int fsamp, int degree, bool semi) :
    _fsamp (fsamp),
    _touch0 (0),
    _touch1 (0),
    _count (0)
{
    if (degree < 1) degree = 1;
    if (degree > 4) degree = 4;
    init (degree, semi);
}


Ambpan4::~Ambpan4 (void)
{
}


void Ambpan4::init (int degree, bool semi)
{
    _nharm = (degree + 1) * (degree + 1);
    // SN3D coefficients.
    _C [0] = 1.0f; 
    _C [1] = sqrtf (3.0f / 4);
    _C [2] = 0.5f;
    _C [3] = sqrtf (5.0f / 8);
    _C [4] = sqrtf (15.0f / 4);
    _C [5] = sqrtf (3.0f / 8);
    _C [6] = 0.5f;
    _C [7]  = sqrtf (35.0f / 64);
    _C [8]  = sqrtf (35.0f / 8);
    _C [9]  = sqrtf (5.0f / 16);
    _C [10] = sqrtf (5.0f / 8);
    _C [11] = 0.125f;
    // N3D 
    if (! semi)
    {
	_C [0] *= sqrtf (3.0f);
	for (int i = 1; i < 3; i++) _C [i] *= sqrtf (5.0f);
	for (int i = 3; i < 7; i++) _C [i] *= sqrtf (7.0f);
	for (int i = 7; i < 12; i++) _C [i] *= 3.0f;
    }
    encode (0.0f, 0.0f, _G);
}



void Ambpan4::set_direction (float az, float el, float dt)
{
    _az = az * M_PI / 180.0f;
    _el = el * M_PI / 180.0f;
    if (dt < 0.0f) dt = 0.0f;
    if (dt > 1.0f) dt = 1.0f;
    _dt = dt;
    _touch0++;
}


void Ambpan4::process (int nframes, float *inp, float *out[])
{
    int    i, k, k0, nf;
    float  g, d;
    float  *q;
    
    if (_touch1 != _touch0) update ();
    memcpy (out [0], inp, nframes * sizeof (float));
    k0 = 0;
    while (nframes)
    {
	nf = nframes;
	if (_count)
	{
	    if (nf > _count) nf = _count;
	    for (i = 1; i < _nharm; i++)
	    {
		q = out [i] + k0;
		g = _G [i];
		d = (_T [i] - g) / _count;
		for (k = 0; k < nf; k++)
		{
		    g += d;
		    q [k] = g * inp [k];
		}
		_G [i] = g;
	    }
	    _count -= nf;
	}
	else
	{
	    for (i = 1; i < _nharm; i++)
	    {
		q = out [i] + k0;
		g = _G [i];
		for (k = 0; k < nf; k++)
		{
		    q [k] = g * inp [k];
		}
	    }
	}
	k0 += nf;
        inp += nf;
	nframes -= nf;
    }
}


void Ambpan4::update (void)
{
    _count = (int)(floorf (_dt * _fsamp + 0.5f));
    encode (_az, _el, _T);
    if (_count == 0) memcpy (_G, _T, _nharm * sizeof (float));
    _touch1 = _touch0;
}


void Ambpan4::encode (float azim, float elev, float *E)
{
    float t, x1, y1, z1, x2, y2, z2, c2, s2, c3, s3, x4, y4, z4;

    E [0] = 1.0f;
    t = cosf (elev);
    x1 = cosf (azim) * t;
    y1 = sinf (azim) * t;
    z1 = sinf (elev);
    t = _C [0];
    E [1] = t * y1;
    E [2] = t * z1;
    E [3] = t * x1;
    if (_nharm < 9) return;
    
    x2 = x1 * x1;
    y2 = y1 * y1;
    z2 = z1 * z1;
    c2 = x2 - y2;
    s2 = 2 * x1 * y1;
    t = _C [1];
    E [8] = t * c2;
    E [4] = t * s2;
    t *= 2 * z1;
    E [7] = t * x1;
    E [5] = t * y1;
    E [6] = _C [2] * (3 * z2 - 1);
    if (_nharm < 16) return;
    
    c3 = x1 * (x2 - 3 * y2);
    s3 = y1 * (3 * x2 - y2);
    t = _C [3];
    E [15] = t * c3;
    E [ 9] = t * s3;
    t = _C [4] * z1;
    E [14] = t * c2;
    E [10] = t * s2;
    t = _C [5] * (5 * z2 - 1);
    E [13] = t * x1; 
    E [11] = t * y1; 
    E [12] = _C [6] * (5 * z2 - 3) * z1;
    if (_nharm < 25) return;

    x4 = x2 * x2;
    y4 = y2 * y2;
    z4 = z2 * z2;
    t = _C [7];
    E [24] = t * (x4 - 6 * x2 * y2 + y4);
    E [16] = t * 2 * s2 * c2;
    t = _C [8] * z1;
    E [23] = t * c3;
    E [17] = t * s3;
    t = _C [9] * (7 * z2 - 1);
    E [22] = t * c2;
    E [18] = t * s2;
    t = _C [10] * z1 * (7 * z2 - 3);
    E [21] = t * x1;
    E [19] = t * y1;
    E [20] = _C [11] * (35 * z4 - 30 * z2 + 3);
}


