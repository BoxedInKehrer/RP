// ----------------------------------------------------------------------------
//
//  Copyright (C) 2012-2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include "ambrot4.h"


float Ambrot4::R2 [3];
float Ambrot4::U2 [3];
float Ambrot4::V2 [3];
float Ambrot4::W2 [3];

float Ambrot4::R3 [4];
float Ambrot4::U3 [4];
float Ambrot4::V3 [4];
float Ambrot4::W3 [4];

float Ambrot4::R4 [5];
float Ambrot4::U4 [5];
float Ambrot4::V4 [5];
float Ambrot4::W4 [5];

bool Ambrot4::initdone = false;



Ambrot4::Ambrot4 (int fsamp, int degree) :
    _fsamp (fsamp),
    _touch0 (0),
    _touch1 (0),
    _count (0)
{
    if (degree < 1) degree = 1;
    if (degree > 4) degree = 4;
    init (degree);
}


Ambrot4::~Ambrot4 (void)
{
}


void Ambrot4::init (int degree)
{
    _nharm = (degree + 1) * (degree + 1);
    // Init rotation matrices.
    memset (_C1, 0,  9 * sizeof (float));
    memset (_C2, 0, 25 * sizeof (float));
    memset (_C3, 0, 49 * sizeof (float));
    memset (_C4, 0, 81 * sizeof (float));
    for (int i = 0; i < 3; i++) _C1 [i][i] = 1.0f;
    for (int i = 0; i < 5; i++) _C2 [i][i] = 1.0f;
    for (int i = 0; i < 7; i++) _C3 [i][i] = 1.0f;
    for (int i = 0; i < 9; i++) _C4 [i][i] = 1.0f;
    // Init magic constants.
    if (! initdone)
    {
	initconst (2, R2, U2, V2, W2);
	initconst (3, R3, U3, V3, W3);
	initconst (4, R4, U4, V4, W4);
	initdone = true;
    }
}


void Ambrot4::initconst (int d, float *R, float *U, float *V, float *W)
{
    int i;

    for (i = 0; i <= d; i++)
    {
	if (i < d)
	{
	    R [i] = sqrtf (d * d - i * i);
	    U [i] = R [i];
	}
	else
	{
	    R [i] = sqrtf (2 * d * (2 * d - 1));
	    U [i] = 0.0f;
	}
	if (i > 0)
	{
  	    V [i] = sqrtf ((d + i) * (d + i - 1) / 4.0f);
 	    W [i] = sqrtf ((d - i) * (d - i - 1) / 4.0f);
	}
	else
	{
  	    V [i] = -sqrtf (d * (d - 1) / 2.0f);
 	    W [i] = 0.0f;
	}
    }
}


void Ambrot4::set_quaternion (float w, float x, float y, float z, float t)
{
    float m;

    _mutex.lock ();
    m = sqrtf (w * w + x * x + y * y + z * z);
    _w = w / m;
    _x = x / m;
    _y = y / m;
    _z = z / m;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    _t = t;
    _touch0++;
    _mutex.unlock ();
}


void Ambrot4::process (int nframes, float *inp[], float *out[])
{
    int  k0, nf;

    if (_touch1 != _touch0) update ();
    memcpy (out [0], inp [0], nframes * sizeof (float));
    k0 = 0;
    while (nframes)
    {
	nf = nframes;
	if (_count)
	{
	    if (nf > _count) nf = _count;
	    process1 (k0, nf, inp, out);
	    _count -= nf;
        }
	else
	{
	    process0 (k0, nf, inp, out);
	}
	nframes -= nf;
	k0 += nf;
    }
}


void Ambrot4::process0 (int k0, int nf, float *inp[], float *out[])
{
    int    i, j, k;
    float  c;
    float  *p, *q;
   
    for (i = 0; i < 3; i++)
    {
	q = out [1 + i] + k0;
	p = inp [1] + k0;
	c = _C1 [i][0];
	for (k = 0; k < nf; k++) q [k] = c * p [k];
	for (j = 1; j < 3; j++)
	{
	    p = inp [1 + j] + k0;
	    c = _C1 [i][j];
	    for (k = 0; k < nf; k++) q [k] += c * p [k];
	}
    }
    if (_nharm < 9) return;

    for (i = 0; i < 5; i++)
    {
	q = out [4 + i] + k0;
	p = inp [4] + k0;
	c = _C2 [i][0];
	for (k = 0; k < nf; k++) q [k] = c * p [k];
	for (j = 1; j < 5; j++)
	{
	    p = inp [4 + j] + k0;
	    c = _C2 [i][j];
	    for (k = 0; k < nf; k++) q [k] += c * p [k];
	}
    }
    if (_nharm < 16) return;

    for (i = 0; i < 7; i++)
    {
	q = out [9 + i] + k0;
	p = inp [9] + k0;
	c = _C3 [i][0];
	for (k = 0; k < nf; k++) q [k] = c * p [k];
	for (j = 1; j < 7; j++)
	{
	    p = inp [9 + j] + k0;
	    c = _C3 [i][j];
	    for (k = 0; k < nf; k++) q [k] += c * p [k];
	}
    }
    if (_nharm < 25) return;

    for (i = 0; i < 9; i++)
    {
	q = out [16 + i] + k0;
	p = inp [16] + k0;
	c = _C4 [i][0];
	for (k = 0; k < nf; k++) q [k] = c * p [k];
	for (j = 1; j < 9; j++)
	{
	    p = inp [16 + j] + k0;
	    c = _C4 [i][j];
	    for (k = 0; k < nf; k++) q [k] += c * p [k];
	}
    }
}


void Ambrot4::process1 (int k0, int nf, float *inp[], float *out[])
{
    int    i, j, k;
    float  c, d;
    float  *p, *q;
    
    for (i = 0; i < 3; i++)
    {
	q = out [1 + i] + k0;
	memset (q, 0, nf * sizeof (float));
	for (j = 0; j < 3; j++)
	{
	    p = inp [1 + j] + k0;
	    c = _C1 [i][j];
	    d = (_M1 [i][j] - c) / _count;
	    for (k = 0; k < nf; k++)
	    {
		c += d;
		q [k] += c * p [k];
	    }
	    _C1 [i][j] = c;
	}
    }
    if (_nharm < 9) return;

    for (i = 0; i < 5; i++)
    {
	q = out [4 + i] + k0;
	memset (q, 0, nf * sizeof (float));
	for (j = 0; j < 5; j++)
	{
	    p = inp [4 + j] + k0;
	    c = _C2 [i][j];
	    d = (_M2 [i][j] - c) / _count;
	    for (k = 0; k < nf; k++)
	    {
		c += d;
		q [k] += c * p [k];
	    }
	    _C2 [i][j] = c;
	}
    }
    if (_nharm < 16) return;

    for (i = 0; i < 7; i++)
    {
	q = out [9 + i] + k0;
	memset (q, 0, nf * sizeof (float));
	for (j = 0; j < 7; j++)
	{
	    p = inp [9 + j] + k0;
	    c = _C3 [i][j];
	    d = (_M3 [i][j] - c) / _count;
	    for (k = 0; k < nf; k++)
	    {
		c += d;
		q [k] += c * p [k];
	    }
	    _C3 [i][j] = c;
	}
    }
    if (_nharm < 25) return;

    for (i = 0; i < 9; i++)
    {
	q = out [16 + i] + k0;
	memset (q, 0, nf * sizeof (float));
	for (j = 0; j < 9; j++)
	{
	    p = inp [16 + j] + k0;
	    c = _C4 [i][j];
	    d = (_M4 [i][j] - c) / _count;
	    for (k = 0; k < nf; k++)
	    {
		c += d;
		q [k] += c * p [k];
	    }
	    _C4 [i][j] = c;
	}
    }	
}


void Ambrot4::update (void)
{
    if (_mutex.trylock ()) return;
    matrix1 ();
    _count = (int)(floorf (_t * _fsamp + 0.5f));
    _touch1 = _touch0;
    _mutex.unlock ();
    if (!_count) memcpy (_C1, _M1, 9 * sizeof (float));
    if (_nharm < 9) return;

    matrix2 ();
    if (!_count) memcpy (_C2, _M2, 25 * sizeof (float));
    if (_nharm < 16) return;

    matrix3 ();
    if (!_count) memcpy (_C3, _M3, 49 * sizeof (float));
    if (_nharm < 25) return;

    matrix4 ();
    if (!_count) memcpy (_C4, _M4, 81 * sizeof (float));
}


void Ambrot4::matrix1 (void)
{
    float xx, yy, zz, wx, wy, wz, xy, xz, yz;
    
    xx = _x * _x;
    yy = _y * _y;
    zz = _z * _z;
    wx = _w * _x;
    wy = _w * _y;
    wz = _w * _z;
    xy = _x * _y;
    xz = _x * _z;
    yz = _y * _z;
   
    _M1 [0][0] = 1 - 2 * (xx + zz);
    _M1 [0][1] = 2 * (yz - wx);
    _M1 [0][2] = 2 * (xy + wz);
    _M1 [1][0] = 2 * (yz + wx);
    _M1 [1][1] = 1 - 2 * (xx + yy);
    _M1 [1][2] = 2 * (xz - wy);
    _M1 [2][0] = 2 * (xy - wz);
    _M1 [2][1] = 2 * (xz + wy);
    _M1 [2][2] = 1 - 2 * (yy + zz);
}

    
void Ambrot4::matrix2 (void)
{
    int    k, m, n;
    float  s, u, v;

    for (m = -2; m <= 2; m++)
    {
	k = abs (m);
        u = U2 [k];
	v = V2 [k];
        for (n = -2; n <= 2; n++)
	{
   	    s = v * funcV (k, m, n);
	    if (u != 0) s += u * funcU (k, m, n);
	    _M2 [m + 2][n + 2] = s / R2 [abs (n)];
	}
    }
}


void Ambrot4::matrix3 (void)
{
    int    k, m, n;
    float  s, u, v, w;

    for (m = -3; m <= 3; m++)
    {
	k = abs (m);
        u = U3 [k];
	v = V3 [k];
	w = W3 [k];
        for (n = -3; n <= 3; n++)
	{
   	    s = v * funcV (k, m, n);
	    if (u != 0) s += u * funcU (k, m, n);
	    if (w != 0) s -= w * funcW (k, m, n);
	    _M3 [m + 3][n + 3] = s / R3 [abs (n)];
	}
    }
}


void Ambrot4::matrix4 (void)
{
    int    k, m, n;
    float  s, u, v, w;

    for (m = -4; m <= 4; m++)
    {
	k = abs (m);
        u = U4 [k];
	v = V4 [k];
	w = W4 [k];
        for (n = -4; n <= 4; n++)
	{
   	    s = v * funcV (k, m, n);
	    if (u != 0) s += u * funcU (k, m, n);
	    if (w != 0) s -= w * funcW (k, m, n);
	    _M4 [m + 4][n + 4] = s / R4 [abs (n)];
	}
    }
}


float Ambrot4::funcV (int k, int m, int n)
{
    float p;
    
    if (m > 0)
    {
	m -= 1;
	p = funcP (k, m, n, 1);
        if (m) return p - funcP (k, -m, n, -1);
        else   return sqrtf (2.0f) * p;
    }
    if (m < 0)
    {
	m += 1;
	p = funcP (k, -m, n, -1);        
        if (m) return p + funcP (k, m, n, 1);
        else   return sqrtf (2.0f) * p;
    }
    return funcP (k, 1, n, 1) + funcP (k, -1, n, -1);
}


float Ambrot4::funcW (int k, int m, int n)
{
    if (m > 0)
    {
        m += 1;
        return funcP (k, m, n, 1) + funcP (k, -m, n, -1);       
    }
    if (m < 0)
    {
        m -= 1;
        return funcP (k, m, n, 1) - funcP (k, -m, n, -1);       
    }
    return 0;
}


float Ambrot4::funcP (int k, int m, int n, int i)
{
    i += 1;
    switch (k)
    {
    case 2:
        m += 1;
        if (n == -2) return _M1 [i][0] * _M1 [m][2] + _M1 [i][2] * _M1 [m][0];
        if (n ==  2) return _M1 [i][2] * _M1 [m][2] - _M1 [i][0] * _M1 [m][0];
	return _M1 [i][1] * _M1 [m][n + 1];
    case 3:
        m += 2;
        if (n == -3) return _M1 [i][0] * _M2 [m][4] + _M1 [i][2] * _M2 [m][0];
        if (n ==  3) return _M1 [i][2] * _M2 [m][4] - _M1 [i][0] * _M2 [m][0];
        return _M1 [i][1] * _M2 [m][n + 2];
    case 4:
        m += 3;
        if (n == -4) return _M1 [i][0] * _M3 [m][6] + _M1 [i][2] * _M3 [m][0];
        if (n ==  4) return _M1 [i][2] * _M3 [m][6] - _M1 [i][0] * _M3 [m][0];
        return _M1 [i][1] * _M3 [m][n + 3];
    }
    return 0;
}
	    
