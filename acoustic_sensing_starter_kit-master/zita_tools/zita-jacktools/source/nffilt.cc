//  ------------------------------------------------------------------------
//
//  Copyright (C) 2006-2018 Fons Adriaensen <fons@linuxaudio.org>
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
//  ------------------------------------------------------------------------


#include "nffilt.h"


#define EPS 1e-25f


void NF_filt1::init (float w)
{
    float b1;

    b1 = 0.5f * w;
    _g = 1 / (1 + b1);
    _d1 = _g * (2 * b1);
}


void NF_filt1::process (int n, float *p)
{
    float x, z1;

    z1 = _z1;
    while (n--)
    {
	x = *p - _d1 * z1 + EPS;
        z1 += x;
        *p++ = _g * x;
    }
    _z1 = z1;
}


void NF_filt2::init (float w)
{
    float r1, r2, b1, b2;

    r1 = 0.5f * w;
    r2 = r1 * r1;
    b1 = 3.0f * r1;
    b2 = 3.0f * r2;         
    _g = 1 / (1 + b1 + b2);
    _d1 = _g * (2 * b1 + 4 * b2);
    _d2 = _g * (4 * b2);
}


void NF_filt2::process (int n, float *p)
{
    float x, z1, z2;

    z1 = _z1;
    z2 = _z2;
    while (n--)
    {
	x = *p - _d1 * z1 - _d2 * z2 + EPS;
        z2 += z1;
        z1 += x;
        *p++ = _g * x;
    }
    _z1 = z1;
    _z2 = z2;
}


void NF_filt3::init (float w)
{
    float r1, r2, b1, b2, g1, g2;

    r1 = 0.5f * w;
    r2 = r1 * r1;
    b1 = 3.6778f * r1;
    b2 = 6.4595f * r2;         
    g2 = 1 + b1 + b2;
    _d1 = (2 * b1 + 4 * b2) / g2;
    _d2 = (4 * b2) / g2;
    b1 = 2.3222f * r1;
    g1 = 1 + b1;
    _d3 = (2 * b1) / g1;
    _g = 1 / (g1 * g2);
}


void NF_filt3::process (int n, float *p)
{
    float x, z1, z2, z3;

    z1 = _z1;
    z2 = _z2;
    z3 = _z3;
    while (n--)
    {
	x = *p - _d1 * z1 - _d2 * z2 + EPS;
        z2 += z1;
        z1 += x;
        x -= _d3 * z3 - EPS;
        z3 += x;
        *p++ = _g * x;
    }
    _z1 = z1;
    _z2 = z2;
    _z3 = z3;
}


void NF_filt4::init (float w)
{
    float r1, r2, b1, b2, g;

    r1 = 0.5f * w;
    r2 = r1 * r1;
    b1 =  4.2076f * r1;
    b2 = 11.4878f * r2;         
    g = 1 / (1 + b1 + b2);
    _d1 = g * (2 * b1 + 4 * b2);
    _d2 = g * (4 * b2);
    _g = g;
    b1 = 5.7924f * r1;
    b2 = 9.1401f * r2;         
    g = 1 / (1 + b1 + b2);
    _d3 = g * (2 * b1 + 4 * b2);
    _d4 = g * (4 * b2);
    _g *= g;
}


void NF_filt4::process (int n, float *p)
{
    float x, z1, z2, z3, z4;

    z1 = _z1;
    z2 = _z2;
    z3 = _z3;
    z4 = _z4;
    while (n--)
    {
	x = *p - _d1 * z1 - _d2 * z2 + EPS;
        z2 += z1;
        z1 += x;
        x -= _d3 * z3 + _d4 * z4 - EPS;
        z4 += z3;
        z3 += x;
        *p++ = _g * x;
    }
    _z1 = z1;
    _z2 = z2;
    _z3 = z3;
    _z4 = z4;
}

