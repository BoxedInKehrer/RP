// ------------------------------------------------------------------------
//
//  Copyright (C) 2015 Fons Adriaensen <fons@linuxaudio.org>
//    
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
//
// ------------------------------------------------------------------------


#include <string.h>
#include "nmeterdsp.h"


int Enb20kfilter::init (int fsamp)
{
    reset ();
    switch (fsamp)
    {
    case 44100:
	_g  =  6.17251745e-01f;
	_b1 =  3.03653041e+00f;
	_b2 =  3.55941563e+00f;
	_b3 =  1.89264513e+00f;
	_b4 =  3.87436747e-01f;
	break;
    case 48000:
	_g  =  4.27293435e-01f;
	_b1 =  2.32683301e+00f;
	_b2 =  2.28195320e+00f;
	_b3 =  1.03148006e+00f;
	_b4 =  1.96428697e-01f;
	break;
    case 88200:
	_g  =  4.26385390e-02f;
	_b1 = -1.02651917e+00f;
	_b2 =  1.07245896e+00f;
	_b3 = -4.86158787e-01f;
	_b4 =  1.22435626e-01f;
	break;
    case 96000:
	_g  =  3.14009927e-02f;
	_b1 = -1.32061860e+00f;
	_b2 =  1.29625957e+00f;
	_b3 = -6.18938600e-01f;
	_b4 =  1.45713514e-01f;
        break;
    default:
	_err = true;
        return 1;
    }
    _err = false;
    return 0;
}

void Enb20kfilter::reset (void)
{
    _z1 = _z2 = _z3 = _z4 = 0;
}

void Enb20kfilter::process (int n, const float *inp, float *out)
{
    float x, z1, z2, z3, z4;

    if (_err)
    {
	memset (out, 0, n * sizeof (float));
        return;
    }
    z1 = _z1;
    z2 = _z2;
    z3 = _z3;
    z4 = _z4;
    while (n--)
    {
	x = *inp++ + 1e-25f;
	x -= _b1 * z1 + _b2 * z2 + _b3 * z3 + _b4 * z4;
	*out++ = _g * (x + z4 + 4 * (z1 + z3) + 6 * z2);
	z4 = z3;
	z3 = z2;
	z2 = z1;
	z1 = x;
    }
    _z1 = z1;
    _z2 = z2;
    _z3 = z3;
    _z4 = z4;
}


// ------------------------------------------------------------------------


#define ACW_F1  20.5990
#define ACW_F2  107.652
#define ACW_F3  737.862
#define ACW_F4  12194.2
#define AW_GAIN 1.257
#define CW_GAIN 1.006


int Iec_ACfilter::init (int fsamp)
{
    double f, g;

    reset ();
    _w1 = _w2 = _w3 = _w4 = _ga = _gc = 0;     

    switch (fsamp)
    {
    case 44100:	_w4 = 0.846; break;
    case 48000:	_w4 = 0.817; break;
    case 88200:	_w4 = 0.587; break;
    case 96000:	_w4 = 0.555; break;
    default:
	_err = true;
        return 1;
    }

    f = ACW_F1 / fsamp;
    _w1 = 2 * M_PI * f;
    g = 4 / ((2 - _w1) * (2 - _w1));
    _w1 *= 1 - 3 * f;
    _gc = CW_GAIN * g;

    f = ACW_F2 / fsamp;
    _w2 = 2 * M_PI * f;
    g *= 2 / (2 - _w2);
    _w2 *= 1 - 3 * f;

    f = ACW_F3 / fsamp;
    _w3 = 2 * M_PI * f;
    g *= 2 / (2 - _w3);
    _w3 *= 1 - 3 * f;
    _ga = AW_GAIN * g;

    _err = false;
    return 0;
}

void Iec_ACfilter::reset (void)
{
    _z1a = _z1b = _z2 = _z3 = _z4a = _z4b = 0;
}

void Iec_ACfilter::process (int n, const float *in, float *opA, float *opC)
{
    float x, e;

    if (_err)
    {
	if (opA) memset (opA, 0, n * sizeof (float));
	if (opC) memset (opC, 0, n * sizeof (float));
        return;
    }

    e = 1e-25f;
    while (n--)
    {
	x = *in++;
        // highpass sections, A and C
        _z1a += _w1 * (x - _z1a + e); 
        x -= _z1a;
        _z1b += _w1 * (x - _z1b + e); 
        x -= _z1b;
        // lowpass sections, A, and C   
        _z4a += _w4 * (x - _z4a);
        x  = 0.25f * _z4b;
        _z4b += _w4 * (_z4a - _z4b);
        x += 0.75f * _z4b;
        if (opC) *opC++ = _gc * x;
        // highpass sections, A only
        _z2 += _w2 * (x - _z2 + e); 
        x -= _z2;
        _z3 += _w3 * (x - _z3 + e); 
        x -= _z3;
        if (opA) *opA++ = _ga * x;
    }
}


// ------------------------------------------------------------------------


int Itu468filter::init (int fsamp, bool dolby)
{
    reset ();
    switch (fsamp)
    {
    case 44100:
	_whp =  4.1330773e-01f;
	_a11 = -7.3360199e-01f;
	_a12 =  2.5954875e-01f;
	_a21 = -6.1104256e-01f;
	_a22 =  2.3008855e-01f;
	_a31 = -1.8076769e-01f;
	_a32 =  4.0974531e-01f;
	_b30 =  1.3153632e+00f;
	_b31 =  7.7909422e-01f;
	_b32 = -8.1194239e-02f;
        break;
    case 48000:
	_whp =  3.8715217e-01f;
	_a11 = -8.4163201e-01f;
	_a12 =  3.0498350e-01f;
	_a21 = -6.5680242e-01f;
	_a22 =  2.3733993e-01f;
	_a31 = -3.3843556e-01f;
	_a32 =  4.3756709e-01f;
	_b30 =  9.8607997e-01f;
	_b31 =  5.4846389e-01f;
	_b32 = -8.2465158e-02f;
  	break;
    case 88200:
	_whp =  2.4577479e-01f;
	_a11 = -1.3820207e+00f;
	_a12 =  5.6534863e-01f;
	_a21 = -9.7786880e-01f;
	_a22 =  2.8603959e-01f;
	_a31 = -1.2184392e+00f;
	_a32 =  6.4096606e-01f;
	_b30 =  9.5345587e-02f;
	_b31 =  3.6653187e-02f;
	_b32 = -2.0960915e-02f;
   	break;
    case 96000:
	_whp =  2.2865345e-01f;
	_a11 = -1.4324744e+00f;
	_a12 =  5.9176731e-01f;
	_a21 = -1.0594915e+00f;
	_a22 =  3.2190937e-01f;
	_a31 = -1.2991971e+00f;
	_a32 =  6.6485137e-01f;
	_b30 =  6.7263212e-02f;
	_b31 =  2.1102539e-02f;
	_b32 = -1.7972740e-02f;
  	break;
    default:
	_err = true;
        return 1;
    }
    _err = false;
    mode (dolby);
    return 0;
}

void Itu468filter::reset (void)
{
    _zhp = 0;
    _z11 = _z12 = 0;
    _z21 = _z22 = 0;
    _z31 = _z32 = 0;
}

void Itu468filter::process (int n, const float *inp, float *out)
{
    float x, zhp, z11, z12, z21, z22, z31, z32;

    if (_err)
    {
	memset (out, 0, n * sizeof (float));
        return;
    }
    zhp = _zhp;
    z11 = _z11;
    z12 = _z12;
    z21 = _z21;
    z22 = _z22;
    z31 = _z31;
    z32 = _z32;

    while (n--)
    {
	x = *inp++ * _gg;
	zhp += _whp * (x - zhp) + 1e-25f;
	x -= zhp;
	x -= _a11 * z11 + _a12 * z12;
	z12 = z11;
	z11 = x;
	x -= _a21 * z21 + _a22 * z22;
	z22 = z21;
	z21 = x;
	x -= _a31 * z31 + _a32 * z32;
	*out++ = _b30 * x + _b31 * z31 + _b32 * z32;
	z32 = z31;
	z31 = x;
    }

    _zhp = zhp;
    _z11 = z11;
    _z12 = z12;
    _z21 = z21;
    _z22 = z22;
    _z31 = z31;
    _z32 = z32;
}


// ------------------------------------------------------------------------


int RMSdetect::init (int fsamp, bool slow)
{
    reset ();
    _slow = slow;
    _w = 8.0f / fsamp; 
    return 0;
}

void RMSdetect::reset (void)
{
    _z = 0;
}

void RMSdetect::process (int n, const float *inp)
{
    float w, x, z;

    w = _slow ? (_w / 8) : _w;
    z = _z + 1e-30f;
    while (n--)
    {
	x = *inp++;
	z += w * (x * x - z);
    }
    _z = z;
}


// ------------------------------------------------------------------------


int VUMdetect::init (int fsamp, bool slow)
{
    reset ();
    _slow = slow;
    _w = 10.6f / fsamp; 
    return 0;
}

void VUMdetect::reset (void)
{
    _z1 = _z2 = 0;
}

void VUMdetect::process (int n, const float *inp)
{
    float w, x, z1, z2;

    w = _slow ? (0.1f * _w) : _w;
    z1 = _z1 + 1e-30f;
    z2 = _z2;
    while (n--)
    {
	x = fabsf (*inp++) - 0.55f * z2;
	z1 += w * (x - z1);
	z2 += w * (z1 - z2);
    }
    if (z2 < 0) z2 = 0;
    _z1 = z1 - 1e-30f;
    _z2 = z2;
}


// ------------------------------------------------------------------------


int Itu468detect::init (int fsamp)
{
    reset ();

    _a1 = 670.0f / fsamp;
    _b1 = 3.50f / fsamp;
    _a2 = 6.60f / fsamp;
    _b2 = 0.65f / fsamp;

    return 0;
}

void Itu468detect::reset (void)
{
    _z1 = _z2 = 0;
}

void Itu468detect::process (int n, const float *inp)
{
    float x, z1, z2;

    z1 = _z1;
    z2 = _z2;

    while (n--)
    {
	x = fabsf (*inp++) + 1e-30f;
	z1 -= z1 * _b1;
	if (x > z1) z1 += _a1 * (x - z1);
	z2 -= z2 * _b2;
	if (z1 > z2) z2 += _a2 * (z1 - z2);
    }

    _z1 = z1;
    _z2 = z2;
}


// ------------------------------------------------------------------------



Nmeterdsp::Nmeterdsp (void) :
    _dcfilt (false),
    _filter (-1),
    _detect (-1),
    _dcw (0),
    _dcz (0),
    _level (0)
{
}


Nmeterdsp::~Nmeterdsp (void)
{
}


void Nmeterdsp::init (int fsamp)
{
    _dcw = 5 * 6.283f / fsamp;
    _dcz = 0.0f;
    _dcfilt = false;
    _filter = FIL_NONE;
    _detect = DET_NONE;
    _enbfilt.init (fsamp);
    _iecfilt.init (fsamp);
    _itufilt.init (fsamp, false);
    _rmsdet.init (fsamp, false);
    _vumdet.init (fsamp, false);
    _itudet.init (fsamp);
    _level = 0;
}


int Nmeterdsp::set_filter (int ftype, int dcfilt)
{
    _dcfilt = dcfilt;
    _dcz = 0;
    _filter = ftype;
    switch (_filter)
    {
    case FIL_ENB20K:
	_enbfilt.reset ();
	return 0;
    case FIL_IEC_A:
    case FIL_IEC_C:
	_iecfilt.reset ();
	return 0;
    case FIL_ITU468: 
    case FIL_DOLBY: 
	_itufilt.reset ();
	_itufilt.mode (_filter == FIL_DOLBY);
	return 0;;
    }
    return -1;
}


int Nmeterdsp::set_detect (int dtype)
{
    _detect = dtype;
    switch (_detect)
    {
    case DET_RMS:
    case DET_RMS_SLOW:
	_rmsdet.reset ();
	_rmsdet.mode (_detect == DET_RMS_SLOW);
	return 0;
    case DET_VUM:
    case DET_VUM_SLOW:
	_vumdet.reset ();
	_vumdet.mode (_detect == DET_VUM_SLOW);
	return 0;
    case DET_ITU468:
	_itudet.reset ();
	return 0;
    }
    return -1;
}


void Nmeterdsp::process (float *inp, float *out, int nframes)
{
    int   i;
    float x, z;
	
    if (_dcfilt)
    {
	z = _dcz;
	for (i = 0; i < nframes; i++)
	{
	    x = inp [i] + 1e-25f;
	    z += _dcw * (x - z);
	    out [i] = x - z;
	}
	_dcz = z;
    }
    else
    {
	memcpy (out, inp, nframes * sizeof (float));
    }

    switch (_filter)
    {
    case FIL_ENB20K:
	_enbfilt.process (nframes, out, out);
	break;
    case FIL_IEC_A:
	_iecfilt.process (nframes, out, out, 0);
	break;
    case FIL_IEC_C:
	_iecfilt.process (nframes, out, 0, out);
	break;
    case FIL_ITU468:
    case FIL_DOLBY:
	_itufilt.process (nframes, out, out);
	break;
    }

    switch (_detect)
    {
    case DET_RMS:
    case DET_RMS_SLOW:
	_rmsdet.process (nframes, out);
	_level = _rmsdet.level ();
	break;
    case DET_VUM:
    case DET_VUM_SLOW:
	_vumdet.process (nframes, out);
	_level = _vumdet.level ();
	break;
    case DET_ITU468:
	_itudet.process (nframes, out);
	_level = _itudet.level ();
	break;
    default:
	_level = 0.0f;
    }
}



