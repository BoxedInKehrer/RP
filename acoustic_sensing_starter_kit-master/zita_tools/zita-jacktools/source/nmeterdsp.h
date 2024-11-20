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


#ifndef __NMETERDSP_H
#define	__NMETERDSP_H


#include <math.h>


class Enb20kfilter
{
public:

    Enb20kfilter (void) {}
    ~Enb20kfilter (void) {}

    int  init (int fsamp);
    void reset (void);
    void process (int n, const float *inp, float *out);

private:

    bool  _err;
    float _g;
    float _b1, _b2, _b3, _b4;	       
    float _z1, _z2, _z3, _z4;
};


class Iec_ACfilter
{
public:

    Iec_ACfilter (void) : _err (true) {}
    ~Iec_ACfilter (void) {}

    int  init (int fsamp);
    void reset (void);
    void process (int n, const float *in, float *opA, float *opC);

private:

    bool  _err;
    float _w1, _w2, _w3, _w4, _ga, _gc;       // filter coefficients and gains
    float _z1a, _z1b, _z2, _z3, _z4a, _z4b;   // filter state
};


class Itu468filter
{
public:

    Itu468filter (void) : _err (true) {}
    ~Itu468filter (void) {}

    int  init (int fsamp, bool dolby = false);
    void mode (bool dolby) { _gg = dolby ? 0.5239f : 1.0f; }
    void reset (void);
    void process (int n, const float *inp, float *out);

private:

    bool     _err;
    float    _gg;
    float    _whp;
    float    _a11, _a12;
    float    _a21, _a22;
    float    _a31, _a32;
    float    _b30, _b31, _b32;
    float    _zhp;
    float    _z11, _z12;
    float    _z21, _z22;
    float    _z31, _z32;
};


class RMSdetect
{
public:

    RMSdetect (void) {}
    ~RMSdetect (void) {}

    int   init (int fsamp, bool slow = false);
    void  mode (bool slow) { _slow = slow; }
    void  reset (void);
    void  process (int n, const float *inp);
    float level (void) { return sqrtf (2 * _z); }
    
private:

    bool  _slow;
    float _w;
    float _z;
};


class VUMdetect
{
public:

    VUMdetect (void) {}
    ~VUMdetect (void) {}

    int   init (int fsamp, bool slow = false);
    void  mode (bool slow) { _slow = slow; }
    void  reset (void);
    void  process (int n, const float *inp);
    float level (void) { return 2.435f * _z2; }
    

private:

    bool  _slow;
    float _w;
    float _z1, _z2;
};


class Itu468detect
{
public:

    Itu468detect (void) {}
    ~Itu468detect (void) {}

    int   init (int fsamp);
    void  reset (void);
    void  process (int n, const float *inp);
    float level (void) { return 1.1453f * _z2; }

private:

    float _a1, _b1;
    float _a2, _b2;
    float _z1, _z2;
};


class Nmeterdsp
{
public:

    enum { FIL_NONE, FIL_ENB20K, FIL_IEC_A, FIL_IEC_C,  FIL_ITU468, FIL_DOLBY };
    enum { DET_NONE, DET_RMS, DET_RMS_SLOW, DET_VUM, DET_VUM_SLOW, DET_ITU468 };
    
    Nmeterdsp (void);
    ~Nmeterdsp (void);

    void  init (int fsamp);
    int   set_filter (int ftype, int dcfilt);
    int   set_detect (int dtype);
    void  process (float *inp, float *out, int nframes);  
    float level (void) const { return _level; }

private:

    bool   _dcfilt;
    int    _filter;
    int    _detect;
    float  _dcw;
    float  _dcz;
    float  _level;
    
    Enb20kfilter  _enbfilt;
    Iec_ACfilter  _iecfilt;
    Itu468filter  _itufilt;
    RMSdetect     _rmsdet;
    VUMdetect     _vumdet;
    Itu468detect  _itudet;
};


#endif
