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


#ifndef __AMBROT4_H
#define __AMBROT4_H


#include "posixthr.h"


class Ambrot4
{
public:

    Ambrot4 (int fsamp, int degree);
    virtual ~Ambrot4 (void);

    void set_quaternion (float w, float x, float y, float z, float dt);
    void process (int nframes, float *inp[], float *out[]);
    
private:

    void init (int degree);
    void process0 (int k0, int nf, float *inp[], float *out[]);
    void process1 (int k0, int nf, float *inp[], float *out[]);
    void update (void);
    void matrix1 (void);
    void matrix2 (void);
    void matrix3 (void);
    void matrix4 (void);
    float funcU (int k, int m, int n) { return funcP (k, m, n, 0); }
    float funcV (int k, int m, int n);
    float funcW (int k, int m, int n);
    float funcP (int k, int m, int n, int i);
    
    int              _fsamp;
    int              _nharm;
    float            _w, _x, _y, _z, _t;
    P_mutex          _mutex;
    volatile int     _touch0;
    volatile int     _touch1;
    int              _count;
    float            _M1 [3][3]; // Target matrices
    float            _M2 [5][5];
    float            _M3 [7][7];
    float            _M4 [9][9];
    float            _C1 [3][3]; // Current matrices
    float            _C2 [5][5];
    float            _C3 [7][7];
    float            _C4 [9][9];

    static void initconst (int d, float *R, float *U, float *V, float *W);
    static bool initdone;
    // Precomputed magic numbers.
    static float R2 [3], U2 [3], V2 [3], W2 [3];
    static float R3 [4], U3 [4], V3 [4], W3 [4];
    static float R4 [5], U4 [5], V4 [5], W4 [5];
};


#endif
