// ----------------------------------------------------------------------------
//
//  Copyright (C) 2015-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __AMBBIN4_H
#define __AMBBIN4_H


#include <math.h>
#include "nffilt.h"
#include "ambrot4.h"
#include "binconv.h"


class Ambbin4
{
public:

    enum { MAXINP = 25, NOUT = 2 };
    
    Ambbin4 (int fsamp, int degree, int maxlen, int period);
    virtual ~Ambbin4 (void);

    void set_nfcomp (float distance);
    void set_filter (int harm, const float *data, int size, int step = 1)
    {
	_binconv->setimp (harm, 1.0f, data, size, step);
    }
    void set_quaternion (float w, float x, float y, float z, float t)
    {
	_ambrot4->set_quaternion (w, x, y, z, t);
    }
    void process (int nframes, float *inp [], float *out []);

private:

    int              _fsamp;
    int              _degree;
    int              _period;
    int              _ninput;
    bool             _nfcomp;
    NF_filt1         _nffilt1 [3];
    NF_filt2         _nffilt2 [5];
    NF_filt3         _nffilt3 [7];
    NF_filt4         _nffilt4 [9];
    Ambrot4         *_ambrot4;
    Binconv         *_binconv;
    float           *_buff [MAXINP];
};


#endif
