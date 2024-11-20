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


#ifndef __BINCONV_H
#define __BINCONV_H


#include <fftw3.h>


class Fdata
{
    friend class Binconv;

private:

    Fdata (int npar, int nbin);
    ~Fdata (void);
    void clear (void);

    int              _npar;  // Number of allocated partitions   
    int              _nact;  // Number of active partitions
    int              _nbin;  // Size of frequency domain array.
    fftwf_complex  **_data;  // Frequency domain arrays.  
};


class Binconv
{
public:

    enum { MAXINP = 25, MAXSIZE = 2048 };
    
    Binconv (int degree, int size, int frag);
    ~Binconv (void);

    void reset (void);
    void process (float *inp [], float *out [2]);
    int  setimp (int inp, float gain, const float *data, int size, int step = 1);
    int  nharm (void) const { return _nsigma + _ndelta; }
    
private:

    void convlist (float *inp [], int nsig, int list []);
    void convadd (float *inp, int ind);

    int              _size;      // Maximum IR size.
    int              _frag;      // Partition size
    int              _lfft;      // FFT size
    int              _nbin;      // FFT bins
    int              _npar;      // Number of partitions
    int              _ipar;      // Current partition index.
    int              _nsigma;    // Number of symmetric components
    int              _ndelta;    // Number of antisymmetric components
    float           *_tfilt;     // Workspace
    float           *_tdata;     // Workspace
    fftwf_complex   *_fdata;     // Wokrspace
    fftwf_complex   *_fdacc;     // Frequency domain accumulator
    fftwf_plan       _plan_r2c;  // FFT plans
    fftwf_plan       _plan_c2r;  //
    float           *_buffS;     // Overlap buffer
    float           *_buffD;     // Overlap buffer
    Fdata           *_fdataA [MAXINP];    // Frequency domain data for IR.
    Fdata           *_fdataB [MAXINP];    // Frequency domain data for input.

    static int sigmalist [15];
    static int deltalist [10];
};    


#endif


