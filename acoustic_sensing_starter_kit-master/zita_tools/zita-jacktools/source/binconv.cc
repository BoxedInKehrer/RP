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


#include <unistd.h>
#include <string.h>
#include "binconv.h"


// -------------------------------------------------------------------------------


Fdata::Fdata (int npar, int nbin):
    _npar (npar),
    _nbin (nbin)
{
    _data = new fftwf_complex* [nbin];
    for (int i = 0; i < npar; i++)
    {
        _data [i] = (fftwf_complex *)(fftwf_malloc (nbin * sizeof (fftwf_complex)));
    }
    clear ();
}


Fdata::~Fdata (void)
{
    for (int i = 0; i < _npar; i++)
    {
        fftwf_free (_data [i]);
    }
    delete[] _data;
}


void Fdata::clear ()
{
    for (int i = 0; i < _npar; i++)
    {
        memset (_data [i], 0, _nbin * sizeof (fftwf_complex));
        _nact = 0;
    }
}


// -------------------------------------------------------------------------------


int Binconv::sigmalist [15] = { 0, 2, 3, 6, 7, 8, 12, 13, 14, 15, 20, 21, 22, 23, 24 };
int Binconv::deltalist [10] = { 1, 4, 5, 9, 10, 11, 16, 17, 18, 19 };


Binconv::Binconv (int degree, int size, int frag):
    _size (size),
    _frag (frag),
    _nsigma (0),
    _ndelta (0),
    _tfilt (0),
    _tdata (0),
    _fdata (0),
    _fdacc (0),
    _buffS (0),
    _buffD (0)
{
    if (_size > MAXSIZE) _size = MAXSIZE;
    _lfft = 2 * _frag;
    _nbin = _frag + 1;
    _npar = (_size + _frag - 1) / _frag;
    _ipar = 0;

    // These are allocated using fftw to ensure correct alignment.
    _tfilt = (float *) (fftwf_malloc (_lfft * sizeof (float)));
    _tdata = (float *) (fftwf_malloc (_lfft * sizeof (float)));
    _fdata = (fftwf_complex *)(fftwf_malloc (_nbin * sizeof (fftwf_complex)));
    _fdacc = (fftwf_complex *)(fftwf_malloc (_nbin * sizeof (fftwf_complex)));

    // FFTW plans.
    _plan_r2c = fftwf_plan_dft_r2c_1d (_lfft, _tdata, _fdata, 0);
    _plan_c2r = fftwf_plan_dft_c2r_1d (_lfft, _fdata, _tdata, 0);

    // Output overlap.
    _buffS = new float [_frag];
    _buffD = new float [_frag];
    
    // Configure for degree.
    switch (degree)
    {
    case 1:
        _nsigma = 3;
        _ndelta = 1;
        break;
    case 2:
        _nsigma = 6;
        _ndelta = 3;
        break;
    case 3:
        _nsigma = 10;
        _ndelta = 6;
        break;
    case 4:
        _nsigma = 15;
        _ndelta = 10;
        break;
    default:
        return;
    }

    // Allocate F-domain data.
    for (int i = 0; i < _nsigma + _ndelta; i++)
    {
        _fdataA [i] = new Fdata (_npar, _nbin);
        _fdataB [i] = new Fdata (_npar, _nbin);
    }
}


Binconv::~Binconv (void)
{
    fftwf_destroy_plan (_plan_r2c);
    fftwf_destroy_plan (_plan_c2r);
    fftwf_free (_tfilt);
    fftwf_free (_tdata);
    fftwf_free (_fdata);
    fftwf_free (_fdacc);
    delete[] _buffS;
    delete[] _buffD;
    for (int i = 0; i < _nsigma + _ndelta; i++)
    {
        delete _fdataA [i];
        delete _fdataB [i];
    }
}


void Binconv::reset (void)
{
    for (int i = 0; i < _nsigma + _ndelta; i++) _fdataB [i]->clear ();
    memset (_buffS, 0, _frag * sizeof (float));
    memset (_buffD, 0, _frag * sizeof (float));
    _ipar = 0;
}


int Binconv::setimp (int inp, float gain, const float *data, int size, int step)
{
    int     i, j, n;
    Fdata   *FA;

    if ((inp < 0) || (inp >= _nsigma + _ndelta)) return 1;
    gain /= _lfft;
    FA = _fdataA [inp];
    FA->clear ();
    if (! data) return 0;
    for (i = 0; i < _npar; i++)
    {
        if (! size) break;
        n = (size < _frag) ? size : _frag;
        for (j = 0; j < n; j++)
        {
            _tfilt [j] = gain * data [j * step];
        }
        memset (_tfilt + n, 0, (_lfft - n) * sizeof (float));
        fftwf_execute_dft_r2c (_plan_r2c, _tfilt, FA->_data [i]);       
        data += n * step;
        size -= n;
    }
    FA->_nact = i;
    return 0;
}


void Binconv::process (float *inp [], float *out [2])
{
    int     i;
    float   d;
    float   *pL, *pR;

    pL = out [0];     // Left ouput
    pR = out [1];     // Right output

    // Process sum channel.
    convlist (inp, _nsigma, sigmalist);
    for (i = 0; i < _frag; i++)
    {
        pL [i] = pR [i] = _buffS [i] + _tdata [i];
    }
    memcpy (_buffS, _tdata + _frag, _frag * sizeof (float));

    // Process difference channel.
    convlist (inp, _ndelta, deltalist);
    for (i = 0; i < _frag; i++)
    {
        d = _buffD [i] + _tdata [i];
        pL [i] += d;
        pR [i] -= d;
    }
    memcpy (_buffD, _tdata + _frag, _frag * sizeof (float));

    // Increment current partition index.
    if (++_ipar == _npar) _ipar = 0;
}


void Binconv::convlist (float *inp [], int nsig, int list [])
{
    int i, k;
    memset (_fdacc, 0, _nbin * sizeof (fftwf_complex));
    for (i = 0; i < nsig; i++)
    {
        k = list [i];
        convadd (inp [k], k);
    }
    fftwf_execute_dft_c2r (_plan_c2r, _fdacc, _tdata); 
}


void Binconv::convadd (float *inp, int ind)
{
    int            i, j, k;
    float          x, y;
    fftwf_complex  *A, *B;
    Fdata          *FA, *FB;
    
    FA = _fdataA [ind];
    FB = _fdataB [ind];
    memcpy (_tdata, inp, _frag * sizeof (float));
    memset (_tdata + _frag, 0, _frag * sizeof (float));
    fftwf_execute_dft_r2c (_plan_r2c, _tdata, FB->_data [_ipar]); 
    j = _ipar;
    for (k = 0; k < FA->_nact; k++)
    {
        A = FA->_data [k];
        B = FB->_data [j];
        for (i = 0; i < _nbin; i++)
        {
            x = A [i][0] * B [i][0] - A [i][1] * B [i][1];
            y = A [i][0] * B [i][1] + A [i][1] * B [i][0];
            _fdacc [i][0] += x; 
            _fdacc [i][1] += y; 
        }
        if (--j < 0) j += _npar;
    }
}
