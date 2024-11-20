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


#include <assert.h>
#include <string.h>
#include "ambbin4.h"


Ambbin4::Ambbin4 (int fsamp, int degree, int maxlen, int period) :
    _fsamp (fsamp),
    _degree (degree),
    _period (period),
    _ninput (0),
    _nfcomp (false),
    _ambrot4 (0),
    _binconv (0)
{
    _ambrot4 = new Ambrot4 (fsamp, degree);
    _binconv = new Binconv (degree, maxlen, period);
    _ninput = _binconv->nharm ();
    for (int i = 0; i < _ninput; i++) _buff [i] = new float [period];
}


Ambbin4::~Ambbin4 (void)
{
    for (int i = 0; i < _ninput; i++) delete _buff [i];
    delete _ambrot4;
    delete _binconv;
}


void Ambbin4::set_nfcomp (float dist)
{
    int   i;
    float w;

    if (dist > 15.0f)
    {
        for (i = 0; i < 3; i++) _nffilt1 [i].reset ();
        for (i = 0; i < 5; i++) _nffilt2 [i].reset ();
        for (i = 0; i < 7; i++) _nffilt3 [i].reset ();
        for (i = 0; i < 9; i++) _nffilt4 [i].reset ();
        _nfcomp = false;
    }
    else
    {
        if (dist < 0.5f) dist = 0.1f;
        w = 343.0f / (dist * _fsamp);
        for (i = 0; i < 3; i++) _nffilt1 [i].init (w);
        for (i = 0; i < 5; i++) _nffilt2 [i].init (w);
        for (i = 0; i < 7; i++) _nffilt3 [i].init (w);
        for (i = 0; i < 9; i++) _nffilt4 [i].init (w);
        _nfcomp = true;
    }
}


void Ambbin4::process (int nframes, float *inp[], float *out[])
{
    int    i;

    assert (nframes == _period);
    _ambrot4->process (nframes, inp, _buff);
    if (_nfcomp)
    {
        if (_degree >= 1)
        {
            for (i = 0; i < 3; i++) _nffilt1 [i].process (nframes, _buff [i + 1]);
        }
        if (_degree >= 2)
        {
            for (i = 0; i < 5; i++) _nffilt2 [i].process (nframes, _buff [i + 4]);
        }
        if (_degree >= 3)
        {
            for (i = 0; i < 7; i++) _nffilt3 [i].process (nframes, _buff [i + 9]);
        }
        if (_degree >= 4)
        {
            for (i = 0; i < 9; i++) _nffilt4 [i].process (nframes, _buff [i + 16]);
        }
    }
    _binconv->process (_buff, out);
}

