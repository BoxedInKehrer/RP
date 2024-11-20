// ----------------------------------------------------------------------------
//
//  Copyright (C) 2013-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __BP6FILTER_H
#define __BP6FILTER_H


class Bp6param
{
public:

    enum { BBB, BBH, HH };

    int    _mode;
    float  _gain;
    float  _coeff [6];
};


class Bp6paramset
{
public:

    float       _fsamp;
    int         _nfilt;
    int         _nsubs;
    int         _rsubs;
    Bp6param   *_param; 
};


class Bp6filter
{
public:

    Bp6filter (void);
    ~Bp6filter (void);

    void setparam (const Bp6param *param);
    void reset (void);
    void process (int nsamp, float *inp, float *out);

private:

    const Bp6param  *_param; 
    double           _z [6];
};


#endif
