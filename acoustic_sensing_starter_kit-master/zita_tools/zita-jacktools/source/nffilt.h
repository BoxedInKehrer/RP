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


#ifndef __NFFILT_H
#define __NFFILT_H


class NF_filt1
{
public:

    NF_filt1 (void) { reset (); }
    ~NF_filt1 (void) {}
    void init (float w);
    void reset (void) { _z1 = 0; }
    void process (int n, float *p);

private:

    float _g;
    float _d1;
    float _z1;
};


class NF_filt2
{
public:

    NF_filt2 (void) { reset (); }
    ~NF_filt2 (void) {}
    void init (float w);
    void reset (void) { _z1 = _z2 = 0; }
    void process (int n, float *p);

private:

    float _g;
    float _d1, _d2;
    float _z1, _z2;
};


class NF_filt3
{
public:

    NF_filt3 (void) { reset (); }
    ~NF_filt3 (void) {}
    void init (float w);
    void reset (void) { _z1 = _z2 = _z3 = 0; }
    void process (int n, float *p);

private:

    float _g;
    float _d1, _d2, _d3;
    float _z1, _z2, _z3;
};


class NF_filt4
{
public:

    NF_filt4 (void) { reset (); }
    ~NF_filt4 (void) {}
    void init (float w);
    void reset (void) { _z1 = _z2 = _z3 = _z4 = 0; }
    void process (int n, float *p);

private:

    float _g;
    float _d1, _d2, _d3, _d4;
    float _z1, _z2, _z3, _z4;
};


#endif
