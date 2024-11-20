// ----------------------------------------------------------------------------
//
//  Copyright (C) 2012..2018 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __AMBPAN4_H
#define __AMBPAN4_H


class Ambpan4
{
public:

    Ambpan4 (int fsamp, int degree, bool semi);
    virtual ~Ambpan4 (void);

    void set_direction (float az, float el, float dt);
    void process (int nframes, float *inp, float *out[]);
    
private:

    void init (int degree, bool semi);
    void update (void);
    void encode (float azim, float elev, float *E);
    
    int              _fsamp;
    int              _nharm;
    volatile int     _touch0;
    volatile int     _touch1;
    float            _az;
    float            _el;
    float            _dt;
    int              _count;
    float            _G [25];
    float            _T [25];
    float            _C [12];
};


#endif
