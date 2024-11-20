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


#ifndef __NOISEGEN_H
#define __NOISEGEN_H


#include "rngen.h"


class Noisegen : private Rngen
{
public:

    enum { OFF, WHITE, PINK };
    
    Noisegen (void);
    ~Noisegen (void);

    void process (int nf, float *out);
    void setparam (int type, float level);

private:

    int    _type;
    float  _gain;
    float  _b0, _b1, _b2, _b3, _b4, _b5, _b6;
};


#endif
