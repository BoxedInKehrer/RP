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


#include <math.h>
#include <string.h>
#include "noisegen.h"


Noisegen::Noisegen (void) :
    _type (0),
    _gain (0),
    _b0 (0),
    _b1 (0),
    _b2 (0),
    _b3 (0),
    _b4 (0),
    _b5 (0),
    _b6 (0)
{
}

Noisegen::~Noisegen (void)
{
}


void Noisegen::setparam (int type, float level)
{
    _type = type;
    _gain = powf (10.0f, 0.05f * level);
}


void Noisegen::process (int nf, float *out)
{
    float g, x;
    
    switch (_type)
    {
    case WHITE:
	g = sqrtf (0.5f) * _gain;
	while (nf--)
	{
	    *out++ = g * grandf ();
	}
	break;
    case PINK:
        g = 0.23f * _gain;
	while (nf--)
	{
	    x = g * grandf ();
            _b0 = 0.99886f * _b0 + 0.0555179f * x;
            _b1 = 0.99332f * _b1 + 0.0750759f * x;
            _b2 = 0.96900f * _b2 + 0.1538520f * x;
            _b3 = 0.86650f * _b3 + 0.3104856f * x;
            _b4 = 0.55000f * _b4 + 0.5329522f * x;
            _b5 = -0.7616f * _b5 - 0.0168980f * x;
            *out++ = _b0 + _b1 + _b2 + _b3 + _b4 + _b5 + _b6 + x * 0.5362f;
            _b6 = x * 0.115926f;
	}
	break;
    default:
	memset (out, 0, nf * sizeof (float));
    }
}

