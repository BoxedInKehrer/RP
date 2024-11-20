// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2017 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef __JMATRIX_H
#define __JMATRIX_H


#include <zita-jclient.h>


class Jmatrix : public Jclient
{
public:

    Jmatrix (const char *client_name, const char *server_name, int ninp, int nout);
    virtual ~Jmatrix (void);

    enum { MAXINP = 64, MAXOUT = 64 };

    void set_gain (int inp, int out, float g);

private:

    void init ();
    void fini ();
    int  jack_process (int nframes);

    float           *_ginp;
    float           *_gout;
    float           *_gmatr;
    float           *_gcurr;
};


#endif
