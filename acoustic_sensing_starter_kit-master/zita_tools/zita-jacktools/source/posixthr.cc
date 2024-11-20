// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2014 Fons Adriaensen <fons@linuxaudio.org>
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


#include "posixthr.h"


P_thread::P_thread (void) :
_ident (0)
{
}


P_thread::~P_thread (void)
{
}


extern "C" void *P_thread_entry_point (void *arg)
{
    P_thread *T = (P_thread *) arg;
    T->thr_main ();
    return NULL;
}


int P_thread::thr_start (int policy, int priority, size_t stacksize)
{
    int                min, max, rc;
    pthread_attr_t     attr;
    struct sched_param parm;

    min = sched_get_priority_min (policy);
    max = sched_get_priority_max (policy);
    if (priority > max) priority = max;
    if (priority < min) priority = min;
    parm.sched_priority = priority;

    pthread_attr_init (&attr);
    pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setschedpolicy (&attr, policy);
    pthread_attr_setschedparam (&attr, &parm);
    pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);
    pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setstacksize (&attr, stacksize);

    _ident = 0; 
    rc = pthread_create (&_ident,
			 &attr,
			 P_thread_entry_point,
			 this);

    pthread_attr_destroy (&attr);

    return rc;
}


void P_thread::thr_main (void)
{
}

