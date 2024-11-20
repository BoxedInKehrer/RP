// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#ifndef _POSIXTHR_H
#define _POSIXTHR_H


#ifdef __linux__

#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>


class P_sema
{
public:

    P_sema (void) { if (sem_init (&_sema, 0, 0)) abort (); }
    ~P_sema (void) { sem_destroy (&_sema); }
    P_sema (const P_sema&);
    P_sema& operator= (const P_sema&);

    int post (void) { return sem_post (&_sema); }
    int wait (void) { return sem_wait (&_sema); }
    int trywait  (void) { return sem_trywait (&_sema); }
    int getvalue (void) { int n; sem_getvalue (&_sema, &n); return n; }

private:

    sem_t  _sema;
};


class P_mutex
{
public:

    P_mutex (void) { if (pthread_mutex_init (&_mutex, 0)) abort (); }
    ~P_mutex (void) { pthread_mutex_destroy (&_mutex); }
    P_mutex (const P_mutex&);
    P_mutex& operator= (const P_mutex&);

    int lock (void) { return pthread_mutex_lock (&_mutex); }
    int unlock (void){ return pthread_mutex_unlock (&_mutex); }
    int trylock (void) { return pthread_mutex_trylock (&_mutex); }

private:

    pthread_mutex_t  _mutex;
};


class P_thread
{
public:

    P_thread (void);
    virtual ~P_thread (void);
    P_thread (const P_thread&);
    P_thread& operator=(const P_thread&);

    void sepuku (void) { pthread_cancel (_ident); }

    virtual void thr_main (void) = 0;
    virtual int  thr_start (int policy, int priority, size_t stacksize = 0);

private:
  
    pthread_t  _ident;
};


#endif // __linux__


#endif //  _POSIXTHR_H

