#ifndef __AFREADER_H
#define __AFREADER_H


#include <sndfile.h>
#include "posixthr.h"


class RDcomm
{
public:

    int _iseq;
    int _frag0;
    int _frag1;
};



class AFreader : private P_thread
{
public:

    enum { QSIZE = 8, QMASK = QSIZE - 1, QUANT = 4096, NFRAG = 6 };
    
    AFreader (void);
    virtual ~AFreader (void);

    void     runthr (int type, int prio);
    int      open (const char *path);
    int      close (void);
    int      file_chan (void) const { return _file_chan; }
    int      file_rate (void) const { return _file_rate; }
    int64_t  file_size (void) const { return _file_size; }
    int64_t  posit (void) const { return _posit; }
    float   *buffp (void) const { return _buff + _file_chan * _buff_offs; }
    void     update_avail (void);
    int32_t  avail_lin (void) const { return _buff_size - _buff_offs; }
    int32_t  avail_rev (void) const { return _posit - _frag0 * _frag_size; }
    int32_t  avail_fwd (void) const { return _frag1 * _frag_size - _posit; }
    int64_t  avail_end (void) const { return _file_size - _posit; }
    bool     locate (int64_t k);
    void     forward (int32_t k);

private:

    virtual void thr_main (void);

    bool           _stop;
    P_sema         _trig;
    P_sema         _sync;
    RDcomm         _queue [QSIZE];
    volatile int   _qin1; 
    volatile int   _qin2; 
    volatile int   _qin3; 
    volatile int   _iseq;
    int64_t        _posit;
    int32_t        _frag0;
    int32_t        _frag1;
    int32_t        _fnext;
    SNDFILE       *_sndf_file;
    int32_t        _file_chan;
    int32_t        _file_rate;
    int64_t        _file_size;
    int32_t        _file_frag;
    int32_t        _frag_size;
    int32_t        _buff_size;
    int32_t        _buff_offs;
    float         *_buff;
};


#endif
