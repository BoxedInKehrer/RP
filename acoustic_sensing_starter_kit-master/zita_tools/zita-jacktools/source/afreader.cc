#include <string.h>
#include "afreader.h"


AFreader::AFreader (void) :
    _stop (true),
    _qin1 (0),
    _qin2 (0),
    _qin3 (0),
    _iseq (0),
    _posit (0),
    _frag0 (0),
    _frag1 (0),
    _sndf_file (0),
    _file_chan (0),
    _file_rate (0),
    _file_size (0),
    _file_frag (0),
    _frag_size (0),
    _buff_size (0),
    _buff_offs (0),
    _buff (0)
{
}


AFreader::~AFreader (void)
{
    if (!_stop)
    {
	_stop = true;
        _trig.post ();
        _sync.wait ();
    }
    close ();
}


void AFreader::runthr (int type, int prio)
{
    _stop = false;
    thr_start (type, prio, 0x10000);
}


int AFreader::open (const char *path)
{
    SF_INFO  I;
    int      n;

    close ();
    if ((_sndf_file = sf_open (path, SFM_READ, &I)) == 0) return 1;
    _file_chan = I.channels;
    _file_rate = I.samplerate;
    _file_size = I.frames;
    n = (_file_rate + QUANT) / (2 * QUANT);
    _frag_size = n * QUANT;
    _file_frag = (_file_size + _frag_size - 1) / _frag_size;
    _buff_size = NFRAG * _frag_size;
    _buff = new float [_file_chan * _buff_size];
    locate (0);

    return 0;
}


int AFreader::close (void)
{
    if (!_sndf_file) return 0;

    sf_close (_sndf_file);
    _posit = 0;
    _frag0 = 0;
    _frag1 = 0;
    _fnext = 0;
    _sndf_file = 0;
    _file_chan = 0;
    _file_rate = 0;
    _file_size = 0;
    _file_frag = 0;
    delete[] _buff;
    _buff = 0;

    return 0;
}


void AFreader::forward (int32_t k)
{
    RDcomm *C;

    if (!_buff) return;

    _buff_offs += k;
    if (_buff_offs >= _buff_size)
    {
	_buff_offs -= _buff_size;
    }

    _posit += k;
    if (_posit >= _file_size)
    {
	_posit = _file_size;
	return;
    }

    if (   (_posit - _frag0 * _frag_size >= _frag_size)
        && (_fnext < _file_frag)
        && (_qin1 - _qin3 < QSIZE)) 
    {
        _frag0++;
        C = _queue + (_qin1 & QMASK);
        C->_iseq  = _iseq;
        C->_frag0 = _fnext++;
        C->_frag1 = _fnext;
        _qin1++;
        _trig.post ();
    }
}


bool AFreader::locate (int64_t k)
{
    RDcomm  *C;
    int     f0, f1, n;

    if (!_buff) return true;

    if (k > _file_size) k = _file_size;
    f0 = k / _frag_size;
    f1 = f0 + NFRAG;
    if (f1 > _file_frag) f1 = _file_frag;
    n = f1 - f0;
    _buff_offs = k % _buff_size;
    _posit = k;
    _frag0 = f0;
    _frag1 = f0;
    _fnext = f1;
    _iseq++;
    if (_qin1 - _qin3 > QSIZE - 2) return true;
    if (n >= 4)
    {
        C = _queue + (_qin1 & QMASK);
        C->_iseq  = _iseq;
	C->_frag0 = f0;
	C->_frag1 = f0 + 3;
	f0 += 3;
        _qin1++;
    } 
    C = _queue + (_qin1 & QMASK);
    C->_iseq  = _iseq;
    C->_frag0 = f0;
    C->_frag1 = f1;
    _qin1++;

    _trig.post ();
    return false;
}


void AFreader::update_avail (void)
{
    RDcomm *C;

    if (!_buff) return;
    while (_qin3 != _qin2)
    {
        C = _queue + (_qin3 & QMASK);
	if (C->_iseq == _iseq)
	{
	    if (C->_frag0 != _frag1) _frag0 = C->_frag0;
	    if (C->_frag1 != _frag0) _frag1 = C->_frag1;
	}
	_qin3++;
    }
}


void AFreader::thr_main (void)
{
    RDcomm    *C;
    float     *p;
    int        f;
    int        m;
    int32_t    n;
    
    while (true)
    {
        _trig.wait ();
	if (_stop)
	{
	    _sync.post ();
	    return;
	}
	while (_qin2 != _qin1)
	{
	    C = _queue + (_qin2 & QMASK);
	    if (C->_iseq == _iseq)
	    {
                f = C->_frag0;
		m = f % NFRAG;
  	        p = _buff + m * _file_chan * _frag_size;
                sf_seek (_sndf_file, (int64_t) f * _frag_size, 0);
		while (f < C->_frag1)
		{
   	            n = sf_readf_float (_sndf_file, p, _frag_size);
		    if (n < _frag_size) break;
		    f++;
                    m++;
		    p += _file_chan * _frag_size;
		    if (m == NFRAG)
		    {
                        m = 0;
			p = _buff;
		    }
		}
	    }
	    _qin2++;
	}	    
    }
}

