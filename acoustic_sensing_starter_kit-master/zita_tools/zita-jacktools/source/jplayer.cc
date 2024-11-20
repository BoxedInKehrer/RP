//#include <stdio.h>
#include <string.h>
#include <math.h>
#include "jplayer.h"


Jplayer::Jplayer (const char *client_name, const char *server_name, int nchan) :
    _new_state (INITIAL),
    _new_posit (0),
    _state_seq1 (0),
    _state_seq2 (0),
    _gain_seq1 (0),
    _gain_seq2 (0),
    _gain0 (0),
    _gain1 (0),
    _tgain (0),
    _dgain (0),
    _ngain (0),
    _resbuffer (0),
    _g0 (1),
    _g1 (1),
    _dg (0)
{
    if (nchan < 0) nchan = 0;
    if (nchan > MAXCHAN) nchan = MAXCHAN;
    if (   open_jack (client_name, server_name, 0, nchan)
        || create_out_ports ("out_%d"))
    {
	_state = FAILED;
	return;
    }
    memset (_port_buff, 0, MAXCHAN * sizeof (float *));
    _afreader.runthr (_schedpol, 0);
    _state = _new_state = SILENCE;
}


Jplayer::~Jplayer (void)
{
    close_jack ();
}


void Jplayer::set_state (int state)
{
    if (_state < PASSIVE) return;
    _new_state = state;
    _state_seq1++;
    while (_state_seq2 != _state_seq1) _state_sync.wait ();
}


void Jplayer::set_posit (int64_t posit)
{
    if (_state < STOPPED) return;
    _new_state = LOCATE;
    _new_posit = posit;
    _state_seq1++;
    while (_state_seq2 != _state_seq1) _state_sync.wait ();
}


void Jplayer::set_gain (float g, float t)
{
    _gain1 = g;
    _tgain = t;
    _gain_seq1++;
}


int Jplayer::open_file (const char *name)
{
    int fr, nc;
    
    if (_state != SILENCE) return 1;
    if (_afreader.open (name)) return 1;
    fr = _afreader.file_rate ();
    nc = _afreader.file_chan ();
    if (fr != _jack_rate)
    {
	_resampler.setup (fr, _jack_rate, nc, 36);
	_resratio = (double) _jack_rate / fr;
	_resbuffer = new float [nc * _jack_size];
    }
    else
    {
        delete[] _resbuffer;
        _resbuffer = 0;
	_resampler.clear ();
    }
    return 0;
}


int Jplayer::close_file (void)
{
    if (_state != SILENCE) return 1;
    delete[] _resbuffer;
    _resbuffer = 0;
    _afreader.close ();
    return 0;
}


int Jplayer::jack_process (int nframes)
{
    int    i, k, n;
    float  *p;

    if (_state < PASSIVE) return 0;
    if (_state_seq2 != _state_seq1)
    {
        update_state ();
        _state_seq2++;
        _state_sync.post ();
    }
    if (_state < SILENCE) return 0;
    
    for (i = 0; i < _nout; i++)
    {
        if (_out_ports [i]) _port_buff [i] = (float *) jack_port_get_buffer (_out_ports [i], nframes);
        else                _port_buff [i] = 0;
    }

    if (_gain_seq2 != _gain_seq1)
    {
	_dgain = _gain1 - _gain0;
	if (fabsf (_dgain) < 0.1f)
	{
	    _ngain = 1;
	}
	else
	{	
 	    _ngain = ceilf ((_tgain + 1e-3f) * _jack_rate / _jack_size);
	    _dgain /= _ngain;
	}
        _gain_seq2 = _gain_seq1;
    }
    if (_ngain)
    {
	_gain0 += _dgain;
	_g1 = (_gain0 <= -150.0f) ? 0.0f : powf (10.0f, 0.05f * _gain0);
        _dg = (_g1 - _g0) / _jack_size;
	_ngain--;
    }
    else
    {
	_dg = 0;
    }

    if (_state == SILENCE) 
    {
	output_silence (nframes);
        _g0 = _g1;
	return 0;
    }

    check_reader ();
    if (_resbuffer)
    {
	_resampler.out_count = nframes;
	_resampler.out_data = _resbuffer;
        _resampler.process ();
	while (_resampler.out_count)
	{
	    k = (int)(ceilf (_resampler.out_count / _resratio));
            get_frames (k, &n, &p);
            _resampler.inp_count = n;
            _resampler.inp_data = p;
            _resampler.process ();
	    if (p) _afreader.forward (n - _resampler.inp_count);
	}	    
	output_frames (nframes, _resbuffer);
    }
    else
    {
        while (nframes)
        {
	    get_frames (nframes, &n, &p);
	    if (p)
	    {
		output_frames (n, p);
	        _afreader.forward (n);
	    }
   	    else output_silence (n);
            nframes -= n;		       
	}
    }

    _g0 = _g1;
    return 0;
}


void Jplayer::update_state (void)
{
    switch (_new_state)
    {
    case SILENCE:
	_state = SILENCE;
	break;

    case STOPPED:
	switch (_state)
	{
	case SILENCE:
	case PLAYING:
	    _state = STOPPED;
	    break;

	case PLAYLOC:
	    _state = STOPLOC;
	    break;
	}
	break;

    case PLAYING:
	switch (_state)
	{
	case STOPPED:
	case STOPLOC:
	    _state = PLAYLOC;
	    break;
	}
	break;

    case LOCATE:
	switch (_state)
	{
	case EOFILE:    
	case STOPPED:    
	case STOPLOC:
	    _afreader.locate (_new_posit);
	    _state = STOPLOC;
	    break;

	case PLAYING:
	case PLAYLOC:
	    _afreader.locate (_new_posit);
	    _state = PLAYLOC;
	    break;
	}
	break;
    }
}


void Jplayer::check_reader (void)
{
    int kf, ke;

    _afreader.update_avail ();
    kf = _afreader.avail_fwd ();
    ke = _afreader.avail_end ();
    if (ke <= 0)
    {
        _state = EOFILE;
	kf = 0;
	ke = 0;
    }
    else if ((kf >= ke) || (kf >= _afreader.file_rate ()))
    {  
        if (_state == STOPLOC) _state = STOPPED;
        if (_state == PLAYLOC) _state = PLAYING;
    }
}


void Jplayer::get_frames (int k, int *n, float **p)
{
    int  kf, ke, kl;

    *n = k;
    *p = 0;
    if (_state != PLAYING) return;	
    kf = _afreader.avail_fwd ();
    ke = _afreader.avail_end ();
    kl = _afreader.avail_lin ();
    if (k > kf) k = kf;
    if (k > ke) k = ke;
    if (k > kl) k = kl;
    if (k > 0)
    {
	*n = k;
	*p = _afreader.buffp ();
    }
}

    
void Jplayer::output_frames (int n, float *b)
{
    int    i, j, nc;
    float  g, dg;
    float  *p, *q;

    nc = _afreader.file_chan ();
    dg = _dg;
    for (i = 0; i < _nout; i++)
    {
        q = _port_buff [i];
        if (q)
        {
	    if (i < nc)
	    {
	        g = _g0;
                p = b + i;
                for (j = 0; j < n; j++)
                {
		    g += dg;
                    q [j] = g * p [nc * j];
                }
	    }
	    else
	    {
                memset (q, 0, n * sizeof (float));
            }
            _port_buff [i] = q + n;
        }
    }
    _g0 += n * _dg;
}


void Jplayer::output_silence (int n)
{
    int   i;
    float *q;

    for (i = 0; i < _nout; i++)
    {
        q = _port_buff [i];
        if (q)
        {
	    memset (q, 0, n * sizeof (float));
	    _port_buff [i] = q + n;
	}
    }
    _g0 += n * _dg;
}


