#ifndef __JPLAYER_H
#define __JPLAYER_H


#include <zita-jclient.h>
#include <zita-resampler/resampler.h>
#include "afreader.h"


class Jplayer : public Jclient
{
public:

    Jplayer (const char *client_name, const char *server_name, int nchan);
    virtual ~Jplayer (void);

    enum { MAXCHAN = 64 };
    enum { STOPPED = 10, STOPLOC, PLAYING, PLAYLOC, EOFILE, LOCATE };

    const char *jack_name (void) const { return _jack_name; }
    int     jack_rate (void) const { return _jack_rate; }
    int     jack_size (void) const { return _jack_size; }
    int64_t get_posit (void) const { return _afreader.posit (); }
    void    set_state (int state);
    void    set_posit (int64_t posit);
    void    set_gain (float g, float t);
    int     open_file (const char *name);
    int     close_file (void);
    int     file_chan (void) { return _afreader.file_chan (); }
    int     file_rate (void) { return _afreader.file_rate (); }
    int64_t file_size (void) { return _afreader.file_size (); }

protected:

    int  jack_process (int nframes);
    void update_state (void);
    void check_reader (void);
    void get_frames (int k, int *n, float **p);
    void output_frames (int n, float *p);
    void output_silence (int n);

    volatile int     _new_state;
    volatile int64_t _new_posit;
    volatile int     _state_seq1;
    volatile int     _state_seq2;
    P_sema           _state_sync;
    volatile int     _gain_seq1;
    volatile int     _gain_seq2;
    volatile float   _gain0;
    volatile float   _gain1;
    volatile float   _tgain;
    float            _dgain;
    int              _ngain;
    float           *_port_buff [MAXCHAN];
    AFreader         _afreader;
    Resampler        _resampler;
    float           *_resbuffer;
    double           _resratio;
    float            _g0;
    float            _g1;
    float            _dg;

    static void jack_static_shutdown (void *arg);
    static int  jack_static_process (jack_nframes_t nframes, void *arg);
};


#endif
