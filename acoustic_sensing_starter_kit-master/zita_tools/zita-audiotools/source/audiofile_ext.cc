// ----------------------------------------------------------------------------
//
//  Copyright (C) 2010-2018 Fons Adriaensen <fons@linuxaudio.org>
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


#include <Python.h>
#include "audiofile.h"


extern "C" void audiofile_destroy (PyObject *P)
{
    delete (Audiofile *) PyCapsule_GetPointer (P, "Audiofile");
}


extern "C" PyObject* audiofile_create (PyObject *self, PyObject *args)
{
    Audiofile *A = new Audiofile ();
    return PyCapsule_New ((void *) A, "Audiofile", audiofile_destroy);
}


extern "C" PyObject* audiofile_open_read (PyObject *self, PyObject *args)
{ 
    Audiofile  *A;
    PyObject   *P;
    char       *name;

    if (! PyArg_ParseTuple (args, "Os", &P, &name)) return 0;
    A = (Audiofile *) PyCapsule_GetPointer (P, "Audiofile");
    if (A->open_read (name))
    {
	PyErr_SetString (PyExc_OSError, "Unable to open audio file");
	return 0;
    }
    Py_RETURN_NONE;
}


extern "C" PyObject* audiofile_open_write (PyObject *self, PyObject *args)
{ 
    Audiofile  *A;
    PyObject   *P;
    char       *name;
    char       opts [64];
    char       *p, *q;
    int        v, chan, rate, type, form, dith;


    if (! PyArg_ParseTuple (args, "Osiiz", &P, &name, &chan, &rate, &p)) return 0;
    A = (Audiofile *) PyCapsule_GetPointer (P, "Audiofile");
    if ((chan < 1) || (chan > 1024))
    {
        PyErr_SetString (PyExc_ValueError, "Bad channel count");
	return 0;
    }
    if (rate < 1)
    {
        PyErr_SetString (PyExc_ValueError, "Bad sample frequency");
	return 0;
    }
    type = Audiofile::TYPE_WAV;
    form = Audiofile::FORM_24BIT;
    dith = 0;
    if (p)
    {  
	strncpy (opts, p, 64);
	opts [63] = 0;
	q = 0;
        p = strtok_r (opts, ",", &q);
	while (p)
	{
	    if      ((v = A->enc_type (p)) >= 0) type = v;
	    else if ((v = A->enc_form (p)) >= 0) form = v;
            else if ((v = A->enc_dith (p)) >= 0) dith = v;
	    else
	    {
                PyErr_SetString (PyExc_KeyError, "Unknown format");
	        return 0;
	    }
	    p = strtok_r (0, ",", &q);
	}
    }
    if (A->open_write (name, type, form, rate, chan))
    {
	PyErr_SetString (PyExc_OSError, "Unable to open audio file");
	return 0;
    }
    A->set_dither (dith);
    Py_RETURN_NONE;
}


extern "C" PyObject* audiofile_close (PyObject *self, PyObject *args)
{ 
    Audiofile  *A;
    PyObject   *P;

    if (! PyArg_ParseTuple (args, "O", &P)) return 0;
    A = (Audiofile *) PyCapsule_GetPointer (P, "Audiofile");
    return Py_BuildValue ("i", A->close ());
}


extern "C" PyObject* audiofile_info (PyObject *self, PyObject *args)
{ 
    Audiofile  *A;
    PyObject   *P;

    if (! PyArg_ParseTuple (args, "O", &P)) return 0;
    A = (Audiofile *) PyCapsule_GetPointer (P, "Audiofile");
    return Py_BuildValue ("iiiLss", A->mode (), A->chan (), A->rate (),
                          A->size (), A->typestr (), A->formstr ());
}


extern "C" PyObject* audiofile_seek (PyObject *self, PyObject *args)
{ 
    Audiofile  *A;
    PyObject   *P;
    int64_t    posit;
    int        mode;

    if (! PyArg_ParseTuple (args, "OLi", &P, &posit, &mode)) return 0;
    A = (Audiofile *) PyCapsule_GetPointer (P, "Audiofile");
    return Py_BuildValue ("L", A->seek (posit, mode));
}


static int check_format (Py_buffer *B, Audiofile *A)
{
    if (strcmp (B->format, "f"))
    {
        PyErr_SetString (PyExc_TypeError, "Data type must be float32");
        return 1;
    }
    if (   ((B->ndim != 1) || (A->chan () != 1))
        && ((B->ndim != 2) || (A->chan () != B->shape [1])))
    {    
        PyErr_SetString (PyExc_TypeError, "Array shape does not match");
        return 1;
    }
    return 0;
}


extern "C" PyObject* audiofile_read (PyObject *self, PyObject *args)
{ 
    Audiofile  *A;
    PyObject   *P, *Q;
    Py_buffer  B;
    int        bits, i, j, n1, n2, nc, d0, d1;
    float      *data, *buff;
    int64_t    size, nf;

    if (! PyArg_ParseTuple (args, "OO", &P, &Q)) return 0;
    A = (Audiofile *) PyCapsule_GetPointer (P, "Audiofile");
    if (!(A->mode () & Audiofile::MODE_READ))
    {
        PyErr_SetString (PyExc_TypeError, "File is not open for reading");
        return 0;
    }

    bits = PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE;
    if (PyObject_GetBuffer (Q, &B, bits)) return 0;
    if (check_format (&B, A))
    {
        PyBuffer_Release (&B);
	return 0;
    }

    size = B.shape [0];
    data = (float *) B.buf;
    if (PyBuffer_IsContiguous (&B, 'C')) nf = A->read (data, size);
    else
    {
        d0 = B.strides [0] / sizeof (float);
        d1 = (B.ndim == 1) ? 1: B.strides [1] / sizeof (float);
        nc = A->chan ();
        nf = 0;
	buff = A->get_buffer ();
	while (size)
	{
	    n1 = (size > Audiofile::BUFFSIZE) ? Audiofile::BUFFSIZE : size;
	    n2 = A->read (buff, n1);
	    for (i = 0; i < n2; i++)
	    {
		for (j = 0; j < nc; j++)
		{
		    data [d0 * i + d1 * j] = buff [nc * i + j];
		}
	    }
	    nf += n2;
	    size -= n2;
	    data += d0 * n2;
	    if (n2 < n1) break;
	}
	n1 = size;
	for (i = 0; i < n1; i++)
	{
	    for (j = 0; j < nc; j++)
	    {
		data [d0 * i + d1 * j] = 0;
	    }
	}
    }

    PyBuffer_Release (&B);
    return Py_BuildValue ("L", nf);
}


extern "C" PyObject* audiofile_write (PyObject *self, PyObject *args)
{ 
    Audiofile  *A;
    PyObject   *P, *Q;
    Py_buffer  B;
    int        bits, i, j, n1, n2, nc, d0, d1;
    float      *data, *buff;
    int64_t    size, nf;

    if (! PyArg_ParseTuple (args, "OO", &P, &Q)) return 0;
    A = (Audiofile *) PyCapsule_GetPointer (P, "Audiofile");
    if (!(A->mode () & Audiofile::MODE_WRITE))
    {
        PyErr_SetString (PyExc_TypeError, "File is not open for writing");
        return 0;
    }

    bits = PyBUF_STRIDES | PyBUF_FORMAT;
    if (PyObject_GetBuffer (Q, &B, bits)) return 0;
    if (check_format (&B, A))
    {
        PyBuffer_Release (&B);
	return 0;
    }

    size = B.shape [0];
    data = (float *) B.buf;
    if (PyBuffer_IsContiguous (&B, 'C')) nf = A->write (data, size);
    else
    {
	d0 = B.strides [0] / sizeof (float);
	d1 = (B.ndim == 1) ? 1: B.strides [1] / sizeof (float);
	nc = A->chan ();
	nf = 0;
	buff = A->get_buffer ();
	while (size)
	{
	    n1 = (size > Audiofile::BUFFSIZE) ? Audiofile::BUFFSIZE : size;
	    for (i = 0; i < n1; i++)
	    {
		for (j = 0; j < nc; j++)
		{
		    buff [nc * i + j] = data [d0 * i + d1 * j];
		}
	    }
	    n2 = A->write (buff, n1);
	    nf += n2;
	    size -= n2;
	    data += d0 * n2;
	    if (n2 < n1) break;
	}
    }

    PyBuffer_Release (&B);
    return Py_BuildValue ("L", nf);
}


static PyMethodDef AudioFileMethods[] =
{
    {"create",           audiofile_create,         METH_VARARGS, "Create an AudioFile."},
    {"open_read",        audiofile_open_read,      METH_VARARGS, "Open audio file for reading"},
    {"open_write",       audiofile_open_write,     METH_VARARGS, "Open audio file for writing"},
    {"close",            audiofile_close,          METH_VARARGS, "Close current audio file"},
    {"info",             audiofile_info,           METH_VARARGS, "Read current status"},
    {"seek",             audiofile_seek,           METH_VARARGS, "Seek to position in frames"},
    {"read",             audiofile_read,           METH_VARARGS, "Read audio frames from file."},
    {"write",            audiofile_write,          METH_VARARGS, "Write audio frames to file."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef AudioFileModule = 
{
   PyModuleDef_HEAD_INIT,
   "audiofile_ext",
   NULL, 
   -1, 
   AudioFileMethods
};

PyMODINIT_FUNC PyInit_audiofile_ext(void)
{
    return PyModule_Create (&AudioFileModule);
}

#else

PyMODINIT_FUNC initaudiofile_ext(void)
{
    (void) Py_InitModule ("audiofile_ext", AudioFileMethods);
}

#endif
