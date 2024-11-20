// ----------------------------------------------------------------------------
//
//  Copyright (C) 2012-2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include <zita-resampler/vresampler.h>


extern "C" void vresampler_destroy (PyObject *P)
{
    delete (VResampler *) PyCapsule_GetPointer (P, "VResampler");
}


extern "C" PyObject* vresampler_create (PyObject *self, PyObject *args)
{
    double      ratio, frel;
    int         nchan, hlen;
    PyObject    *P;
    VResampler  *R = new VResampler ();

    if (! PyArg_ParseTuple (args, "Odiid", &P, &ratio, &nchan, &hlen, &frel)) return 0;
    if (R->setup (ratio, nchan, hlen, frel))
    {
	delete R;
	return 0;
    }
    return PyCapsule_New ((void *) R, "VResampler", vresampler_destroy);
}


extern "C" PyObject* vresampler_reset (PyObject *self, PyObject *args)
{ 
    VResampler  *R;
    PyObject    *P;

    if (! PyArg_ParseTuple (args, "O", &P)) return 0;
    R = (VResampler *) PyCapsule_GetPointer (P, "VResampler");
    return Py_BuildValue ("i", R->reset ());
}


extern "C" PyObject* vresampler_inpsize (PyObject *self, PyObject *args)
{ 
    VResampler  *R;
    PyObject    *P;

    if (! PyArg_ParseTuple (args, "O", &P)) return 0;
    R = (VResampler *) PyCapsule_GetPointer (P, "VResampler");
    return Py_BuildValue ("i", R->inpsize ());
}


extern "C" PyObject* vresampler_inpdist (PyObject *self, PyObject *args)
{ 
    VResampler  *R;
    PyObject    *P;

    if (! PyArg_ParseTuple (args, "O", &P)) return 0;
    R = (VResampler *) PyCapsule_GetPointer (P, "VResampler");
    return Py_BuildValue ("d", R->inpdist ());
}


extern "C" PyObject* vresampler_set_phase (PyObject *self, PyObject *args)
{ 
    VResampler  *R;
    PyObject    *P;
    double      v;

    if (! PyArg_ParseTuple (args, "Od", &P, &v)) return 0;
    R = (VResampler *) PyCapsule_GetPointer (P, "VResampler");
    R->set_phase (v);
    Py_RETURN_NONE;
}


extern "C" PyObject* vresampler_set_rrfilt (PyObject *self, PyObject *args)
{ 
    VResampler  *R;
    PyObject    *P;
    double      v;

    if (! PyArg_ParseTuple (args, "Od", &P, &v)) return 0;
    R = (VResampler *) PyCapsule_GetPointer (P, "VResampler");
    R->set_rrfilt (v);
    Py_RETURN_NONE;
}


extern "C" PyObject* vresampler_set_rratio (PyObject *self, PyObject *args)
{ 
    VResampler  *R;
    PyObject    *P;
    double      v;

    if (! PyArg_ParseTuple (args, "Od", &P, &v)) return 0;
    R = (VResampler *) PyCapsule_GetPointer (P, "VResampler");
    R->set_rratio (v);
    Py_RETURN_NONE;
}


static int check_format (Py_buffer *B, int nchan)
{
    if (strcmp (B->format, "f"))
    {
        PyErr_SetString (PyExc_TypeError, "Data type must be float32");
        return 1;
    }
    if (   ((B->ndim != 1) || (nchan != 1))
        && ((B->ndim != 2) || (nchan != B->shape [1])))
    {    
        PyErr_SetString (PyExc_TypeError, "Array shape doesn't match.");
        return 1;
    }
    if (! PyBuffer_IsContiguous (B, 'C'))
    {
        PyErr_SetString (PyExc_TypeError, "Array is not C-contiguous.");
        return 1;
    }	
    return 0;
}


extern "C" PyObject* vresampler_process (PyObject *self, PyObject *args)
{ 
    VResampler  *R;
    PyObject    *P, *X, *Y;
    Py_buffer   Bx, By;
    int         bits, ninp, nout;

    if (! PyArg_ParseTuple (args, "OOO", &P, &X, &Y)) return 0;
    R = (VResampler *) PyCapsule_GetPointer (P, "VResampler");

    if (PyLong_Check (X))
    {
        R->inp_count = PyLong_AsLong (X);
        R->inp_data = 0;
    }
    else
    {
  	bits = PyBUF_FORMAT | PyBUF_STRIDES;
        if (PyObject_GetBuffer (X, &Bx, bits)) return 0;
        if (check_format (&Bx, R->nchan ()))
        {
            PyBuffer_Release (&Bx);
	    return 0;
        }
        R->inp_count = Bx.shape [0];
        R->inp_data = (float *) Bx.buf;
    }

    if (PyLong_Check (Y))
    {
        R->out_count = PyLong_AsLong (Y);
        R->out_data = 0;
    }
    else
    {
        bits = PyBUF_FORMAT | PyBUF_STRIDES | PyBUF_WRITABLE;
        if (PyObject_GetBuffer (Y, &By, bits)) return 0;
        if (check_format (&By, R->nchan ()))
        {
            if (R->inp_data) PyBuffer_Release (&Bx);
            PyBuffer_Release (&By);
	    return 0;
        }
        R->out_count = By.shape [0];
        R->out_data = (float *) By.buf;
    }

    ninp = R->inp_count;
    nout = R->out_count;
    R->process ();
    if (R->inp_data) PyBuffer_Release (&Bx);
    if (R->out_data) PyBuffer_Release (&By);
    return Py_BuildValue ("II", ninp - R->inp_count, nout - R->out_count);
}


static PyMethodDef VResamplerMethods[] =
{
    {"create",           vresampler_create,         METH_VARARGS, "Create a VResampler."},
    {"reset",            vresampler_reset,          METH_VARARGS, "Reset resampler state."},
    {"inpsize",          vresampler_inpsize,        METH_VARARGS, "Return inpsize()."},
    {"inpdist",          vresampler_inpdist,        METH_VARARGS, "Return inpdist()."},
    {"set_phase",        vresampler_set_phase,      METH_VARARGS, "Set filter phase."},
    {"set_rrfilt",       vresampler_set_rrfilt,     METH_VARARGS, "Set rratio filter."},
    {"set_rratio",       vresampler_set_rratio,     METH_VARARGS, "Set relative ratio."},
    {"process",          vresampler_process,        METH_VARARGS, "Resample data."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef VResamplerModule = 
{
   PyModuleDef_HEAD_INIT,
   "vresampler_ext",
   NULL, 
   -1, 
   VResamplerMethods
};

PyMODINIT_FUNC PyInit_vresampler_ext(void)
{
    return PyModule_Create (&VResamplerModule);
}

#else

PyMODINIT_FUNC initvresampler_ext(void)
{
    (void) Py_InitModule ("vresampler_ext", VResamplerMethods);
}

#endif
