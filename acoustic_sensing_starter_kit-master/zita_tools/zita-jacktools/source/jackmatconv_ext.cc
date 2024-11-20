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
#include "jmatconv.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jmatconv *) PyCapsule_GetPointer (P, "Jmatconv");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jmatconv  *J;
    PyObject *P;
    int size, ninp, nout, nthr;
    const char *client_name;
    const char *server_name;

    if (! PyArg_ParseTuple (args, "Osziiii", &P, &client_name, &server_name,
                            &size, &ninp, &nout, &nthr)) return NULL;
    J = new Jmatconv (client_name, server_name, size, ninp, nout, nthr);
    return Py_BuildValue ("NN",
			  PyCapsule_New ((void *) J, "Jmatconv", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_state (PyObject *self, PyObject *args)
{
    Jmatconv   *J;
    PyObject  *P;
    int       state;

    if (! PyArg_ParseTuple(args, "Oi", &P, &state)) return NULL;
    J = (Jmatconv *) PyCapsule_GetPointer (P, "Jmatconv");
    J->set_state (state);
    Py_RETURN_NONE;
}


static int check_format (Py_buffer *B)
{
    if (strcmp (B->format, "f"))     
    {
        PyErr_SetString (PyExc_TypeError, "Data type must be float32");
        return 1;
    }
    if (B->ndim != 1)     
    {
        PyErr_SetString (PyExc_TypeError, "Array must be single dimension");
        return 1;
    }
    return 0;
}


extern "C" PyObject* load_impulse (PyObject *self, PyObject *args)
{ 
    Jmatconv   *J;
    PyObject   *P, *Q;
    Py_buffer  B;
    int        inp, out, rv;
    float      gain;
    
    if (! PyArg_ParseTuple (args, "OOiif", &P, &Q, &inp, &out, &gain)) return 0;
    J = (Jmatconv *) PyCapsule_GetPointer (P, "Jmatconv");
    if (Q == Py_None) rv = J->convproc ()->setimp (inp, out, 0.0f, 0, 0, 0);
    else
    {
        if (PyObject_GetBuffer (Q, &B, PyBUF_FULL_RO)) return 0;
        if (check_format (&B))
        {
            PyBuffer_Release (&B);
            return 0;
	    
        }
        rv = J->convproc ()->setimp (inp, out, gain, (float *)(B.buf), B.shape [0], B.strides [0] / sizeof (float));
        PyBuffer_Release (&B);
    }
    return Py_BuildValue ("i", rv);
}


static PyMethodDef JackMatconvMethods[] =
{
    {"makecaps",      makecaps,      METH_VARARGS, "Create object capsules."},
    {"set_state",     set_state,     METH_VARARGS, "Set new state."},
    {"load_impulse",  load_impulse,  METH_VARARGS, "Load single impulse."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackMatconvModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackmatconv_ext",
   NULL, 
   -1, 
   JackMatconvMethods
};

PyMODINIT_FUNC PyInit_jackmatconv_ext(void)
{
    return PyModule_Create(&JackMatconvModule);
}

#else

PyMODINIT_FUNC initjackmatconv_ext(void)
{
    (void) Py_InitModule("jackmatconv_ext", JackMatconvMethods);
}

#endif
