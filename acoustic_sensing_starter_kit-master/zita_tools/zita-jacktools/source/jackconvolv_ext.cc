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
#include "jconvolv.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jconvolv *) PyCapsule_GetPointer (P, "Jconvolv");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jconvolv  *J;
    PyObject  *P;
    uint32_t   ninp, nout;
    const char *client_name;
    const char *server_name;

    if (! PyArg_ParseTuple (args, "Oszii", &P, &client_name, &server_name,
                            &ninp, &nout)) return NULL;
    J = new Jconvolv (client_name, server_name, ninp, nout);
    return Py_BuildValue ("NN",
			  PyCapsule_New ((void *) J, "Jconvolv", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_state (PyObject *self, PyObject *args)
{
    Jconvolv  *J;
    PyObject  *P;
    int       state;

    if (! PyArg_ParseTuple(args, "Oi", &P, &state)) return NULL;
    J = (Jconvolv *) PyCapsule_GetPointer (P, "Jconvolv");
    J->set_state (state);
    Py_RETURN_NONE;
}


extern "C" PyObject* configure (PyObject *self, PyObject *args)
{
    Jconvolv  *J;
    PyObject  *P;
    int       maxlen, rv;
    float     density;

    if (! PyArg_ParseTuple(args, "Oif", &P, &maxlen, &density)) return NULL;
    J = (Jconvolv *) PyCapsule_GetPointer (P, "Jconvolv");
    if (maxlen) rv = J->configure (maxlen, density);
    else        rv = J->convproc ()->cleanup ();
    return Py_BuildValue ("i", rv);
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


extern "C" PyObject* impdata_write (PyObject *self, PyObject *args)
{ 
    Jconvolv   *J;
    PyObject   *P, *Q;
    Py_buffer  B;
    Convproc   *C;
    int        inp, out, offs, size, step, cnew, rv = 0;
    float      *data;
    
    if (! PyArg_ParseTuple (args, "OOiiii", &P, &Q, &inp, &out, &offs, &cnew)) return 0;
    J = (Jconvolv *) PyCapsule_GetPointer (P, "Jconvolv");
    C = J->convproc ();
    if (Q == Py_None) rv = C->impdata_clear (inp, out);
    else	 
    {
        if (PyObject_GetBuffer (Q, &B, PyBUF_FULL_RO)) return 0;
        if (check_format (&B))
        {
            PyBuffer_Release (&B);
            return 0;
        }
	data = (float *)(B.buf);
	size = B.shape [0];
	step = B.strides [0] / sizeof (float);
        if (cnew) rv = C->impdata_create (inp, out, step, data, offs, offs + size);
	else      rv = C->impdata_update (inp, out, step, data, offs, offs + size);
        PyBuffer_Release (&B);
    }
    return Py_BuildValue ("i", rv);
}


extern "C" PyObject* impdata_link (PyObject *self, PyObject *args)
{ 
    Jconvolv   *J;
    PyObject   *P;
    int        inp1, out1, inp2, out2, rv;
    
    if (! PyArg_ParseTuple (args, "Oiiii", &P, &inp1, &out1, &inp2, &out2)) return 0;
    J = (Jconvolv *) PyCapsule_GetPointer (P, "Jconvolv");
    rv = J->convproc ()->impdata_link (inp1, out1, inp2, out2);
    return Py_BuildValue ("i", rv);
}


static PyMethodDef JackConvolvMethods[] =
{
    {"makecaps",      makecaps,       METH_VARARGS, "Create objects."},
    {"set_state",     set_state,      METH_VARARGS, "Set new state." },
    {"configure",     configure,      METH_VARARGS, "Configure."     },
    {"impdata_write", impdata_write,  METH_VARARGS, "Write IR data." },
    {"impdata_link",  impdata_link,   METH_VARARGS, "Link IR data."  },
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackConvolvModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackconvolv_ext",
   NULL, 
   -1, 
   JackConvolvMethods
};

PyMODINIT_FUNC PyInit_jackconvolv_ext(void)
{
    return PyModule_Create(&JackConvolvModule);
}

#else

PyMODINIT_FUNC initjackconvolv_ext(void)
{
    (void) Py_InitModule("jackconvolv_ext", JackConvolvMethods);
}

#endif
