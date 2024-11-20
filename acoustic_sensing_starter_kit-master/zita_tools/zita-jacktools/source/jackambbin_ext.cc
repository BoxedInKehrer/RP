// ----------------------------------------------------------------------------
//
//  Copyright (C) 2015-2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include "jambbin.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jambbin *) PyCapsule_GetPointer (P, "Jambbin");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jambbin   *J;
    PyObject  *P;
    const char *client_name;
    const char *server_name;
    int maxlen;
    int degree;
    
    if (! PyArg_ParseTuple(args, "Oszii", &P, &client_name, &server_name,
			   &maxlen, &degree)) return NULL;
    J = new Jambbin (client_name, server_name, maxlen, degree);
    return Py_BuildValue ("NN",
			  PyCapsule_New (J, "Jambbin", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_nfcomp (PyObject *self, PyObject *args)
{
    PyObject  *P;
    Jambbin   *J;
    float     d;

    if (! PyArg_ParseTuple(args, "Of", &P, &d)) return 0;
    J = (Jambbin *) PyCapsule_GetPointer (P, "Jambbin");
    J->ambbin ()->set_nfcomp (d);
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


extern "C" PyObject* set_filter (PyObject *self, PyObject *args)
{
    PyObject     *P, *Q;
    Jambbin      *J;
    Py_buffer    B;
    int          harm;
    
    if (! PyArg_ParseTuple (args, "OiO", &P, &harm, &Q)) return 0;
    J = (Jambbin *) PyCapsule_GetPointer (P, "Jambbin");
    if (Q == Py_None) J->ambbin ()->set_filter (harm, 0, 0, 0);
    else
    {
        if (PyObject_GetBuffer (Q, &B, PyBUF_FULL_RO)) return 0;
        if (check_format (&B))
        {
            PyBuffer_Release (&B);
            return 0;
        }
        J->ambbin ()->set_filter (harm, (float *)(B.buf), B.shape [0], B.strides [0] / sizeof (float));
        PyBuffer_Release (&B);
    }
    return Py_BuildValue ("i", 0);
}


extern "C" PyObject* set_quaternion (PyObject *self, PyObject *args)
{
    PyObject  *P;
    Jambbin   *J;
    float     w, x, y, z, t;

    if (! PyArg_ParseTuple(args, "Offfff", &P, &w, &x, &y, &z, &t)) return 0;
    J = (Jambbin *) PyCapsule_GetPointer (P, "Jambbin");
    J->ambbin ()->set_quaternion (w, x, y, z, t);
    return Py_BuildValue ("i", 0);
}


static PyMethodDef JackAmbbinMethods[] =
{ 
    {"makecaps",        makecaps,        METH_VARARGS,  "Create object capsules."  },
    {"set_nfcomp",      set_nfcomp,      METH_VARARGS,  "Set NF compensation."     },
    {"set_filter",      set_filter,      METH_VARARGS,  "Set rendering filter."    },
    {"set_quaternion",  set_quaternion,  METH_VARARGS,  "Set rotation quaternion." },
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackAmbbinModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackambbin_ext",
   NULL, 
   -1, 
   JackAmbbinMethods
};

PyMODINIT_FUNC PyInit_jackambbin_ext(void)
{
    return PyModule_Create(&JackAmbbinModule);
}

#else

PyMODINIT_FUNC initjackambbin_ext(void)
{
    (void) Py_InitModule("jackambbin_ext", JackAmbbinMethods);
}

#endif
