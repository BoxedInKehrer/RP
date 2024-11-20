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


#include <Python.h>
#include "jkmeter.h"


static float *checkbuff (PyObject *P, int nchan)
{
    Py_buffer B;
    float    *F = 0;

    if (   PyObject_CheckBuffer (P)
	&& (PyObject_GetBuffer(P, &B, PyBUF_FULL) == 0)
        && (B.ndim == 1)
        && (B.shape [0] == nchan)   
        && (B.strides [0] == sizeof (float))) F = (float *)(B.buf);
    // The Python code MUS NOT DELETE this object !!
    PyBuffer_Release (&B);
    return F;
}


extern "C" void destroy (PyObject *P)
{
    delete (Jkmeter *) PyCapsule_GetPointer (P, "Jkmeter");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jkmeter  *J;
    PyObject *P, *Prms, *Ppks;
    const char *client_name;
    const char *server_name;
    int   nchan;
    float *rms;
    float *pks;
    
    if (! PyArg_ParseTuple(args, "OsziOO", &P, &client_name, &server_name, &nchan, &Prms, &Ppks)) return 0;
    rms = checkbuff (Prms, nchan);
    pks = checkbuff (Ppks, nchan);
    if (!rms || !pks) return 0;
    J = new Jkmeter (client_name, server_name, nchan, rms, pks);
    return Py_BuildValue ("NN",
			  PyCapsule_New (J, "Jkmeter", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* get_levels (PyObject *self, PyObject *args)
{
    Jkmeter  *J;
    PyObject *P;
    
    if (! PyArg_ParseTuple(args, "O", &P)) return 0;
    J = (Jkmeter *) PyCapsule_GetPointer (P, "Jkmeter");
    return Py_BuildValue ("i", J->get_levels ());
}


static PyMethodDef JackKmeterMethods[] =
{
    {"makecaps",      makecaps,     METH_VARARGS, "Create object capsules."},
    {"get_levels",    get_levels,   METH_VARARGS, "Get current state and levels."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackKmeterModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackkmeter_ext",
   NULL, 
   -1, 
   JackKmeterMethods
};

PyMODINIT_FUNC PyInit_jackkmeter_ext(void)
{
    return PyModule_Create(&JackKmeterModule);
}

#else

PyMODINIT_FUNC initjackkmeter_ext(void)
{
    (void) Py_InitModule("jackkmeter_ext", JackKmeterMethods);
}

#endif
