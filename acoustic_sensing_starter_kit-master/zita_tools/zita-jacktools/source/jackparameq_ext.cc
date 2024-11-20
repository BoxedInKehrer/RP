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
#include "jparameq.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jparameq *) PyCapsule_GetPointer (P, "Jparameq");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jparameq *J;
    PyObject *P;
    const char *client_name;
    const char *server_name;
    const char *types;
    int nchan;

    if (! PyArg_ParseTuple(args, "Oszis", &P, &client_name, &server_name,
			   &nchan, &types)) return NULL;
    J = new Jparameq (client_name, server_name, nchan, types);
    return Py_BuildValue ("NN",
			  PyCapsule_New (J, "Jparameq", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_filter (PyObject *self, PyObject *args)
{
    Jparameq  *J;
    PyObject  *P;
    int       sect;
    float     freq, gain, bandw;  

    if (! PyArg_ParseTuple(args, "Oifff", &P, &sect, &freq, &gain, &bandw)) return NULL;
    J = (Jparameq *) PyCapsule_GetPointer (P, "Jparameq");
    J->set_filter (sect, freq, gain, bandw);
    Py_RETURN_NONE;
}


extern "C" PyObject* set_bypass (PyObject *self, PyObject *args)
{
    Jparameq  *J;
    PyObject  *P;
    int       onoff;

    if (! PyArg_ParseTuple(args, "Oi", &P, &onoff)) return NULL;
    J = (Jparameq *) PyCapsule_GetPointer (P, "Jparameq");
    J->set_bypass (onoff);
    Py_RETURN_NONE;
}


extern "C" PyObject* set_gain (PyObject *self, PyObject *args)
{
    Jparameq  *J;
    PyObject  *P;
    float     gain;

    if (! PyArg_ParseTuple(args, "Of", &P, &gain)) return NULL;
    J = (Jparameq *) PyCapsule_GetPointer (P, "Jparameq");
    J->set_gain (gain);
    Py_RETURN_NONE;
}


static PyMethodDef JackParameqMethods[] =
{  
    {"makecaps",      makecaps,      METH_VARARGS, "Create object capsules."},
    {"set_filter",    set_filter,    METH_VARARGS, "Set filter parameters." },
    {"set_bypass",    set_bypass,    METH_VARARGS, "Set global bypass."     },
    {"set_gain",      set_gain,      METH_VARARGS, "Set make up gain."      },
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackParameqModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackparameq_ext",
   NULL, 
   -1, 
   JackParameqMethods
};

PyMODINIT_FUNC PyInit_jackparameq_ext(void)
{
    return PyModule_Create(&JackParameqModule);
}

#else

PyMODINIT_FUNC initjackparameq_ext(void)
{
    (void) Py_InitModule("jackparameq_ext", JackParameqMethods);
}

#endif
