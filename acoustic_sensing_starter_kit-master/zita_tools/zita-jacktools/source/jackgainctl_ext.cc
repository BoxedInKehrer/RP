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
#include "jgainctl.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jgainctl *) PyCapsule_GetPointer (P, "Jgainctl");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    PyObject *P;
    Jgainctl *J;
    const char *client_name;
    const char *server_name;
    int nchan;

    if (! PyArg_ParseTuple(args, "Oszi", &P, &client_name, &server_name, &nchan)) return NULL;
    J = new Jgainctl (client_name, server_name, nchan);
    return Py_BuildValue ("NN",
			  PyCapsule_New (J, "Jgainctl", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_gain (PyObject *self, PyObject *args)
{
    PyObject  *P;
    Jgainctl  *J;
    float     gain, rate;  

    if (! PyArg_ParseTuple(args, "Off", &P, &gain, &rate)) return NULL;
    J = (Jgainctl *) PyCapsule_GetPointer (P, "Jgainctl");
    J->set_gain (gain, rate);
    Py_RETURN_NONE;
}


extern "C" PyObject* set_muted (PyObject *self, PyObject *args)
{
    PyObject  *P;
    Jgainctl  *J;
    int       onoff;

    if (! PyArg_ParseTuple(args, "Oi", &P, &onoff)) return NULL;
    J = (Jgainctl *) PyCapsule_GetPointer (P, "Jgainctl");
    J->set_muted (onoff);
    Py_RETURN_NONE;
}


static PyMethodDef JackGainctlMethods[] =
{ 
    {"makecaps",   makecaps,    METH_VARARGS, "Create object capsules."  },
    {"set_gain",   set_gain,    METH_VARARGS, "Set gain and change rate."},
    {"set_muted",  set_muted,   METH_VARARGS, "Set muted state on or off"},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackGainctlModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackgainctl_ext",
   NULL, 
   -1, 
   JackGainctlMethods
};

PyMODINIT_FUNC PyInit_jackgainctl_ext(void)
{
    return PyModule_Create(&JackGainctlModule);
}

#else

PyMODINIT_FUNC initjackgainctl_ext(void)
{
    (void) Py_InitModule("jackgainctl_ext", JackGainctlMethods);
}

#endif
