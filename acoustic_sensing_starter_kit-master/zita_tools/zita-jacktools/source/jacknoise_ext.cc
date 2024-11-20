// ----------------------------------------------------------------------------
//
//  Copyright (C) 2013-2018 Fons Adriaensen <fons@linuxaudio.org>
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
#include "jnoise.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jnoise *) PyCapsule_GetPointer (P, "Jnoise");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jnoise *J;
    PyObject *P;
    const char *client_name;
    const char *server_name;
    int nchan;

    if (! PyArg_ParseTuple(args, "Oszi", &P, &client_name, &server_name, &nchan)) return NULL;
    J = new Jnoise (client_name, server_name, nchan);
    return Py_BuildValue ("NN",
	 		  PyCapsule_New (J, "Jnoise", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_output (PyObject *self, PyObject *args)
{
    Jnoise  *J;
    PyObject  *P;
    int       chan, type;
    float     level;

    if (! PyArg_ParseTuple(args, "Oiif", &P, &chan, &type, &level)) return NULL;
    J = (Jnoise *) PyCapsule_GetPointer (P, "Jnoise");
    J->set_output (chan, type, level);
    Py_RETURN_NONE;
}


static PyMethodDef JacknoiseMethods[] =
{ 
    {"makecaps",    makecaps,    METH_VARARGS, "Create object capsules."},
    {"set_output",  set_output,  METH_VARARGS, "Set noise type and level."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JacknoiseModule = 
{
   PyModuleDef_HEAD_INIT,
   "jacknoise_ext",
   NULL, 
   -1, 
   JacknoiseMethods
};

PyMODINIT_FUNC PyInit_jacknoise_ext(void)
{
    return PyModule_Create(&JacknoiseModule);
}

#else

PyMODINIT_FUNC initjacknoise_ext(void)
{
    (void) Py_InitModule("jacknoise_ext", JacknoiseMethods);
}

#endif
