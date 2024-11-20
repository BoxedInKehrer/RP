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
#include "jiecfilt.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jiecfilt *) PyCapsule_GetPointer (P, "Jiecfilt");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jiecfilt *J;
    PyObject *P;
    const char *client_name;
    const char *server_name;
    int ninp, nout;

    if (! PyArg_ParseTuple(args, "Oszii", &P, &client_name, &server_name,
			   &ninp, &nout)) return NULL;
    J = new Jiecfilt (client_name, server_name, ninp, nout);
    return Py_BuildValue ("NN",
	 		  PyCapsule_New (J, "Jiecfilt", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_filter (PyObject *self, PyObject *args)
{
    Jiecfilt  *J;
    PyObject  *P;
    int       inp, out, type, band;  

    if (! PyArg_ParseTuple(args, "Oiiii", &P, &inp, &out, &type, &band)) return NULL;
    J = (Jiecfilt *) PyCapsule_GetPointer (P, "Jiecfilt");
    J->set_filter (inp, out, type, band);
    Py_RETURN_NONE;
}


static PyMethodDef JackiecfiltMethods[] =
{ 
    {"makecaps",     makecaps,    METH_VARARGS, "Create object capsules."},
    {"set_filter",   set_filter,  METH_VARARGS, "Set input and filter for channel."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackiecfiltModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackiecfilt_ext",
   NULL, 
   -1, 
   JackiecfiltMethods
};

PyMODINIT_FUNC PyInit_jackiecfilt_ext(void)
{
    return PyModule_Create(&JackiecfiltModule);
}

#else

PyMODINIT_FUNC initjackiecfilt_ext(void)
{
    (void) Py_InitModule("jackiecfilt_ext", JackiecfiltMethods);
}

#endif
