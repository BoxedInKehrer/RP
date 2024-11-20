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
#include "jambpan.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jambpan *) PyCapsule_GetPointer (P, "Jambpan");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jambpan *J;
    PyObject *P;
    const char *client_name;
    const char *server_name;
    int degree;

    if (! PyArg_ParseTuple(args, "Oszi", &P, &client_name, &server_name, &degree)) return NULL;
    J = new Jambpan (client_name, server_name, degree);
    return Py_BuildValue ("NN",
			  PyCapsule_New (J, "Jambpan", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_direction (PyObject *self, PyObject *args)
{
    Jambpan  *J;
    PyObject  *P;
    float     azim, elev, time;  

    if (! PyArg_ParseTuple(args, "Offf", &P, &azim, &elev, &time)) return NULL;
    J = (Jambpan *) PyCapsule_GetPointer (P, "Jambpan");
    J->set_direction (azim, elev, time);
    Py_RETURN_NONE;
}


static PyMethodDef JackAmbpanMethods[] =
{ 
    {"makecaps",       makecaps,       METH_VARARGS, "Create object capsules."},
    {"set_direction",  set_direction,  METH_VARARGS, "Set panning direction."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackAmbpanModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackambpan_ext",
   NULL, 
   -1, 
   JackAmbpanMethods
};

PyMODINIT_FUNC PyInit_jackambpan_ext(void)
{
    return PyModule_Create(&JackAmbpanModule);
}

#else

PyMODINIT_FUNC initjackambpan_ext(void)
{
    (void) Py_InitModule("jackambpan_ext", JackAmbpanMethods);
}

#endif
