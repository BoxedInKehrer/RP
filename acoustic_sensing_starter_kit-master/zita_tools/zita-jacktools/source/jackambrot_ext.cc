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
#include "jambrot.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jambrot *) PyCapsule_GetPointer (P, "Jambrot");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jambrot *J;
    PyObject *P;
    const char *client_name;
    const char *server_name;
    int degree;

    if (! PyArg_ParseTuple(args, "Oszi", &P, &client_name, &server_name, &degree)) return NULL;
    J = new Jambrot (client_name, server_name, degree);
    return Py_BuildValue ("NN",
			  PyCapsule_New (J, "Jambrot", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_quaternion (PyObject *self, PyObject *args)
{
    Jambrot  *J;
    PyObject  *P;
    float     w, x, y, z, t;  

    if (! PyArg_ParseTuple(args, "Offfff", &P, &w, &x, &y, &z, &t)) return NULL;
    J = (Jambrot *) PyCapsule_GetPointer (P, "Jambrot");
    J->set_quaternion (w, x, y, z, t);
    Py_RETURN_NONE;
}


static PyMethodDef JackAmbrotMethods[] =
{ 
    {"makecaps",        makecaps,        METH_VARARGS, "Create object capsules."},
    {"set_quaternion",  set_quaternion,  METH_VARARGS, "Set rotation quaternion."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackAmbrotModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackambrot_ext",
   NULL, 
   -1, 
   JackAmbrotMethods
};

PyMODINIT_FUNC PyInit_jackambrot_ext(void)
{
    return PyModule_Create(&JackAmbrotModule);
}

#else

PyMODINIT_FUNC initjackambrot_ext(void)
{
    (void) Py_InitModule("jackambrot_ext", JackAmbrotMethods);
}

#endif
