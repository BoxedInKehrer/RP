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
#include "jmatrix.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jmatrix *) PyCapsule_GetPointer (P, "Jmatrix");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jmatrix   *J;
    PyObject  *P;
    const char *client_name;
    const char *server_name;
    int ninp, nout;

    if (! PyArg_ParseTuple(args, "Oszii", &P, &client_name, &server_name, &ninp, &nout)) return NULL;
    J = new Jmatrix (client_name, server_name, ninp, nout);
    return Py_BuildValue ("NN",
	 		  PyCapsule_New (J, "Jmatrix", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_gain (PyObject *self, PyObject *args)
{
    Jmatrix   *J;
    PyObject  *P;
    int       inp, out;
    float     gain;  

    if (! PyArg_ParseTuple(args, "Oiif", &P, &inp, &out, &gain)) return NULL;
    J = (Jmatrix *) PyCapsule_GetPointer (P, "Jmatrix");
    J->set_gain (inp, out, gain);
    Py_RETURN_NONE;
}


static PyMethodDef JackMatrixMethods[] =
{ 
    {"makecaps",   makecaps,     METH_VARARGS, "Create object capsules."},
    {"set_gain",   set_gain,     METH_VARARGS, "Set gain."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackMatrixModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackmatrix_ext",
   NULL, 
   -1, 
   JackMatrixMethods
};

PyMODINIT_FUNC PyInit_jackmatrix_ext(void)
{
    return PyModule_Create(&JackMatrixModule);
}

#else

PyMODINIT_FUNC initjackmatrix_ext(void)
{
    (void) Py_InitModule("jackmatrix_ext", JackMatrixMethods);
}

#endif
