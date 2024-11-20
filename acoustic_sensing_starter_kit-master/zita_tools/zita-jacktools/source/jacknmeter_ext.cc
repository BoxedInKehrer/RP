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
#include "jnmeter.h"


static float *checkbuff (PyObject *P, int dim0)
{
    Py_buffer B;
    float    *F = 0;

    if (   PyObject_CheckBuffer (P)
	&& (PyObject_GetBuffer(P, &B, PyBUF_FULL) == 0)
        && (B.ndim == 1)
        && (B.shape [0] == dim0)   
        && (B.strides [0] == sizeof (float))) F = (float *)(B.buf);
    // The Python code MUS NOT DELETE this object !!
    PyBuffer_Release (&B);
    return F;
}


extern "C" void destroy (PyObject *P)
{
    delete (Jnmeter *) PyCapsule_GetPointer (P, "Jnmeter");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jnmeter  *J;
    PyObject *P, *L;
    const char *client_name;
    const char *server_name;
    int   ninp, nout;
    float *levels;
    
    if (! PyArg_ParseTuple(args, "OsziiO", &P, &client_name, &server_name,
			   &ninp, &nout, &L)) return 0;
    levels = checkbuff (L, nout);
    if (!levels) return 0;
    J = new Jnmeter (client_name, server_name, ninp, nout, levels);
    return Py_BuildValue ("NN",
			  PyCapsule_New (J, "Jnmeter", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_input (PyObject *self, PyObject *args)
{
    Jnmeter  *J;
    PyObject *P;
    int inp, out;
    
    if (! PyArg_ParseTuple(args, "Oii", &P, &inp, &out)) return 0;
    J = (Jnmeter *) PyCapsule_GetPointer (P, "Jnmeter");
    return Py_BuildValue ("i", J->set_input (inp, out));
}


extern "C" PyObject* set_filter (PyObject *self, PyObject *args)
{
    Jnmeter  *J;
    PyObject *P;
    int out, ftype, dcfilt;
    
    if (! PyArg_ParseTuple(args, "Oiii", &P, &out, &ftype, &dcfilt)) return 0;
    J = (Jnmeter *) PyCapsule_GetPointer (P, "Jnmeter");
    return Py_BuildValue ("i", J->set_filter (out, ftype, dcfilt));
}


extern "C" PyObject* set_detect (PyObject *self, PyObject *args)
{
    Jnmeter  *J;
    PyObject *P;
    int out, dtype;
    
    if (! PyArg_ParseTuple(args, "Oii", &P, &out, &dtype)) return 0;
    J = (Jnmeter *) PyCapsule_GetPointer (P, "Jnmeter");
    return Py_BuildValue ("i", J->set_detect (out, dtype));
}


extern "C" PyObject* get_levels (PyObject *self, PyObject *args)
{
    Jnmeter  *J;
    PyObject *P;
    
    if (! PyArg_ParseTuple(args, "O", &P)) return 0;
    J = (Jnmeter *) PyCapsule_GetPointer (P, "Jnmeter");
    return Py_BuildValue ("i", J->get_levels ());
}


static PyMethodDef JackNmeterMethods[] =
{
    {"makecaps",      makecaps,      METH_VARARGS, "Create object capsules."},
    {"set_input",     set_input,     METH_VARARGS, "Select input."},
    {"set_filter",    set_filter,    METH_VARARGS, "Select filter type."},
    {"set_detect",    set_detect,    METH_VARARGS, "Select detector type."},
    {"get_levels",    get_levels,    METH_VARARGS, "Get current state and levels."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackNmeterModule = 
{
   PyModuleDef_HEAD_INIT,
   "jacknmeter_ext",
   NULL, 
   -1, 
   JackNmeterMethods
};

PyMODINIT_FUNC PyInit_jacknmeter_ext(void)
{
    return PyModule_Create(&JackNmeterModule);
}

#else

PyMODINIT_FUNC initjacknmeter_ext(void)
{
    (void) Py_InitModule("jacknmeter_ext", JackNmeterMethods);
}

#endif
