// ----------------------------------------------------------------------------
//
//  Copyright (C) 2008-2015 Fons Adriaensen <fons@linuxaudio.org>
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
#include "jsignal.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jsignal *) PyCapsule_GetPointer (P, "Jsignal");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jsignal  *J;
    PyObject *P;
    const char *client_name;
    const char *server_name;

    if (! PyArg_ParseTuple(args, "Osz", &P, &client_name, &server_name)) return NULL;
    J = new Jsignal (client_name, server_name);
    return Py_BuildValue ("NN",
			  PyCapsule_New ((void *) J, "Jsignal", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_state (PyObject *self, PyObject *args)
{
    Jsignal   *J;
    PyObject  *P;
    int       state;

    if (! PyArg_ParseTuple(args, "Oi", &P, &state)) return NULL;
    J = (Jsignal *) PyCapsule_GetPointer (P, "Jsignal");
    J->set_state (state);
    Py_RETURN_NONE;
}


extern "C" PyObject* get_posit (PyObject *self, PyObject *args)
{
    Jsignal   *J;
    PyObject  *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jsignal *) PyCapsule_GetPointer (P, "Jsignal");
    return Py_BuildValue ("iL", J->get_state (), J->get_posit ());
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


extern "C" PyObject* set_input_data (PyObject *self, PyObject *args)
{ 
    Jsignal    *J;
    PyObject   *P, *Q;
    Py_buffer  B;
    int        ind, bits, nloop, nskip;

    if (! PyArg_ParseTuple (args, "OiOii", &P, &ind, &Q, &nloop, &nskip)) return 0;
    J = (Jsignal *) PyCapsule_GetPointer (P, "Jsignal");
    if (Q == Py_None) J->set_inp_data (ind, 0, 0, 0, 0);
    else
    {
        bits = PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE;
	if (PyObject_GetBuffer (Q, &B, bits)) return 0;
        if (check_format (&B))
        {
            PyBuffer_Release (&B);
	    return 0;
        }
        J->set_inp_data (ind, Q, bits, nloop, nskip);
        PyBuffer_Release (&B);
    }
    Py_RETURN_NONE;
}


extern "C" PyObject* set_output_data (PyObject *self, PyObject *args)
{ 
    Jsignal    *J;
    PyObject   *P, *Q;
    Py_buffer  B;
    int        ind, bits, nloop, nskip;

    if (! PyArg_ParseTuple (args, "OiOii", &P, &ind, &Q, &nloop, &nskip)) return 0;
    J = (Jsignal *) PyCapsule_GetPointer (P, "Jsignal");
    if (Q == Py_None) J->set_out_data (ind, 0, 0, 0, 0);
    else
    {
        bits = PyBUF_STRIDES | PyBUF_FORMAT;
	if (PyObject_GetBuffer (Q, &B, bits)) return 0;
        if (check_format (&B))
        {
            PyBuffer_Release (&B);
	    return 0;
        }
        J->set_out_data (ind, Q, bits, nloop, nskip);
        PyBuffer_Release (&B);
    }
    Py_RETURN_NONE;
}


extern "C" PyObject* set_trigger_inp (PyObject *self, PyObject *args)
{
    Jsignal   *J;
    PyObject  *P;
    int       ind;

    if (! PyArg_ParseTuple(args, "Oi", &P, &ind)) return NULL;
    J = (Jsignal *) PyCapsule_GetPointer (P, "Jsignal");
    J->set_trig_inp (ind);
    Py_RETURN_NONE;
}


static PyMethodDef JackSignalMethods[] =
{
    {"makecaps",          makecaps,         METH_VARARGS, "Create object capsules."},
    {"set_state",         set_state,        METH_VARARGS, "Set new jack state."},
    {"set_input_data",    set_input_data,   METH_VARARGS, "Set input buffer."},
    {"set_output_data",   set_output_data,  METH_VARARGS, "Set output buffer."},
    {"set_trigger_inp",   set_trigger_inp,  METH_VARARGS, "Select trigger input."},
    {"get_posit",         get_posit,        METH_VARARGS, "Get current state and position."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackSignalModule = 
{
   PyModuleDef_HEAD_INIT,
   "jacksignal_ext",
   NULL, 
   -1, 
   JackSignalMethods
};

PyMODINIT_FUNC PyInit_jacksignal_ext(void)
{
    return PyModule_Create(&JackSignalModule);
}

#else

PyMODINIT_FUNC initjacksignal_ext(void)
{
    (void) Py_InitModule("jacksignal_ext", JackSignalMethods);
}

#endif
