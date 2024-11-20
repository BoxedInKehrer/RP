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
#include <zita-jclient.h>


extern "C" PyObject* get_state (PyObject *self, PyObject *args)
{
    Jclient  *J;
    PyObject *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jclient *) PyCapsule_GetPointer (P, "Jclient");
    return Py_BuildValue ("i", J->get_state());
}


extern "C" PyObject* get_jack_info (PyObject *self, PyObject *args)
{
    Jclient  *J;
    PyObject *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jclient *) PyCapsule_GetPointer (P, "Jclient");
    return Py_BuildValue ("sii", J->jack_name(), J->jack_rate(), J->jack_size());
}


extern "C" PyObject* port_operation (PyObject *self, PyObject *args)
{
    Jclient  *J;
    PyObject *P;
    int      opc, ind;
    char     *name;
    
    if (! PyArg_ParseTuple(args, "Oiiz", &P, &opc, &ind, &name)) return NULL;
    J = (Jclient *) PyCapsule_GetPointer (P, "Jclient");
    return Py_BuildValue ("i", J->port_operation (opc, ind, name));
}


static PyMethodDef JackClientMethods[] =
{
    {"get_state",         get_state,        METH_VARARGS, "Get current state"     },
    {"get_jack_info",     get_jack_info,    METH_VARARGS, "Get Jack server info." },
    {"port_operation",    port_operation,   METH_VARARGS, "Port operations."      },     
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackClientModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackclient_ext",
   NULL, 
   -1, 
   JackClientMethods
};

PyMODINIT_FUNC PyInit_jackclient_ext(void)
{
    return PyModule_Create(&JackClientModule);
}

#else

PyMODINIT_FUNC initjackclient_ext(void)
{
    (void) Py_InitModule("jackclient_ext", JackClientMethods);
}

#endif
