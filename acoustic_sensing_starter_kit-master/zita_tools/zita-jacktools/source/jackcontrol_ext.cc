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
#include "jcontrol.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jcontrol *) PyCapsule_GetPointer (P, 0);
}


extern "C" PyObject* create (PyObject *self, PyObject *args)
{
    Jcontrol  *J;
    PyObject   *P;
    const char *client_name;
    const char *server_name;

    if (! PyArg_ParseTuple(args, "Osz", &P, &client_name, &server_name)) return NULL;
    J = new Jcontrol (client_name, server_name);
    return PyCapsule_New((void *) J, 0, destroy);
}


extern "C" PyObject* get_jack_info (PyObject *self, PyObject *args)
{
    Jcontrol  *J;
    PyObject   *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jcontrol *) PyCapsule_GetPointer (P, 0);
    return Py_BuildValue ("(sii)", J->jack_name(), J->jack_rate(), J->jack_size());
}


extern "C" PyObject* get_transport_state (PyObject *self, PyObject *args)
{
    Jcontrol  *J;
    PyObject    *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jcontrol *) PyCapsule_GetPointer (P, 0);
    return Py_BuildValue ("(ii)", J->transport_state(), J->transport_frame());
}


extern "C" PyObject* transport_start (PyObject *self, PyObject *args)
{
    Jcontrol  *J;
    PyObject    *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jcontrol *) PyCapsule_GetPointer (P, 0);
    J->transport_start ();
    Py_RETURN_NONE;
}


extern "C" PyObject* transport_stop (PyObject *self, PyObject *args)
{
    Jcontrol  *J;
    PyObject    *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jcontrol *) PyCapsule_GetPointer (P, 0);
    J->transport_stop ();
    Py_RETURN_NONE;
}


extern "C" PyObject* transport_locate (PyObject *self, PyObject *args)
{
    Jcontrol  *J;
    PyObject    *P;
    unsigned int frame;

    if (! PyArg_ParseTuple(args, "Oi", &P, &frame)) return NULL;
    J = (Jcontrol *) PyCapsule_GetPointer (P, 0);
    J->transport_locate (frame);
    Py_RETURN_NONE;
}


extern "C" PyObject* connect_ports (PyObject *self, PyObject *args)
{
    Jcontrol  *J;
    PyObject    *P;
    const char  *srce, *dest;

    if (! PyArg_ParseTuple(args, "Oss", &P, &srce, &dest)) return NULL;
    J = (Jcontrol *) PyCapsule_GetPointer (P, 0);
    return Py_BuildValue ("i", J->connect_ports (srce, dest));
}


extern "C" PyObject* disconn_ports (PyObject *self, PyObject *args)
{
    Jcontrol  *J;
    PyObject    *P;
    const char  *srce, *dest;

    if (! PyArg_ParseTuple(args, "Oss", &P, &srce, &dest)) return NULL;
    J = (Jcontrol *) PyCapsule_GetPointer (P, 0);
    return Py_BuildValue ("i", J->disconn_ports (srce, dest));
}


static PyMethodDef JackControlMethods[] =
{
    {"create",              create,              METH_VARARGS, "Create a JackControl."},
    {"get_jack_info",       get_jack_info,       METH_VARARGS, "Get (name, fsamp, fsize)."},
    {"get_transport_state", get_transport_state, METH_VARARGS, "Get current transport state."},
    {"transport_start",     transport_start,     METH_VARARGS, "Start jack transport" },
    {"transport_stop",      transport_stop,      METH_VARARGS, "Stop jack transport" },
    {"transport_locate",    transport_locate,    METH_VARARGS, "Locate jack transport" }, 
    {"connect_ports",       connect_ports,       METH_VARARGS, "Connect jack ports" },
    {"disconn_ports",       disconn_ports,       METH_VARARGS, "Disconnect jack ports" },
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackControlModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackcontrol_ext",
   NULL, 
   -1, 
   JackControlMethods
};

PyMODINIT_FUNC PyInit_jackcontrol_ext(void)
{
    return PyModule_Create(&JackControlModule);
}

#else

PyMODINIT_FUNC initjackcontrol_ext(void)
{
    (void) Py_InitModule ("jackcontrol_ext", JackControlMethods);
}

#endif
