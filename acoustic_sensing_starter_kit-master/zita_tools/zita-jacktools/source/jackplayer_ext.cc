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
#include "jplayer.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jplayer *) PyCapsule_GetPointer (P, "Jplayer");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jplayer    *J;
    PyObject   *P;
    const char *client_name;
    const char *server_name;
    int        nchan;

    if (! PyArg_ParseTuple(args, "Oszi", &P, &client_name, &server_name,
			   &nchan)) return NULL;
    J = new Jplayer (client_name, server_name, nchan);
    return Py_BuildValue ("NN",
			  PyCapsule_New ((void *) J, "Jplayer", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_state (PyObject *self, PyObject *args)
{
    Jplayer    *J;
    PyObject   *P;
    int        state;

    if (! PyArg_ParseTuple(args, "Oi", &P, &state)) return NULL;
    J = (Jplayer *) PyCapsule_GetPointer (P, "Jplayer");
    J->set_state (state);
    Py_RETURN_NONE;
}


extern "C" PyObject* get_posit (PyObject *self, PyObject *args)
{
    Jplayer    *J;
    PyObject   *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jplayer *) PyCapsule_GetPointer (P, "Jplayer");
    return Py_BuildValue ("iL", J->get_state(), J->get_posit());
}


extern "C" PyObject* set_posit (PyObject *self, PyObject *args)
{
    Jplayer    *J;
    PyObject   *P;
    long long  posit;

    if (! PyArg_ParseTuple(args, "OL", &P, &posit)) return NULL;
    J = (Jplayer *) PyCapsule_GetPointer (P, "Jplayer");
    J->set_posit (posit);
    Py_RETURN_NONE;
}


extern "C" PyObject* set_gain (PyObject *self, PyObject *args)
{
    Jplayer   *J;
    PyObject  *P;
    float     gain, time;  

    if (! PyArg_ParseTuple(args, "Off", &P, &gain, &time)) return NULL;
    J = (Jplayer *) PyCapsule_GetPointer (P, "Jplayer");
    J->set_gain (gain, time);
    Py_RETURN_NONE;
}


extern "C" PyObject* open_file (PyObject *self, PyObject *args)
{
    Jplayer    *J;
    PyObject   *P;
    const char *name;

    if (! PyArg_ParseTuple(args, "Os", &P, &name)) return NULL;
    J = (Jplayer *) PyCapsule_GetPointer (P, "Jplayer");
    return Py_BuildValue ("i", J->open_file(name));
}


extern "C" PyObject* close_file (PyObject *self, PyObject *args)
{
    Jplayer    *J;
    PyObject   *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jplayer *) PyCapsule_GetPointer (P, "Jplayer");
    return Py_BuildValue ("i", J->close_file());
}


extern "C" PyObject* get_file_info (PyObject *self, PyObject *args)
{
    Jplayer    *J;
    PyObject   *P;

    if (! PyArg_ParseTuple(args, "O", &P)) return NULL;
    J = (Jplayer *) PyCapsule_GetPointer (P, "Jplayer");
    return Py_BuildValue ("(iiL)", J->file_chan(), J->file_rate(), J->file_size());
}


static PyMethodDef JackPlayerMethods[] =
{
    {"makecaps",      makecaps,       METH_VARARGS, "Create object capsules."},
    {"set_state",     set_state,      METH_VARARGS, "Set new state."},
    {"get_posit",     get_posit,      METH_VARARGS, "Get current state and position."},
    {"set_posit",     set_posit,      METH_VARARGS, "Locate to new postition."},
    {"set_gain",      set_gain,       METH_VARARGS, "Set playback gain."},
    {"open_file",     open_file,      METH_VARARGS, "Open an audio file."},
    {"close_file",    close_file,     METH_VARARGS, "Close current audio file."},
    {"get_file_info", get_file_info,  METH_VARARGS, "Get (chan, rate, size)."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackPlayerModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackplayer_ext",
   NULL, 
   -1, 
   JackPlayerMethods
};

PyMODINIT_FUNC PyInit_jackplayer_ext(void)
{
    return PyModule_Create (&JackPlayerModule);
}

#else

PyMODINIT_FUNC initjackplayer_ext(void)
{
    (void) Py_InitModule ("jackplayer_ext", JackPlayerMethods);
}

#endif
