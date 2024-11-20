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
#include "jpeaklim.h"


extern "C" void destroy (PyObject *P)
{
    delete (Jpeaklim *) PyCapsule_GetPointer (P, "Jpeaklim");
}


extern "C" PyObject* makecaps (PyObject *self, PyObject *args)
{
    Jpeaklim *J;
    PyObject *P;
    const char *client_name;
    const char *server_name;
    int nchan;

    if (! PyArg_ParseTuple(args, "Oszi", &P, &client_name, &server_name, &nchan)) return NULL;
    J = new Jpeaklim (client_name, server_name, nchan);
    return Py_BuildValue ("NN",
			  PyCapsule_New (J, "Jpeaklim", destroy),
                          PyCapsule_New (dynamic_cast<Jclient *>(J), "Jclient", 0));
}


extern "C" PyObject* set_inpgain (PyObject *self, PyObject *args)
{
    Jpeaklim  *J;
    PyObject  *P;
    float     inpgain;  

    if (! PyArg_ParseTuple(args, "Of", &P, &inpgain)) return NULL;
    J = (Jpeaklim *) PyCapsule_GetPointer (P, "Jpeaklim");
    J->peaklim ()->set_inpgain (inpgain);
    Py_RETURN_NONE;
}


extern "C" PyObject* set_threshold (PyObject *self, PyObject *args)
{
    Jpeaklim  *J;
    PyObject  *P;
    float     threshold;  

    if (! PyArg_ParseTuple(args, "Of", &P, &threshold)) return NULL;
    J = (Jpeaklim *) PyCapsule_GetPointer (P, "Jpeaklim");
    J->peaklim ()->set_threshold (threshold);
    Py_RETURN_NONE;
}


extern "C" PyObject* set_release (PyObject *self, PyObject *args)
{
    Jpeaklim  *J;
    PyObject  *P;
    float     release;  

    if (! PyArg_ParseTuple(args, "Of", &P, &release)) return NULL;
    J = (Jpeaklim *) PyCapsule_GetPointer (P, "Jpeaklim");
    J->peaklim ()->set_release (release);
    Py_RETURN_NONE;
}


static PyMethodDef JackPeaklimMethods[] =
{ 
    {"makecaps",      makecaps,       METH_VARARGS, "Create object capsules."},
    {"set_inpgain",   set_inpgain,    METH_VARARGS, "Set input gain."},
    {"set_threshold", set_threshold,  METH_VARARGS, "Set threshold."},
    {"set_release",   set_release,    METH_VARARGS, "Set release time."},
    {NULL, NULL, 0, NULL}
};



#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef JackPeaklimModule = 
{
   PyModuleDef_HEAD_INIT,
   "jackpeaklim_ext",
   NULL, 
   -1, 
   JackPeaklimMethods
};

PyMODINIT_FUNC PyInit_jackpeaklim_ext(void)
{
    return PyModule_Create(&JackPeaklimModule);
}

#else

PyMODINIT_FUNC initjackpeaklim_ext(void)
{
    (void) Py_InitModule("jackpeaklim_ext", JackPeaklimMethods);
}

#endif
