#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>

#include "HoG.cpp"

extern "C" {

static PyObject *hogpy_hog(PyObject *self, PyObject *args) {
  // You code goes here!
}

static PyMethodDef HogpyMethods[] = {
    {"hog", hogpy_hog, METH_VARARGS,
     "Compute the HOG feature vector for an image."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef hogymodule = {
    PyModuleDef_HEAD_INIT, "hogpy", /* name of module */
    NULL,                           /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module,
           or -1 if the module keeps state in global variables. */
    HogpyMethods};

PyMODINIT_FUNC PyInit_hogpy() {
  PyObject *m = PyModule_Create(&hogymodule);
  if (m == nullptr)
    return nullptr;

  import_array();

  return m;
}
}