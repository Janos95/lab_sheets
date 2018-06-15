#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>
#include <cassert>

#include "HoG.cpp"

extern "C" {

static PyObject* hogpy_hog(PyObject* self, PyObject* args) 
{
    PyObject *arg=NULL, *arr=NULL;

    int nb_bins, cwidth, block_size;
    char unsigned_dirs; 
    double clip_val;

    if (!PyArg_ParseTuple(
	    	args
	    	,"Oiiibd"
	    	,&arg
	    	,&nb_bins
	    	,&cwidth
	    	,&block_size
	    	,&unsigned_dirs
	    	,&clip_val))
    {
    	return NULL;	
    } 

    arr = PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_ALIGNED); 
    if (!arr) 
    {
    	Py_XDECREF(arr);
    	return NULL;
    }
    PyArrayObject *np_arr = reinterpret_cast<PyArrayObject*>(arr);

    std::vector<size_t> img_size(PyArray_DIMS(np_arr), PyArray_DIMS(np_arr)+PyArray_NDIM(np_arr));
    if(img_size.size() != 2)
    {
    	char message[100*sizeof(char)];
    	snprintf(message, sizeof(message)
    		, "Your array has %d dimensions (needs to be 2)!", (int)img_size.size());
    	PyErr_SetString(PyExc_RuntimeError, message);
    	Py_XDECREF(arr);
    	return NULL;
    }
  
    const double* arr_data = (const double*)PyArray_DATA(np_arr);

    const size_t num_features = getNumFeatures(img_size.data(), nb_bins, cwidth, block_size);
    npy_intp out_dims(num_features);
    PyObject* oarr = PyArray_SimpleNew(1, &out_dims, NPY_DOUBLE);
    if(!oarr)
    {
    	char message[50*sizeof(char)];
    	snprintf(message, sizeof(message), "Could not allocate %d bytes.", (int)out_dims*sizeof(double));
    	PyErr_SetString(PyExc_RuntimeError, message);
    	Py_XDECREF(arr);
    	return NULL;
    }
    PyArrayObject *np_oarr = reinterpret_cast<PyArrayObject*>(oarr);
    double* oarr_data = (double*)PyArray_DATA(np_oarr);
   
	HoG(arr_data, nb_bins, cwidth, block_size, unsigned_dirs, clip_val,
        img_size.data(), PyArray_STRIDE(np_arr,1)/sizeof(double),
        oarr_data, true);

    Py_DECREF(arr);
    Py_INCREF(oarr);
    return oarr;
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
  if (!m)
    return nullptr;

  import_array();

  return m;
}
}