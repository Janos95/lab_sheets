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
    std::vector<size_t> strides(PyArray_STRIDES(np_arr), PyArray_STRIDES(np_arr)+PyArray_NDIM(np_arr));
    if(img_size.size() != 2 || img_size.size() != 3)
    {
    	Py_XDECREF(arr);
    	return NULL;
    }
    if(img_size.size() == 2)
    {
    	img_size.insert(img_size.begin(), 1);
    	strides.insert(strides.begin(), 0); //insert dummy stride
    }
    
    const double* arr_data = (const double*)PyArray_DATA(np_arr);

    printf("The strides are %d %d %d\n",strides[0]/8, strides[1]/8, strides[2]/8);


    const size_t num_features = getNumFeatures(img_size.data(), nb_bins, cwidth, block_size);
    npy_intp out_dims(num_features*img_size[0]);
    PyObject* oarr = PyArray_SimpleNew(1, &out_dims, NPY_DOUBLE);
    PyArrayObject *np_oarr = reinterpret_cast<PyArrayObject*>(oarr);
    double* oarr_data = (double*)PyArray_DATA(np_oarr);
    
    #pragma omp parallel for
    for (size_t i = 0; i < img_size[0]; ++i)
	{
    	HoG(arr_data+i*strides[2]/sizeof(double), nb_bins, cwidth, block_size, unsigned_dirs, clip_val,
            img_size.data(), strides[1]/sizeof(double),
            oarr_data+i*num_features, true);
	} 

	//reshape the array if necessary
	if(img_size[0] > 1)
	{
		PyObject* oarr_reshaped = PyArray_Newshape(
			oarr, PyArray_Dims* newshape, NPY_FORTRANORDER);
		Py_DECREF(arr);
    	Py_INCREF(oarr_reshaped);
    	return oarr_reshaped;
	}


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