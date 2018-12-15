
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2);
py::array_t<double> subtract_arrays(py::array_t<double> input1, py::array_t<double> input2);
double sum_array(py::array_t<double> input);
double rsquared(py::array_t<double> obs, py::array_t<double> pre);
double rmse(py::array_t<double> obs, py::array_t<double> pre);