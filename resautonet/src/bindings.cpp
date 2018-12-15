#include <pybind11/pybind11.h>
#include "cmetrics.hpp"

PYBIND11_MODULE(_metrics, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: _metrics

        .. autosummary::
           :toctree: _generate
           subtract_arrays
           add_arrays
           sum_array
           rsquared
	       rmse
    )pbdoc";

    m.def("add_arrays", &add_arrays, "Addition of the arrays");
    m.def("subtract_arrays", &subtract_arrays, "Subtract of two arrays");
    m.def("sum_array", &sum_array, "Sum of one arrays");
    m.def("rsquared", &rsquared, R"doc(
            Calculate rsquared for the observed and predicted values.
              :param y_true: array tensor for observation, just the last output .
              :param y_pred: array tensor for predictions, just the last output.
         )doc");
    m.def("rmse", &rmse, R"doc(
            Calculate RMSE for the observed and predicted values!
              :param y_true: array tensor for observation, just the last output .
              :param y_pred: array tensor for predictions, just the last output.
            )doc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

