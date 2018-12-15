#include "cmetrics.hpp"

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
	auto buf1 = input1.request(), buf2 = input2.request();

	if (buf1.size != buf2.size)
	throw std::runtime_error("Input shapes must match");

	/*  allocate the buffer */
	py::array_t<double> result = py::array_t<double>(buf1.size);

	auto buf3 = result.request();

	double *ptr1 = (double *) buf1.ptr,
		*ptr2 = (double *) buf2.ptr,
		*ptr3 = (double *) buf3.ptr;
	int X = buf1.shape[0];
	for (size_t idx = 0; idx < X; idx++)
		ptr3[idx] = ptr1[idx] + ptr2[idx];
	// reshape array to match input shape
	//result.resize({X,Y});
	return result;
}

py::array_t<double> subtract_arrays(py::array_t<double> input1, py::array_t<double> input2) {
	auto buf1 = input1.request(), buf2 = input2.request();

	if (buf1.size != buf2.size)
	throw std::runtime_error("Input shapes must match");

	/*  allocate the buffer */
	py::array_t<double> result = py::array_t<double>(buf1.size);

	auto buf3 = result.request();

	double *ptr1 = (double *) buf1.ptr,
		*ptr2 = (double *) buf2.ptr,
		*ptr3 = (double *) buf3.ptr;
	int X = buf1.shape[0];
	for (size_t idx = 0; idx < X; idx++)
		ptr3[idx] = ptr1[idx] - ptr2[idx];
	// reshape array to match input shape
	//result.resize({X,Y});
	return result;
}

double sum_array(py::array_t<double> input) {
	auto buf1 = input.request();
	/*  allocate the buffer */
	double result = 0;
	double *ptr1 = (double *) buf1.ptr;
	int X = buf1.shape[0];
	for (size_t idx = 0; idx < X; idx++)
		result = result+ptr1[idx];
	// reshape array to match input shape
	//result.resize({X,Y});
	return result;
}

double rsquared(py::array_t<double> obs, py::array_t<double> pre){
    auto buf1 = obs.request(),buf2 = pre.request();
	/*  allocate the buffer */
	double result = 0;
	double *ptr1 = (double *) buf1.ptr;
	double *ptr2 = (double *) buf2.ptr;
	int X = buf1.shape[0];
    double ss_res=0;
	for (size_t idx = 0; idx < X; idx++)
		ss_res = ss_res+(ptr1[idx]-ptr2[idx])*(ptr1[idx]-ptr2[idx]);
	double mean=sum_array(obs)/(double)X;
	double ss_tot=0;
	for (size_t idx = 0; idx < X; idx++)
		ss_tot = ss_tot+(ptr1[idx]-mean)*(ptr1[idx]-mean);
	// reshape array to match input shape
	//result.resize({X,Y});
	return (1-ss_res/(ss_tot+0.000000001));
}

double rmse(py::array_t<double> obs, py::array_t<double> pre){
	auto buf1 = obs.request(),buf2 = pre.request();
	/*  allocate the buffer */
	double result = 0;
	double *ptr1 = (double *) buf1.ptr;
	double *ptr2 = (double *) buf2.ptr;
	int X = buf1.shape[0];
    double ss_res=0;
	for (size_t idx = 0; idx < X; idx++)
		ss_res = ss_res+(ptr1[idx]-ptr2[idx])*(ptr1[idx]-ptr2[idx]);
	double mean=ss_res/(double)X;
	return sqrt(mean);
}