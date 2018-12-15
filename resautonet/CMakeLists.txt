cmake_minimum_required(VERSION 2.8.12)
project(cmake_example)

include_directories(src)

add_subdirectory(pybind11)
pybind11_add_module(_math src/cmetrics.cpp src/bindings.cpp)
