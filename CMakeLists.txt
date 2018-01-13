cmake_minimum_required(VERSION 3.9)
project(MinusDarwin)

set(CMAKE_CXX_STANDARD 11)

add_library(MinusDarwin Solver.cpp Solver.hpp RunTracer.cpp RunTracer.hpp)