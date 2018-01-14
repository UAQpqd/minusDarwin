cmake_minimum_required(VERSION 3.9)
project(MinusDarwin)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS chrono)
add_library(MinusDarwin Solver.cpp Solver.hpp RunTracer.cpp RunTracer.hpp Utility.hpp Utility.cpp)
target_include_directories(MinusDarwin PUBLIC ${OpenCL_INCLUDE_DIR})
target_link_libraries(MinusDarwin OpenCL::OpenCL Boost::chrono)


find_package(GTest REQUIRED)
enable_testing()
add_executable(MinusDarwinTest MinusDarwinTest.cpp SolverTest.cpp SolverTest.hpp)
target_include_directories(MinusDarwinTest PUBLIC ${GTEST_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})
target_link_libraries(MinusDarwinTest MinusDarwin OpenCL::OpenCL ${GTEST_LIBRARY})
add_test(NAME all_tests COMMAND MinusDarwinTest)