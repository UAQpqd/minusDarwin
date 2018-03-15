cmake_minimum_required(VERSION 3.9)
project(MinusDarwin)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)
find_package(Boost COMPONENTS chrono random REQUIRED)
add_library(MinusDarwin Solver.cpp Solver.hpp RunTracer.cpp RunTracer.hpp Utility.hpp Utility.cpp)
target_include_directories(MinusDarwin PUBLIC ${OpenCL_INCLUDE_DIR})
target_link_libraries(MinusDarwin OpenCL::OpenCL ${Boost_LIBRARIES})

find_package(GTest REQUIRED)
enable_testing()
add_executable(MinusDarwinTest MinusDarwinTest.cpp SolverTest.cpp SolverTest.hpp)
target_include_directories(MinusDarwinTest PUBLIC ${GTEST_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})
message(STATUS "${Boost_LIBS}")
target_link_libraries(MinusDarwinTest OpenCL::OpenCL ${GTEST_LIBRARY} ${Boost_LIBRARIES} MinusDarwin)
add_test(NAME library_tests COMMAND MinusDarwinTest)


add_executable(MinusDarwinSinewaveFittingTuningTest MinusDarwinTest.cpp SinewaveFittingTuningTest.cpp SinewaveFittingTuningTest.hpp)
target_include_directories(MinusDarwinSinewaveFittingTuningTest PUBLIC ${GTEST_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})
message(STATUS "${Boost_LIBS}")
target_link_libraries(MinusDarwinSinewaveFittingTuningTest OpenCL::OpenCL ${GTEST_LIBRARY} ${Boost_LIBRARIES} MinusDarwin)
add_test(NAME sinewavefitting_tests COMMAND MinusDarwinSinewaveFittingTuningTest)