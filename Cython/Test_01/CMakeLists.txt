cmake_minimum_required(VERSION 3.0.0)
project(cython_test VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(cython_test main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
