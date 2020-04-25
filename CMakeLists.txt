cmake_minimum_required (VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include)

project(NetComm)

add_executable(main main.cpp)
