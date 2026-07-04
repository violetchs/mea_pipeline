cmake_minimum_required(VERSION 3.10)
project(closed_loop_runner)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
    message(FATAL_ERROR
        "closed_loop_runner must be built with GCC/G++ 12 or newer because the "
        "bundled MaxLab static library requires newer libstdc++ symbols. "
        "Install g++-12 or g++-13, remove closed_loop/cpp/build, then configure "
        "with: CXX=/usr/bin/g++-12 CC=/usr/bin/gcc-12 cmake -S closed_loop/cpp -B closed_loop/cpp/build")
endif()
add_executable(closed_loop_runner src/closed_loop_runner.cpp)
if(NOT DEFINED ENV{MAXLAB_ROOT})
    set(MAXLAB_ROOT "$ENV{HOME}/MaxLab/share/libmaxlab/maxlab_lib")
else()
    set(MAXLAB_ROOT "$ENV{MAXLAB_ROOT}")
endif()
target_include_directories(closed_loop_runner PRIVATE "${MAXLAB_ROOT}/maxlab/include")
find_library(MAXLAB_LIBRARY maxlab PATHS "${MAXLAB_ROOT}/maxlab/lib" NO_DEFAULT_PATH)
if(MAXLAB_LIBRARY)
    target_link_libraries(closed_loop_runner PRIVATE "${MAXLAB_LIBRARY}" pthread)
else()
    target_link_libraries(closed_loop_runner PRIVATE "${MAXLAB_ROOT}/maxlab/lib/libmaxlab.a" pthread)
endif()
