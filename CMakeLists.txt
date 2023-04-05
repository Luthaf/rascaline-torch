cmake_minimum_required(VERSION 3.10)

project(rascaline_torch
    VERSION 0.1
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

if (BUILD_FOR_PYTHON)
    # add path to the cmake configuration of the version of libtorch used
    # by the Python torch module. PYTHON_EXECUTABLE is provided by skbuild
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
        OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
        ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
    )

    if (NOT ${TORCH_CMAKE_PATH_RESULT} EQUAL 0)
        message(FATAL_ERROR "failed to find your pytorch installation\n${TORCH_CMAKE_PATH_ERROR}")
    endif()

    string(STRIP ${TORCH_CMAKE_PATH_OUTPUT} TORCH_CMAKE_PATH_OUTPUT)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PATH_OUTPUT}")

    # add path to the cmake configuration of the version of librascaline used
    # by the Python rascaline module.
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import rascaline; print(rascaline.utils.cmake_prefix_path)"
        RESULT_VARIABLE RASCALINE_CMAKE_PATH_RESULT
        OUTPUT_VARIABLE RASCALINE_CMAKE_PATH_OUTPUT
        ERROR_VARIABLE RASCALINE_CMAKE_PATH_ERROR
    )

    if (NOT ${RASCALINE_CMAKE_PATH_RESULT} EQUAL 0)
        message(FATAL_ERROR "failed to find your rascaline installation\n${RASCALINE_CMAKE_PATH_ERROR}")
    endif()

    string(STRIP "${RASCALINE_CMAKE_PATH_OUTPUT}" RASCALINE_CMAKE_PATH_OUTPUT)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${RASCALINE_CMAKE_PATH_OUTPUT}")
endif(BUILD_FOR_PYTHON)

find_package(Torch 1.11 REQUIRED)

if (NOT TARGET rascaline)
    find_package(rascaline 0.1 REQUIRED)
endif()

add_library(rascaline_torch SHARED
    src/rascaline_torch/cxx/equistore_torch.cpp
    src/rascaline_torch/cxx/system.cpp
    src/rascaline_torch/cxx/autograd.cpp
    src/rascaline_torch/cxx/register.cpp
)

target_link_libraries(rascaline_torch torch rascaline)
target_include_directories(rascaline_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rascaline_torch/src)

install(TARGETS rascaline_torch
    LIBRARY DESTINATION "lib"
)
