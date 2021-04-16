cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(ncnn-portrait-segmentation LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "[ncnn-portrait-segmentation] CMAKE_BUILD_TYPE is Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Select build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_parties)
add_subdirectory(example)