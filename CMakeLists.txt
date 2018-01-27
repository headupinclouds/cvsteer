# Copyright (c) 2015-2018, David Hirvonen
# All rights reserved.

cmake_minimum_required(VERSION 3.1)

### Hunter snapshot that will be used ###
include("cmake/HunterGate.cmake")
HunterGate(
  URL "https://github.com/ruslo/hunter/archive/v0.19.238.tar.gz"
  SHA1 "9e103596184adb7206430047375a07eb1fa6a9b4"
)
  
project(cvsteer VERSION 0.1.1)

set(CVSTEER_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")

option(CVSTEER_BUILD_EXAMPLE "Build example" OFF)
option(CVSTEER_BUILD_TEST "Build test" OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(IOS OR ANDROID)
  set(CVSTEER_IS_MOBILE TRUE)
else()
  set(CVSTEER_IS_MOBILE FALSE)
endif()

## #################################################################
## Dependencies - OpenCV
## #################################################################

hunter_add_package(OpenCV)
find_package(OpenCV REQUIRED)

## #################################################################
## Project
## #################################################################

add_subdirectory(cvsteer)

if(CVSTEER_BUILD_EXAMPLE)
  add_subdirectory(example)
endif()

if(CVSTEER_BUILD_TEST AND NOT CVSTEER_IS_MOBILE)
  enable_testing()
  add_subdirectory(test)
endif()
