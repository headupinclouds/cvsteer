# Copyright (c) 2015, David Hirvonen
# All rights reserved.

cmake_minimum_required(VERSION 3.0)

### Hunter snapshot that will be used ###
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.12.39.tar.gz"
    SHA1 "587bc8f6278f2b09ec8759d2b6a2f8d828c30f49"
    LOCAL
)
project(cvsteer VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(is_android ${ANDROID}) # syntax compatibility
string(COMPARE EQUAL "${CMAKE_OSX_SYSROOT}" "iphoneos" is_ios)
string(COMPARE EQUAL "${CMAKE_SYSTEM_NAME}" "Linux" is_linux)
string(COMPARE EQUAL "${CMAKE_GENERATOR}" "Xcode" is_xcode)

if(is_xcode)
  option(GATHERER_DISABLE_DSYM "Disable generation of dSYM information" ON)
  if(GATHERER_DISABLE_DSYM)
    set(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf")
  endif()
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release;MinSizeRel;RelWithDebInfo" CACHE STRING "Configuration types")
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)
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
add_subdirectory (test)
