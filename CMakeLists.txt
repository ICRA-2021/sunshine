# WARNING: Do not run cmake on this file/directory. Use `catkin build` in the parent workspace
# This file only exists so that tools can identify this folder as a CMake project.
cmake_minimum_required(VERSION 3.0)
project(sunshine_project)

option(BUILD_EXT "Builds Sunshine extension packages that may have additional dependencies" OFF)

IF (NOT DEFINED ENV{ROS_DISTRO})
    set(ENV{ROS_DISTRO} noetic)
    message(WARNING "Assuming distro $ENV{ROS_DISTRO}. Use the ROS_DISTRO environment variable to override.")
ENDIF()
#include_directories("/opt/ros/$ENV{ROS_DISTRO}/include")
include_directories("$ENV{HOME}/warp_ws/devel/include")
LIST(APPEND CMAKE_PREFIX_PATH "/opt/ros/$ENV{ROS_DISTRO};$ENV{HOME}/warp_ws/devel")
set(CMAKE_VERBOSE_MAKEFILE ON)
add_subdirectory(sunshine_msgs)
add_subdirectory(sunshine)

IF (BUILD_EXT)
add_subdirectory(sunshine_ext)
ENDIF()
