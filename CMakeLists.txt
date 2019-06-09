# WARNING: Do not run cmake on this file/directory. Use `catkin build` in the parent workspace
# This file only exists so that tools can identify this folder as a CMake project.
cmake_minimum_required(VERSION 2.8.3)
project(sunshine_project)
IF (NOT DEFINED ENV{ROS_DISTRO})
    message(FATAL "Please ensure that the ROS_DISTRO environment variable is set.")
ENDIF()
include_directories("/opt/ros/$ENV{ROS_DISTRO}/include")
LIST(APPEND CMAKE_PREFIX_PATH "/opt/ros/$ENV{ROS_DISTRO}")
set(CMAKE_VERBOSE_MAKEFILE ON)
add_subdirectory(sunshine_msgs)
add_subdirectory(sunshine)
