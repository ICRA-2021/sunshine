# WARNING: Do not run cmake on this file/directory. Use `catkin build` in the parent workspace
# This file only exists so that tools can identify this folder as a CMake project.
cmake_minimum_required(VERSION 2.8.3)
project(sunshine_project)
include_directories("/opt/ros/kinetic/include")
set(CMAKE_VERBOSE_MAKEFILE ON)
add_subdirectory(sunshine_msgs)
add_subdirectory(sunshine)
