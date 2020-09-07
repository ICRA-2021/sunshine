cmake_minimum_required(VERSION 3.8)
project(sunshine)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)

set(USE_ASAN off)
if (USE_ASAN)
    message(WARNING "Using ASAN; performance will be degraded!")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=leak")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fsanitize=leak")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=leak")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=leak")
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  sunshine_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf
  )
find_package(tf2_geometry_msgs REQUIRED)

find_package(Boost COMPONENTS program_options filesystem system REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
set(OPENCV_ENABLE_NONFREE "Enable non-free algorithms" ON)
set(WITH_OPENCL "Enable OpenCL" OFF)
find_package(OpenCV 4.2 QUIET)
if (NOT OpenCV_FOUND)
    message(WARNING "Failed to find OpenCV 4.2 -- falling back to OpenCV 3.2 (deprecated)")
    find_package(OpenCV 3.2 REQUIRED)
endif()

find_package(roslaunch REQUIRED)

find_library(libvisualwords libvisualwords.so)
find_library(libclear libclear.so)
find_library(liblimbo liblimbo.a PATHS /home/stewart/devel/3rdparty/limbo/build/src)
find_library(libnlopt libnlopt.so NAMES libnlopt_cxx.so)
#message(WARNING ${liblimbo})

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sunshine
   CATKIN_DEPENDS cv_bridge roscpp sensor_msgs std_msgs sunshine_msgs tf2 tf2_ros tf2_geometry_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
#  common
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/sunshine.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/sunshine_node.cpp)

add_library(sunshine_vision_adapters SHARED vision/visual_word_adapter.cpp vision/depth_adapter.cpp vision/segmentation_label_adapter.cpp)

add_executable(visual_word_extractor nodes/visual_word_extractor.cpp)
add_executable(label_word_extractor nodes/label_extractor.cpp include/sunshine/common/topic_map_utils.hpp)
add_executable(topic_model nodes/topic_model_node.cpp topic_model/rost_adapter.cpp)
add_executable(visualize nodes/visualize/visualize2d.cpp)
add_executable(visualize3d nodes/visualize/visualize3d.cpp)
add_executable(save_topic_map nodes/visualize/save_topic_map.cpp)
add_executable(save_point_cloud_2d nodes/visualize/save_point_cloud_2d.cpp)
add_executable(perplexity_viz nodes/visualize/perplexity_viz.cpp)
add_executable(test.image_publisher nodes/image_publisher.cpp)
add_executable(test.image_walker nodes/image_walk.cpp)
add_executable(utils.sunshine_eval utilities/sunshine_eval.cpp)
add_executable(test.adapters test/test_adapters.cpp)
add_executable(test.topic_model_node test/dummy_topic_model.cpp)

add_dependencies(visual_word_extractor sunshine_msgs_generate_messages_cpp)
add_dependencies(label_word_extractor sunshine_msgs_generate_messages_cpp)
add_dependencies(topic_model sunshine_msgs_generate_messages_cpp)
add_dependencies(visualize sunshine_msgs_generate_messages_cpp)
add_dependencies(visualize3d sunshine_msgs_generate_messages_cpp)
add_dependencies(save_topic_map sunshine_msgs_generate_messages_cpp)
add_dependencies(save_point_cloud_2d sunshine_msgs_generate_messages_cpp)
add_dependencies(perplexity_viz sunshine_msgs_generate_messages_cpp)
add_dependencies(test.image_publisher sunshine_msgs_generate_messages_cpp)
add_dependencies(test.image_walker sunshine_msgs_generate_messages_cpp)
add_dependencies(utils.sunshine_eval sunshine_msgs_generate_messages_cpp)
add_dependencies(test.adapters sunshine_msgs_generate_messages_cpp)
add_dependencies(test.topic_model_node sunshine_msgs_generate_messages_cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )


target_include_directories(sunshine_vision_adapters PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(sunshine_vision_adapters ${libvisualwords} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} lz4)

target_include_directories(visual_word_extractor PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(visual_word_extractor sunshine_vision_adapters ${libvisualwords} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} lz4)

target_include_directories(label_word_extractor PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(label_word_extractor sunshine_vision_adapters ${libvisualwords} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} lz4)

target_include_directories(topic_model PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(topic_model ${catkin_LIBRARIES} sunshine_vision_adapters)

target_include_directories(perplexity_viz PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(perplexity_viz ${catkin_LIBRARIES})

target_include_directories(visualize PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(visualize ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_include_directories(visualize3d PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(visualize3d ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_include_directories(save_topic_map PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(save_topic_map ${OpenCV_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_include_directories(save_point_cloud_2d PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(save_point_cloud_2d ${OpenCV_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_include_directories(test.image_publisher PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test.image_publisher ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_include_directories(test.image_walker PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test.image_walker ${OpenCV_LIBS} ${catkin_LIBRARIES})

target_include_directories(utils.sunshine_eval PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(utils.sunshine_eval ${OpenCV_LIBS} ${catkin_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} sunshine_vision_adapters)

target_include_directories(test.adapters PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test.adapters ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} sunshine_vision_adapters)
target_compile_features(test.adapters PRIVATE cxx_std_17)

target_include_directories(test.topic_model_node PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test.topic_model_node ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES})

if (libclear)
    find_package(Eigen3 REQUIRED)
    message(STATUS "Found libclear -- building model_translator and related utilities")
    add_executable(model_translator nodes/model_translator.cpp)
    add_executable(utils.topic_match utilities/topic_match.cpp)
    add_executable(utils.find_lifting utilities/find_lifting.cpp)
    add_executable(utils.multi_agent_match nodes/multi_agent_sim.cpp)

    add_dependencies(model_translator sunshine_msgs_generate_messages_cpp)
    target_include_directories(model_translator PUBLIC ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(model_translator ${catkin_LIBRARIES} ${libclear})

    target_include_directories(utils.topic_match PUBLIC ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(utils.topic_match ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${libclear})
    target_compile_features(utils.topic_match PRIVATE cxx_std_17)

    target_include_directories(utils.find_lifting PUBLIC ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(utils.find_lifting ${Boost_LIBRARIES} ${libclear})
    target_compile_features(utils.find_lifting PRIVATE cxx_std_17)

    add_dependencies(utils.multi_agent_match sunshine_msgs_generate_messages_cpp)
    target_include_directories(utils.multi_agent_match PUBLIC ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(utils.multi_agent_match ${Boost_LIBRARIES} ${libclear} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} sunshine_vision_adapters)
    target_compile_features(utils.multi_agent_match PRIVATE cxx_std_17)
else()
    message(STATUS "Did not find libclear -- skipping topic matching executables")
endif()

if (liblimbo)
    message(STATUS "Found liblimbo -- building optimizers")
    if (NOT libnlopt)
        message(WARNING "Cannot build optimizers without libnlopt")
    else()
        find_package(Eigen3 REQUIRED)
        add_executable(utils.rost_optimize utilities/rost_optimize.cpp)
        add_executable(test.test_optimize test/test_optimize.cpp)

        target_include_directories(utils.rost_optimize PUBLIC "/home/stewart/devel/3rdparty/limbo/src/" ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
        target_link_libraries(utils.rost_optimize ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} sunshine_vision_adapters ${liblimbo} ${libnlopt} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})

        target_include_directories(test.test_optimize PUBLIC "/home/stewart/devel/3rdparty/limbo/src/" ${EIGEN3_INCLUDE_DIR})
        target_link_libraries(test.test_optimize ${liblimbo} ${libnlopt} ${Boost_LIBRARIES})
    endif()
else()
    message(STATUS "Did not find liblimbo -- skipping optimizers")
endif()

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_sunshine.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

roslaunch_add_file_check(launch)
