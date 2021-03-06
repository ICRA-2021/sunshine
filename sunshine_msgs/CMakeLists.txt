cmake_minimum_required(VERSION 2.8.3)
project(sunshine_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  LocalSurprise.msg
  Perplexity.msg
  Summary.msg
  TopicMatches.msg
  TopicModel.msg
  TopicWeights.msg
  TopicMap.msg
  WordObservation.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  GetImage.srv
  GetModelPerplexity.srv
  GetTopicModel.srv
  GetTopicMap.srv
  GetTopicsForTime.srv
  GetTopicSummary.srv
  LoadFile.srv
  MatchModels.srv
  MergeModels.srv
  Pause.srv
  RefineTopics.srv
  SaveFile.srv
  SaveObservationModel.srv
  SetTopicModel.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

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
    #INCLUDE_DIRS include
    #LIBRARIES
    CATKIN_DEPENDS
        geometry_msgs
        message_runtime
        std_msgs
    #DEPENDS
)

