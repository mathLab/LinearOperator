CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.3 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

MESSAGE("-- Found ${_build_type} version of deal.II.")

IF(NOT DEAL_II_BUILD_TYPE MATCHES "Release")
  MESSAGE(FATAL_ERROR
    "deal.II has to be build in Release, or DebugRelease mode"
    )
ENDIF()

SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(linear_operator)

INCLUDE_DIRECTORIES(include eigen blaze /opt/local/include/)

FILE(GLOB _mains apps/*cc)
FOREACH(_main ${_mains})
  MESSAGE("-- Processing ${_main}")
  GET_FILENAME_COMPONENT(_name ${_main} NAME_WE)
  MESSAGE("-- Configuring executable ${_name}")
  ADD_EXECUTABLE(${_name} ${_main})
  DEAL_II_SETUP_TARGET(${_name} ${_BUILD_TYPE})
ENDFOREACH()
