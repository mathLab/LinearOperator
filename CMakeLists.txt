##
# CMake script for large project, with one executable
# and a library of objects depending on deal2lkit.
#
# The structure of the directory is assumed to be: 
# ./source: containing implementations and no files with "int main()"
# ./include: containing all class declarations
# ./tests: containing pairs of files: test.cc/test.output to test
# ./main: containing files with "int main()", which are compiled to
# one executable each
#
# If you set the environemnt variable DEAL2LKIT_DIR or D2K_DIR, 
# everything will work out of the box
##

SET(TARGET linear_operator)

# Declare all source files the target consists of:
file(GLOB _mains apps/*cc)

INCLUDE_DIRECTORIES(include eigen blaze /opt/local/include/)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.3 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()


PROJECT(${TARGET})

# We add one library and one target for each type of deal.II library
# we found. If you compiled deal.II with both Release and Debug
# mode, this will generate both Release and Debug programs for you.
# The debug library and program are postfixed with ".g"
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}")

    MESSAGE("-- Found ${_build_type} version of deal.II.")

    STRING(TOUPPER "${_build_type}" _BUILD_TYPE)

    FOREACH(_main ${_mains})
	MESSAGE("-- Processing ${_main}")
	GET_FILENAME_COMPONENT(_app_name ${_main} NAME_WE)
    	SET(_exe "${_app_name}${_p}")
    	MESSAGE("-- Configuring executable ${_exe}")
    	ADD_EXECUTABLE(${_exe} ${_main})
	DEAL_II_SETUP_TARGET(${_exe})
    ENDFOREACH()

  ENDIF()
ENDFOREACH()
