# - Try to find FLANN
# Once done this will define
#
# FLANN_FOUND           - system has FLANN
# FLANN_INCLUDE_DIR - the FLANN include directory
# FLANN_LIBRARY         - Link these to use FLANN
# FLANN_LIBRARY_DIR  - Library DIR of FLANN
#

IF (FLANN_INCLUDE_DIR)
 # Already in cache, be silent
 SET(FLANN_FIND_QUIETLY TRUE)
ENDIF (FLANN_INCLUDE_DIR)


FIND_PATH(FLANN_INCLUDE_DIR flann/flann.hpp
	  PATHS "/usr/include")


set( LIBDIR lib )

if( FLANN_INCLUDE_DIR )
   set( FLANN_FOUND TRUE )

   set( FLANN_LIBRARY_DIR "/usr/lib" )

   set( FLANN_LIBRARY optimized flann_cpp debug flann_cpp )

ELSE (FLANN_INCLUDE_DIR)
   SET(FLANN_FOUND FALSE)
ENDIF (FLANN_INCLUDE_DIR)

