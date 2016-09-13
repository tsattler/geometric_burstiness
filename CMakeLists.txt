cmake_minimum_required (VERSION 2.6)

project (GeometricBurstiness)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# add our macro directory to cmake search path
if (EXISTS "${CMAKE_SOURCE_DIR}/cmake")
  set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
endif ()

if ( NOT ENABLE_TESTS )
    set (ENABLE_TESTS OFF CACHE BOOL "Turn on testing framework.")
endif()

if ( NOT STL_VECTOR_CHECKS )
    set ( STL_VECTOR_CHECKS false CACHE BOOL "Include full stl vector checks in debug mode (This option is only used in debug Mode!)" )
endif ( NOT STL_VECTOR_CHECKS )

if ( STL_VECTOR_CHECKS )
    add_definitions (
        -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC
    )
endif ()

set( CMAKE_DEBUG_POSTFIX "d" )


add_subdirectory (src)

