cmake_minimum_required (VERSION 2.8)

if (EXISTS ${CMAKE_SOURCE_DIR}/cmake)
  set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
endif ()

find_package (Eigen3 REQUIRED)
find_package(FLANN REQUIRED)

include_directories (
  ${CMAKE_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}
  ${FLANN_INCLUDE_DIR}
)

link_directories (
  ${FLANN_LIBRARY_DIR}
)

add_executable(compute_index_weights compute_index_weights.cc structures.h inverted_file.h inverted_index.h)
add_executable(query query.cc fast_spatial_matching.h geometric_transforms.h structures.h inverted_file.h inverted_index.h)
add_executable(hesaff_sift_to_binary_root_sift hesaff_sift_to_binary_root_sift.cc hesaff_sift.h)
add_executable(compute_word_assignments compute_word_assignments.cc hesaff_sift.h)
add_executable(build_partial_index build_partial_index.cc structures.h inverted_file.h inverted_index.h)
add_executable(compute_hamming_thresholds compute_hamming_thresholds.cc structures.h inverted_file.h inverted_index.h)

target_link_libraries(compute_word_assignments ${FLANN_LIBRARY})

install( PROGRAMS ${CMAKE_BINARY_DIR}/src/compute_index_weights
         DESTINATION ${CMAKE_INSTALL_PREFIX})
         
install( PROGRAMS ${CMAKE_BINARY_DIR}/src/query
         DESTINATION ${CMAKE_INSTALL_PREFIX})

install( PROGRAMS ${CMAKE_BINARY_DIR}/src/hesaff_sift_to_binary_root_sift
        DESTINATION ${CMAKE_INSTALL_PREFIX})

install( PROGRAMS ${CMAKE_BINARY_DIR}/src/compute_word_assignments
        DESTINATION ${CMAKE_INSTALL_PREFIX})

install( PROGRAMS ${CMAKE_BINARY_DIR}/src/build_partial_index
        DESTINATION ${CMAKE_INSTALL_PREFIX})
        
install( PROGRAMS ${CMAKE_BINARY_DIR}/src/compute_hamming_thresholds
        DESTINATION ${CMAKE_INSTALL_PREFIX})