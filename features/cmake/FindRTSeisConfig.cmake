# Already in cache, be silent
if (RTSEIS_INCLUDE_DIR AND RTSEIS_LIBRARY)
   set (RTSEIS_FIND_QUIETLY TRUE)
endif()

find_path(RTSEIS_INCLUDE_DIR
          NAMES rtseis
          HINTS $ENV{RTSEIS_ROOT}/include
                /usr/local/include
                /usr/include)
find_library(RTSEIS_LIBRARY
             NAMES rtseis
             PATHS $ENV{RTSEIS_ROOT}/lib
                   $ENV{RTSEIS_ROOT}/lib64
                   /usr/local/lib
                   /usr/local/lib64)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FindRTSeis DEFAULT_MSG RTSEIS_INCLUDE_DIR RTSEIS_LIBRARY)
mark_as_advanced(RTSEIS_INCLUDE_DIR RTSEIS_LIBRARY)
