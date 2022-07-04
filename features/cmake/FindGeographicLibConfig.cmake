# Already in cache, be silent
if (GEOGRAPHICLIB_INCLUDE_DIR AND GEOGRAPHICLIB_LIBRARY)
   set (GeographiCLib_FIND_QUIETLY TRUE)
endif()

#if (NOT BUILD_SHARED_LIBS)
#   set(CORE "libGeographic.a")
#else()
#   set(CORE "Geographiclib")
#endif()

find_path(GEOGRAPHICLIB_INCLUDE_DIR
          NAMES GeographicLib
          HINTS $ENV{GEOGRAPHICLIB_ROOT}/include
                /usr/local/include
                /usr/include)
find_library(GEOGRAPHICLIB_LIBRARY
             NAMES Geographic GeographicLib
             PATHS $ENV{GEOGRAPHICLIB_ROOT}/lib
                   $ENV{GEOGRAPHICLIB_ROOT}/lib64
                   /usr/local/lib
                   /usr/local/lib64)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FindGeographicLib DEFAULT_MSG GEOGRAPHICLIB_LIBRARY GEOGRAPHICLIB_INCLUDE_DIR)
mark_as_advanced(GEOGRAPHICLIB_INCLUDE_DIR GEOGRAPHICLIB_LIBRARY)
