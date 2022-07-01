cmake_minimum_required(VERSION 3.12)
project(uussFeatures VERSION 0.1.1 LANGUAGES CXX)
enable_testing()

find_package(GTest REQUIRED)
set(FindRTSeis_DIR ${CMAKE_SOURCE_DIR}/cmake)
find_package(FindRTSeis REQUIRED)

if (${WRAP_PYTHON})
   find_package(PythonInterp 3)
   find_package(PythonLibs 3 REQUIRED)
   message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
   message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
   message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
   find_package(pybind11 REQUIRED)
endif()

set(PUBLIC_HEADER_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/include/)
set(PRIVATE_HEADER_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/include/private)

set(SRC
    src/features/magnitude/verticalChannelFeatures.cpp)

set(BUILD_SHARED_LIBS YES)
add_library(uussFeatures ${SRC})
set_target_properties(uussFeatures PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)
target_link_libraries(uussFeatures PRIVATE ${RTSEIS_LIBRARY})
target_include_directories(uussFeatures
                           PRIVATE ${RTSEIS_INCLUDE_DIR}
                           PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADER_DIRECTORIES}>
                           PUBLIC $<INSTALL_INTERFACE:${PUBLIC_HEADER_DIRECTORIES}>)

#========================================================================================#
#                                       Unit Tests                                       #
#========================================================================================#
file(COPY ${CMAKE_SOURCE_DIR}/testing/data DESTINATION .)
add_executable(magnitudes
               testing/magnitude/main.cpp
               testing/magnitude/magnitudes.cpp)
set_target_properties(magnitudes PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES 
                      CXX_EXTENSIONS NO)
target_link_libraries(magnitudes
                      PRIVATE uussFeatures ${GTEST_BOTH_LIBRARIES})
target_include_directories(magnitudes
                           PRIVATE ${PUBLIC_HEADER_DIRECTORIES}
                                   ${PRIVATE_HEADER_DIRECTORIES} ${GTEST_INCLUDE_DIRS})


#========================================================================================#
#                                      Installation                                      #
#========================================================================================#
include(GNUInstallDirs)
if (${WRAP_PYTHON})
   install(TARGETS uussFeatures
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
           LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
else()
   install(TARGETS uussFeatures
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
           LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

##########################################################################################
#                                     CPACK Packaging                                    #
##########################################################################################
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "UUSS")
set(CPACK_PACKAGE_CONTACT "ben.baker@utah.edu")
set(CPACK_PACKAGE_LICENSE "MIT")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Production-level feature extraction tools for feature-based UUSS machine learning models.")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_VERBATIM_VARIABLES YES)
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_COMPONENTS_ALL libraries headers)
if (WIN32)
    set(CPACK_GENERATOR ZIP WIX)
elseif (APPLE)
    set(CPACK_GENERATOR TGZ productbuild)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CPACK_GENERATOR TGZ RPM)
else()
    set(CPACK_GENERATOR TGZ)
endif()
set(CPACK_SOURCE_IGNORE_FILES
  /\\.git/
  \\.swp
  \\.orig
  /CMakeLists\\.txt\\.user
  /private/
)
include(CPack) # Put this last!
