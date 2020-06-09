cmake_minimum_required(VERSION 3.12)
project(uussmlmodels VERSION 0.1.1 LANGUAGES CXX)
enable_testing()

##########################################################################################
#                    Define the compiler standard and requisites                         #
##########################################################################################
include(CheckCXXCompilerFlag)
include(FindOpenMP)
if (OPENMP_FOUND)
   message("OpenMP found")
   string(APPEND CMAKE_CXX_FLAGS " ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

configure_file(${CMAKE_SOURCE_DIR}/include/private/version.hpp.in
               ${CMAKE_SOURCE_DIR}/include/uuss/version.hpp)
set(PUBLIC_HEADER_DIRECTORIES
    include/uuss)
set(PRIVATE_HEADER_DIRECTORIES
    includeprivate)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RTSEIS_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

if (WRAP_PYTHON)
   find_package(PythonInterp 3)
   find_package(PythonLibs 3 REQUIRED)
   message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
   message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
   message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
   #string(APPEND CMAKE_CXX_FLAGS " -DPYBIND11_CPP17 ")
   find_package(pybind11 REQUIRED)
endif()

##########################################################################################
#                                 Download Models to Test                                #
##########################################################################################
if (NOT ${CMAKE_SOURCE_DIR}/testing/models/test_zrunet_p.h5)
   message("Downloading files...")
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/model_P_006.h5
        ${CMAKE_SOURCE_DIR}/testing/models/test_zrunet_p.h5
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/PB.B206.EH1..txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EH1.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/PB.B206.EH2..txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EH2.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/PB.B206.EHZ..txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EHZ.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/PB.UNET.P.PYTHON.txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.UNET.P.PYTHON.zrunet.txt
        INACTIVITY_TIMEOUT 5)
endif()


# Set the source
set(LIB_SRC
    src/version.cpp
    src/threeComponentPicker/zrunet/processData.cpp
    src/threeComponentPicker/zrunet/model.cpp)

set(BUILD_SHARED_LIBS YES)
add_library(uussmlmodels ${LIB_SRC})
target_link_libraries(uussmlmodels PRIVATE
                      ${RTSEIS_LIBRARY} ${HDF5_LIBRARIES} ${TORCH_LIBRARIES})
set_target_properties(uussmlmodels PROPERTIES
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)
target_include_directories(uussmlmodels PUBLIC ${PUBLIC_HEADER_FILES})
target_include_directories(uussmlmodels PRIVATE ${PRIVATE_HEADER_FILES})
if (OpenMP_CXX_FOUND)
    target_link_libraries(uussmlmodels PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python
if (WRAP_PYTHON)
   add_library(pyuussmlmodels MODULE
               python/waveformTemplate.cpp)
   set_property(TARGET pyuussmlmodels PROPERTY CXX_STD_14)
   set_property(TARGET pyuussmlmodels PROPERTY PREFIX "")
   target_include_directories(pyuussmlmodels PRIVATE pybind11 ${CMAKE_SOURCE_DIR}/python/include)
endif()

##########################################################################################
#                                          Unit Tests                                    #
##########################################################################################
add_executable(unitTests
               testing/main.cpp
               testing/zrunet.cpp)
target_link_libraries(unitTests
                      uussmlmodels ${RTSEIS_LIBRARY} ${GTEST_BOTH_LIBRARIES}
                                   ${HDF5_LIBRARIES} ${TORCH_LIBRARIES})
target_include_directories(unitTests PRIVATE ${GTEST_INCLUDE_DIR})
set_target_properties(unitTests PROPERTIES
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)
##########################################################################################
#                                         Installation                                   #
##########################################################################################
include(GNUInstallDirs)
if (WRAP_PYTHON)
   install(TARGETS uussmlmodels pyuussmlmodels
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
           LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
else()
   install(TARGETS uussmlmodels
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
           LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

##########################################################################################
#                                     CPACK Packaging                                    #
##########################################################################################
set(CPACK_PACKAGE_NAME "mlmodels")
set(CPACK_PACKAGE_VENDOR "The Internet")
set(CPACK_PACKAGE_CONTACT "bbaker@seis.utah.edu")
set(CPACK_PACKAGE_LICENSE "MIT")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Production level UUSS Machine learning models")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
