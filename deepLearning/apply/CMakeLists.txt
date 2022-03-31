cmake_minimum_required(VERSION 3.12)
project(uussmlmodels VERSION 0.1.1 LANGUAGES CXX)
enable_testing()

##########################################################################################
#                    Define the compiler standard and requisites                         #
##########################################################################################
include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
   #string(APPEND CMAKE_CXX_FLAGS " -DPYBIND11_CPP20 ")
   find_package(pybind11 REQUIRED)
endif()

##########################################################################################
#                                 Download Models to Test                                #
##########################################################################################
if (NOT ${CMAKE_SOURCE_DIR}/testing/models/test_zrunet_p.h5)
   message("Downloading files...")
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/model_P_006.h5
        ${CMAKE_SOURCE_DIR}/testing/models/test_zrunet_p.h5
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/PB.B206.EH1..txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EH1.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/PB.B206.EH2..txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EH2.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/PB.B206.EHZ..txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EHZ.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/PB.UNET.P.PYTHON.txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.UNET.P.PYTHON.zrunet.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/PB.B206.EH1.PROC.txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EH1.PROC.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/PB.B206.EH2.PROC.txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EH2.PROC.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/PB.B206.EHZ.PROC.txt
        ${CMAKE_SOURCE_DIR}/testing/data/PB.B206.EHZ.PROC.zrunet_p.txt
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/fmnet_test_005.h5
        ${CMAKE_SOURCE_DIR}/testing/models/test_fmnet.h5
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/p_signals_zcnn_p.csv
        ${CMAKE_SOURCE_DIR}/testing/data/p_signals_zcnn_p.csv
        INACTIVITY_TIMEOUT 5)
   file(DOWNLOAD
        http://home.chpc.utah.edu/~u6024448/uussmlmodels/test_zrcnnpick_p.h5
        ${CMAKE_SOURCE_DIR}/testing/models/test_zrcnnpick_p.h5
        INACTIVITY_TIMEOUT 5)
endif()


# Set the source
set(LIB_SRC
    src/version.cpp
    src/amplitudes/localMagnitudeProcessing.cpp
    src/firstMotion/fmnet/processData.cpp
    src/firstMotion/fmnet/model.cpp
    src/oneComponentPicker/zcnn/model.cpp
    src/oneComponentPicker/zcnn/processData.cpp
    src/threeComponentPicker/zrunet/processData.cpp
    src/threeComponentPicker/zrunet/model.cpp
    src/threeComponentPicker/zcnn/processData.cpp
    src/threeComponentPicker/zcnn/model.cpp)

set(BUILD_SHARED_LIBS YES)
add_library(uussmlmodels ${LIB_SRC})
target_link_libraries(uussmlmodels PRIVATE
                      ${RTSEIS_LIBRARY} ${HDF5_LIBRARIES} ${TORCH_LIBRARIES})
set_target_properties(uussmlmodels PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)
target_include_directories(uussmlmodels PUBLIC ${PUBLIC_HEADER_FILES})
target_include_directories(uussmlmodels PRIVATE ${PRIVATE_HEADER_FILES})
if (OpenMP_CXX_FOUND)
    target_link_libraries(uussmlmodels PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python
if (WRAP_PYTHON)
   find_package(pybind11 REQUIRED)
   add_library(pyuussmlmodels MODULE
               python/pyuussmlmodels.cpp
               python/oneComponentPicker/processData.cpp
               python/firstMotion/processData.cpp
               python/threeComponentPicker/processData.cpp
               python/amplitudes/localMagnitudeProcessing.cpp)
   target_link_libraries(pyuussmlmodels PRIVATE
                         pybind11::module uussmlmodels)
   set_target_properties(pyuussmlmodels PROPERTIES
                         PREFIX ""
                         CXX_STANDARD 20
                         CXX_STANDARD_REQUIRED YES 
                         CXX_EXTENSIONS NO) 
   target_include_directories(pyuussmlmodels PRIVATE pybind11
                              ${CMAKE_SOURCE_DIR}/python)
endif()

##########################################################################################
#                                          Unit Tests                                    #
##########################################################################################
add_executable(unitTests
               testing/main.cpp
               testing/zrcnn.cpp
               testing/fmnet.cpp
               testing/zrunet.cpp)
target_link_libraries(unitTests PRIVATE uussmlmodels ${TORCH_LIBRARIES} ${GTEST_BOTH_LIBRARIES})
#${RTSEIS_LIBRARY} ${HDF5_LIBRARIES} ${TORCH_LIBRARIES}
#                                  ${GTEST_BOTH_LIBRARIES})
target_include_directories(unitTests PRIVATE ${GTEST_INCLUDE_DIRS})
set_target_properties(unitTests PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)
add_test(NAME unitTests
         COMMAND unitTests)
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
set(CPACK_PACKAGE_NAME "uussmlmodels")
set(CPACK_PACKAGE_VENDOR "UUSS")
set(CPACK_PACKAGE_CONTACT "bbaker@seis.utah.edu")
set(CPACK_PACKAGE_LICENSE "MIT")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Production level UUSS Machine learning models")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
