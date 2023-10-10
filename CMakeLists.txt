cmake_minimum_required(VERSION 3.16)
project(UUSSMLModels VERSION 0.1.0 LANGUAGES C CXX)
enable_testing()

option(WRAP_PYTHON "Builds the pybind11 Python wrappers" OFF)

set(BUILD_SHARED_LIBS YES)
include(CheckCXXCompilerFlag)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
message("Module paths: " ${CMAKE_MODULE_PATH})
find_package(GTest REQUIRED)
find_package(RTSeis REQUIRED)
find_package(HDF5 COMPONENTS C REQUIRED)
find_package(Torch)
find_package(OpenVINO CONFIG COMPONENTS Runtime)
if (${TORCH_FOUND})
   message("Torch found")
   add_compile_definitions(WITH_TORCH)
   #message("extra flags: " ${TORCH_CXX_FLAGS})
endif()
if (${OpenVINO_Runtime_FOUND})
   message("OpenVINO runtime found")
   add_compile_definitions(WITH_OPENVINO)
endif()
if (NOT ${OpenVINO_Runtime_FOUND})
   if (NOT ${TORCH_FOUND})
      message("WARNING: No deep learning inference will be performed")
    endif()
endif()

configure_file(${CMAKE_SOURCE_DIR}/include/private/version.hpp.in
               ${CMAKE_SOURCE_DIR}/include/uussmlmodels/version.hpp)
set(PUBLIC_HEADERS
    ${CMAKE_SOURCE_DIR}/include/uussmlmodels/version.hpp
    ${CMAKE_SOURCE_DIR}/include/uussmlmodels/detectors/uNetThreeComponentP/inference.hpp
    ${CMAKE_SOURCE_DIR}/include/uussmlmodels/detectors/uNetThreeComponentP/preprocessing.hpp
    ${CMAKE_SOURCE_DIR}/include/uussmlmodels/detectors/uNetThreeComponentS/inference.hpp
    ${CMAKE_SOURCE_DIR}/include/uussmlmodels/detectors/uNetThreeComponentS/preprocessing.hpp
    ${CMAKE_SOURCE_DIR}/include/uussmlmodels/detectors/uNetOneComponentP/inference.hpp
    ${CMAKE_SOURCE_DIR}/include/uussmlmodels/detectors/uNetOneComponentP/preprocessing.hpp
)
set(INFERENCE_SRC
    detectors/uNetOneComponentP/inference/inference.cpp
    detectors/uNetOneComponentP/preprocessing/preprocessing.cpp
    detectors/uNetThreeComponentP/inference/inference.cpp
    detectors/uNetThreeComponentP/preprocessing/preprocessing.cpp
    detectors/uNetThreeComponentS/inference/inference.cpp
    detectors/uNetThreeComponentS/preprocessing/preprocessing.cpp
    eventClassifiers/cnnThreeComponent/preprocessing/preprocessing.cpp
    firstMotionClassifiers/cnnOneComponentP/inference/inference.cpp
    firstMotionClassifiers/cnnOneComponentP/preprocessing/preprocessing.cpp
    pickers/cnnOneComponentP/inference/inference.cpp
    pickers/cnnOneComponentP/preprocessing/preprocessing.cpp
    pickers/cnnThreeComponentS/inference/inference.cpp
    pickers/cnnThreeComponentS/preprocessing/preprocessing.cpp
    include/private/h5io.hpp
)
add_library(uussmlmodels ${INFERENCE_SRC})
set_target_properties(uussmlmodels PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES 
                      CXX_EXTENSIONS NO)
target_include_directories(uussmlmodels
                           PRIVATE ${RTSeis_INCLUDE_DIR} hdf5::hdf5
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                           PUBLIC  $<INSTALL_INTERFACE:include>)
target_link_libraries(uussmlmodels PRIVATE ${RTSeis_LIBRARY} hdf5::hdf5)
if (${OpenVINO_Runtime_FOUND})
   target_include_directories(uussmlmodels PRIVATE openvino::runtime)
   target_link_libraries(uussmlmodels PRIVATE openvino::runtime)
endif()
if (${TORCH_FOUND})
   target_include_directories(uussmlmodels PRIVATE ${TORCH_INDLUDE_DIRS})
   target_link_libraries(uussmlmodels INTERFACE ${TORCH_LIBRARIES})
endif()

##########################################################################################
#                                           Python                                       #
##########################################################################################
if (${WRAP_PYTHON})
   message("Will build Python bindings")
   find_package(pybind11 REQUIRED)
   add_library(pyuussmlmodels MODULE
               python/pyuussmlmodels.cpp
               python/detectors.cpp
               python/pickers.cpp
               python/firstMotionClassifiers.cpp
               python/eventClassifiers.cpp
               )
   set_target_properties(pyuussmlmodels PROPERTIES
                         PREFIX ""
                         CXX_STANDARD 20
                         CXX_STANDARD_REQUIRED YES
                         CXX_EXTENSIONS NO)
   target_include_directories(pyuussmlmodels
                              PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
   target_link_libraries(pyuussmlmodels
                         PRIVATE pybind11::pybind11 pybind11::lto uussmlmodels) 
   file(COPY ${CMAKE_SOURCE_DIR}/python/unit_test.py DESTINATION .)
   add_test(NAME python_tests
            COMMAND ${PYTHON_EXECUTABLE} -m pytest)
endif()


##########################################################################################
#                                           Testing                                      #
##########################################################################################
file(COPY ${CMAKE_SOURCE_DIR}/testing/data DESTINATION .)
set(TEST_SRC
    testing/main.cpp
    testing/pickers/cnnOneComponentP.cpp
    testing/pickers/cnnThreeComponentS.cpp
    testing/firstMotionClassifiers/cnnOneComponentP.cpp
    testing/detectors/uNetOneComponentP.cpp 
    testing/detectors/uNetThreeComponentP.cpp
    testing/detectors/uNetThreeComponentS.cpp
    testing/eventClassifiers/cnnThreeComponent.cpp
)
add_executable(unitTests ${TEST_SRC})
set_target_properties(unitTests PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES 
                      CXX_EXTENSIONS NO)
target_link_libraries(unitTests PRIVATE uussmlmodels ${GTEST_BOTH_LIBRARIES})
if (${OpenVINO_Runtime_FOUND})
   target_link_libraries(unitTests PRIVATE openvino::runtime)
endif()
if (${TORCH_FOUND})
   target_link_libraries(unitTests PRIVATE ${TORCH_LIBRARIES})
endif()

target_include_directories(unitTests
                           PRIVATE ${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
add_test(NAME unitTests
         COMMAND unitTests)

##########################################################################################
#                                         Installation                                   #
##########################################################################################
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/UUSSMLModelsConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)
if (${WRAP_PYTHON})
   message("add python result")
   install(TARGETS uussmlmodels pyuussmlmodels
           EXPORT ${PROJECT_NAME}-targets
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
           LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
           INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
           COMPONENT Runtime)
else()
   install(TARGETS uussmlmodels
           EXPORT ${PROJECT_NAME}-targets
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
           LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
           INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
           COMPONENT Runtime)
endif()
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/uussmlmodels
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
install(FILES
        ${CMAKE_SOURCE_DIR}/detectors/uNetThreeComponentP/models/detectorsUNetThreeComponentP.onnx
        ${CMAKE_SOURCE_DIR}/detectors/uNetThreeComponentS/models/detectorsUNetThreeComponentS.onnx
        ${CMAKE_SOURCE_DIR}/detectors/uNetOneComponentP/models/detectorsUNetOneComponentP.onnx
        ${CMAKE_SOURCE_DIR}/firstMotionClassifiers/cnnOneComponentP/models/firstMotionClassifiersCNNOneComponentP.onnx
        ${CMAKE_SOURCE_DIR}/pickers/cnnOneComponentP/models/pickersCNNOneComponentP.onnx
        ${CMAKE_SOURCE_DIR}/pickers/cnnThreeComponentS/models/pickersCNNThreeComponentS.onnx
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
##########################################################################################
#                                     CPACK Packaging                                    #
##########################################################################################
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VENDOR "UUSS")
set(CPACK_PACKAGE_CONTACT "ben.baker@utah.edu")
set(CPACK_PACKAGE_LICENSE "MIT")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "University of Utah Seismograph Machine Learning Models")
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

