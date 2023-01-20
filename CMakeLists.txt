cmake_minimum_required(VERSION 3.16)
project(UUSSMLModels VERSION 0.1.0 LANGUAGES CXX)
enable_testing()

set(BUILD_SHARED_LIBS YES)
include(CheckCXXCompilerFlag)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(GTest REQUIRED)
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

set(PUBLIC_HEADERS
    ${CMAKE_SOURCE_DIR}/detectors/uNetThreeComponentP/inference/inference.hpp)
set(INFERENCE_SRC
   ${CMAKE_SOURCE_DIR}/detectors/uNetThreeComponentP/inference/inference.cpp)

add_library(uussmlmodels ${INFERENCE_SRC})
set_target_properties(uussmlmodels PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES 
                      CXX_EXTENSIONS NO) 
target_include_directories(uussmlmodels
                           PUBLIC  $<BUILD_INTERFACE:${PUBLIC_HEADERS}>
                           PUBLIC  $<INSTALL_INTERFACE:${PUBLIC_HEADERS}>
                           PRIVATE ${RTSEIS_INCLUDE_DIR})
target_link_libraries(uussmlmodels PRIVATE ${RTSEIS_LIBRARY})
if (${OpenVINO_Runtime_FOUND})
   target_include_directories(uussmlmodels PRIVATE openvino::runtime)
   target_link_libraries(uussmlmodels PRIVATE openvino::runtime)
endif()
if (${TORCH_FOUND})
   target_include_directories(uussmlmodels PRIVATE ${TORCH_INDLUE_DIRS})
   target_link_libraries(uussmlmodels PRIVATE ${TORCH_LIBRARIES})
endif()

##########################################################################################
#                                         Installation                                   #
##########################################################################################
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)
if (WRAP_PYTHON)
   message("add python result")
   install(TARGETS uussmlmodels
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/urts
           LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
else()
   install(TARGETS uussmlmodels
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/urts
           LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
#install(DIRECTORY ${PUBLIC_HEADERS}
#        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(FILES
#        #"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
#        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
#        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
#        COMPONENT ${PROJECT_NAME})
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

