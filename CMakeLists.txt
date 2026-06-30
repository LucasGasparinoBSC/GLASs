# Set minimum CMake version
cmake_minimum_required(VERSION 3.21)

# Set project versions: major, minor and patch
set(TEST_VERSION_MAJOR "0")
set(TEST_VERSION_MINOR "0")
set(TEST_VERSION_PATCH "0")
set(TEST_VERSION "${TEST_VERSION_MAJOR}.${TEST_VERSION_MINOR}.${TEST_VERSION_PATCH}")

## Set the project versions
set(PROJECT_VERSION_MAJOR "26")
set(PROJECT_VERSION_MINOR "3")
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

# Options
option(UNIT_TESTING "Enable unit tests" ON)
option(USE_GPU "Compile using GPU" OFF)
option(USE_HDF5 "Compile using HDF5" OFF)
option(ENABLE_APPLICATIONS "Enable standalone applications" OFF)

# Select from a variety of pre-defined machines
LIST(APPEND PREDEF_MACHINES "LUMI" "MN5_ACC" "MN5_GPP" "NG_GRACE" "USER")
LIST(GET PREDEF_MACHINES 4 PREDEF_MACHINE_DEFAULT)
set(PREDEF_MACHINE "${PREDEF_MACHINE_DEFAULT}" CACHE STRING "Select a pre-defined machine: ${PREDEF_MACHINES}")
set_property(CACHE PREDEF_MACHINE PROPERTY STRINGS ${PREDEF_MACHINES})

# Override based on pre-defined machine
if(PREDEF_MACHINE STREQUAL "LUMI")
    set(USE_GPU ON)
    set(USE_AMD_GPU ON)
    set(USE_NVIDIA_GPU OFF)
elseif(PREDEF_MACHINE STREQUAL "MN5_ACC")
    set(USE_GPU ON)
    set(USE_NVIDIA_GPU ON)
    set(USE_AMD_GPU OFF)
elseif(PREDEF_MACHINE STREQUAL "MN5_GPP")
    set(USE_GPU OFF)
    set(USE_NVIDIA_GPU OFF)
    set(USE_AMD_GPU OFF)
elseif(PREDEF_MACHINE STREQUAL "NG_GRACE")
    set(USE_GPU OFF)
endif()

# Select GPU vendor
if(USE_GPU)
    option(USE_NVIDIA_GPU "Use NVIDIA GPU" ON)
    option(USE_AMD_GPU "Use AMD GPU" OFF)
    # Check that only one GPU vendor is selected
    if(USE_NVIDIA_GPU AND USE_AMD_GPU)
        message(FATAL_ERROR "Please select only one GPU vendor: either USE_NVIDIA_GPU or USE_AMD_GPU.")
    elseif(NOT USE_NVIDIA_GPU AND NOT USE_AMD_GPU)
        message(FATAL_ERROR "Please select a GPU vendor: either USE_NVIDIA_GPU or USE_AMD_GPU.")
    endif()
    # NCCL/RCCL support
    if(USE_NVIDIA_GPU)
        option(USE_NCCL "Enable NCCL support for multi-GPU communication (NVIDIA GPUs only)" OFF)
    elseif(USE_AMD_GPU)
        option(USE_RCCL "Enable RCCL support for multi-GPU communication (AMD GPUs only)" OFF)
    endif()
endif()

# Folder with files configuring extra CMake options
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Set compilers to MPI wrappers
if(PREDEF_MACHINE STREQUAL "LUMI")
    set(ENV{CC} "cc")
    set(ENV{CXX} "CC")
    set(ENV{FC} "ftn")
else()
    set(ENV{CC} "mpicc")
    set(ENV{CXX} "mpicxx")
    set(ENV{FC} "mpif90")
endif()

# Set project name, version and languages
if(USE_GPU)
    if(USE_NVIDIA_GPU)
        project(GLASs VERSION ${PROJECT_VERSION} LANGUAGES C CXX Fortran CUDA)
    elseif(USE_AMD_GPU)
        project(GLASs VERSION ${PROJECT_VERSION} LANGUAGES C CXX Fortran HIP)
    endif()
else()
    project(GLASs VERSION ${PROJECT_VERSION} LANGUAGES C CXX Fortran)
endif()

# Configure MPI module, unless its LUMI where MPI is configured by the environment
if(NOT PREDEF_MACHINE STREQUAL "LUMI")
    include(mpi)
endif()

# Configure compiler options
if(USE_GPU AND USE_NVIDIA_GPU)
    include(gpu)
endif()
include(compilerOps)

# Configure HDF5 module
if(USE_HDF5)
    include(hdf5)
endif()

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

# Build everything in src/ (this includes Libs and all sublibraries)
add_subdirectory(src)

# --------------------------------------------------------------------------
# Install + Export GLASS package
# --------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/GLASSConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GLASSConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/GLASSConfig.cmake"
    INSTALL_DESTINATION lib/cmake/GLASS
)

# Export all sublibrary targets that were installed with EXPORT GLASSTargets
install(EXPORT GLASSTargets
    FILE GLASSTargets.cmake
    NAMESPACE GLASS::
    DESTINATION lib/cmake/GLASS
)

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/GLASSConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/GLASSConfigVersion.cmake"
    DESTINATION lib/cmake/GLASS
)

# --------------------------------------------------------------------------
# Applications and Tests (after package setup)
# --------------------------------------------------------------------------

# Add application folder
if(ENABLE_APPLICATIONS)
    add_subdirectory(Apps)
endif()

# Enable testing
if(UNIT_TESTING)
    enable_testing()
    add_subdirectory(Tests)
endif()