# Set minimum CMake version
cmake_minimum_required(VERSION 3.15)

# Set project versions: major, minor and patch
set(TEST_VERSION_MAJOR "0")
set(TEST_VERSION_MINOR "0")
set(TEST_VERSION_PATCH "0")
set(TEST_VERSION "${TEST_VERSION_MAJOR}.${TEST_VERSION_MINOR}.${TEST_VERSION_PATCH}")

# Options
option(UNIT_TESTING "Enable unit tests" ON)
option(USE_GPU "Compile using GPU" ON)
option(USE_NCCL "Compile with NCCL support" OFF)

# Folder with files configuring extra CMake options
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Set compilers to MPI wrappers
set(ENV{CC} "mpicc")
set(ENV{CXX} "mpicxx")
set(ENV{FC} "mpif90")

# Set project name, version and languages
if(USE_GPU)
    project(GLASs LANGUAGES C CXX Fortran CUDA)
else()
    project(GLASs LANGUAGES C CXX Fortran)
endif()

# Configure MPI module
include(mpi)

# Configure compiler options
if(USE_GPU)
    include(gpu)
endif()
include(compilerOps)

# Configure HDF5 module
include(hdf5)

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

# Start compiling the source code
add_subdirectory(src)

# Enable testing
if(UNIT_TESTING)
    enable_testing()
    add_subdirectory(Tests)
endif()

