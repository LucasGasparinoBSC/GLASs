project(IterSolvers LANGUAGES CXX Fortran)

# Find all source and header files in IterSolvers
file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.F90)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)

# Find MathOps source files (ArnoldiIter.cpp)
set(MATHOPS_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../MathOps/sources)
file(GLOB_RECURSE MATHOPS_SRC_FILES ${MATHOPS_SRC_DIR}/*.cpp)

# Header directories
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/includes)
set(LIB_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/modules)
set(MATHOPS_HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/../MathOps/includes)

# Build the library — includes both IterSolvers and MathOps sources
add_library(${PROJECT_NAME} ${LIBRARY_TYPE} ${SRC_FILES} ${INC_FILES} ${MATHOPS_SRC_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC EntryPoint)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${HEADER_DIR}>
        $<BUILD_INTERFACE:${LIB_MODULE_DIR}>
        $<BUILD_INTERFACE:${MATHOPS_HEADER_DIR}>
        $<INSTALL_INTERFACE:include/IterSolvers>
)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    POSITION_INDEPENDENT_CODE ON 
    Fortran_MODULE_DIRECTORY ${LIB_MODULE_DIR})

install(TARGETS ${PROJECT_NAME}
    EXPORT GLASSTargets
    DESTINATION lib
)

install(FILES ${INC_FILES} DESTINATION include/IterSolvers)

install(DIRECTORY ${LIB_MODULE_DIR}/ DESTINATION include/IterSolvers 
    FILES_MATCHING PATTERN "*.mod")