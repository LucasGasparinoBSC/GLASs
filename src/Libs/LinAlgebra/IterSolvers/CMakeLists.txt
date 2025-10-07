project(IterSolvers LANGUAGES CXX Fortran)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.F90)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/includes)
set(LIB_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/modules)

if (USE_GPU)
    include_directories(${HEADER_DIR} ${CUDA_Utils_INCLUDE_DIRS} ${TensorUtils_INCLUDE_DIRS})
    add_library(${PROJECT_NAME} ${LIBRARY_TYPE} ${SRC_FILES} ${INC_FILES})
    target_link_libraries(${PROJECT_NAME} CUDA_Utils TensorUtils)
else()
    include_directories(${HEADER_DIR} ${TensorUtils_INCLUDE_DIRS})
    add_library(${PROJECT_NAME} ${LIBRARY_TYPE} ${SRC_FILES} ${INC_FILES})
    target_link_libraries(${PROJECT_NAME} TensorUtils)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON Fortran_MODULE_DIRECTORY ${LIB_MODULE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${LIB_MODULE_DIR} ${HEADER_DIR})

set(${PROJECT_NAME}_INCLUDE_DIRS ${HEADER_DIR} CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${INC_FILES} DESTINATION include/IterSolvers)