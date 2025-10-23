project(Diffusion1D)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.F90)

if(USE_GPU)
    include_directories(SYSTEM ${IterSolvers_INCLUDE_DIRS} ${CUDA_Utils_INCLUDE_DIRS} ${TensorUtils_INCLUDE_DIRS})
    add_executable(${PROJECT_NAME} ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME} CUDA_Utils TensorUtils IterSolvers)
else()
    include_directories(SYSTEM ${HEADER_DIR} ${IterSolvers_INCLUDE_DIRS} ${TensorUtils_INCLUDE_DIRS})
    add_executable(${PROJECT_NAME} ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME} TensorUtils IterSolvers)
endif()

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})