project(unitt_IterSolvers)

if (USE_GPU)
    include_directories(SYSTEM ${IterSolvers_INCLUDE_DIRS} ${CUDA_Utils_INCLUDE_DIRS} ${TensorUtils_INCLUDE_DIRS})
else()
    include_directories(SYSTEM ${IterSolvers_INCLUDE_DIRS} ${TensorUtils_INCLUDE_DIRS})
endif()

add_executable(${PROJECT_NAME}_32 unitt_32.cpp)
target_link_libraries(${PROJECT_NAME}_32 IterSolvers)
add_test(NAME ${PROJECT_NAME}_32 COMMAND ${PROJECT_NAME}_32)