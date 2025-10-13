project(unitt_IterSolvers)

add_executable(${PROJECT_NAME}_32 unitt_32.cpp)
if (USE_GPU)
    target_link_libraries(${PROJECT_NAME}_32 CUDA_Utils IterSolvers TensorUtils)
else()
    target_link_libraries(${PROJECT_NAME}_32 IterSolvers TensorUtils)
endif()

add_test(NAME ${PROJECT_NAME}_32 COMMAND ${PROJECT_NAME}_32)