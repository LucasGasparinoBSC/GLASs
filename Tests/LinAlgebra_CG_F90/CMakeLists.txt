project(unitt_LinAlgebra_CG_F90)

if(USE_GPU)

    # FP32
    add_executable(${PROJECT_NAME}_32 unitt_32.F90)
    target_link_libraries(${PROJECT_NAME}_32 CUDA_Utils TensorUtils IterSolvers)

    # FP64
    # add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES} ${INC_FILES})
    # target_link_libraries(${PROJECT_NAME}_64 CUDA_Utils TensorUtils IterSolvers)
else()

    # FP32
    add_executable(${PROJECT_NAME}_32 unitt_32.F90)
    target_link_libraries(${PROJECT_NAME}_32 TensorUtils IterSolvers)

    # FP64
    # add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES} ${INC_FILES})
    # target_link_libraries(${PROJECT_NAME}_64 TensorUtils IterSolvers)
endif()

add_test(NAME ${PROJECT_NAME}_32 COMMAND ${PROJECT_NAME}_32)