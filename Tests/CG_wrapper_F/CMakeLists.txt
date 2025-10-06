project(unitt_cg_wrapper_f)

if(USE_GPU)
    include_directories(SYSTEM ${IterSolvers_INCLUDE_DIRS} ${CUDA_Utils_INCLUDE_DIRS} ${TensorUtils_INCLUDE_DIRS})

    # FP32
    add_executable(${PROJECT_NAME}_32 unitt_32.F90)
    target_link_libraries(${PROJECT_NAME}_32 CUDA_Utils TensorUtils IterSolvers)

    # FP64
    # add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES} ${INC_FILES})
    # target_link_libraries(${PROJECT_NAME}_64 CUDA_Utils TensorUtils IterSolvers)
else()
    include_directories(SYSTEM ${HEADER_DIR} ${IterSolvers_INCLUDE_DIRS} ${TensorUtils_INCLUDE_DIRS})

    # FP32
    add_executable(${PROJECT_NAME}_32 unitt_32.F90)
    target_link_libraries(${PROJECT_NAME}_32 TensorUtils IterSolvers)

    # FP64
    # add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES} ${INC_FILES})
    # target_link_libraries(${PROJECT_NAME}_64 TensorUtils IterSolvers)
endif()

add_test(NAME ${PROJECT_NAME}_32 COMMAND ${PROJECT_NAME}_32)