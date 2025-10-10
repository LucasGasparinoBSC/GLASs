project(unitt_ConjugateGradient)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cu)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cuh)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR})

if(USE_GPU)
    # FP32
    add_executable(${PROJECT_NAME}_32 unitt_32.cpp ${SRC_FILES} ${INC_FILES})
    target_link_libraries(${PROJECT_NAME}_32 PUBLIC CUDA_Utils TensorUtils IterSolvers)
    # FP64
    add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES} ${INC_FILES})
    target_link_libraries(${PROJECT_NAME}_64 PUBLIC CUDA_Utils TensorUtils IterSolvers)
else()
    # FP32
    add_executable(${PROJECT_NAME}_32 unitt_32.cpp ${SRC_FILES} ${INC_FILES})
    target_link_libraries(${PROJECT_NAME}_32 PUBLIC TensorUtils IterSolvers)
    # FP64
    add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES} ${INC_FILES})
    target_link_libraries(${PROJECT_NAME}_64 PUBLIC TensorUtils IterSolvers)
endif()

target_include_directories(${PROJECT_NAME}_32 PUBLIC ${HEADER_DIR})

add_test(NAME ${PROJECT_NAME}_32 COMMAND ${PROJECT_NAME}_32)
add_test(NAME ${PROJECT_NAME}_64 COMMAND ${PROJECT_NAME}_64)
