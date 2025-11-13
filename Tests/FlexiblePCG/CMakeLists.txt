project(unitt_FlexiblePCG)

if(USE_GPU)
    file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cu ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
    file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cuh ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
else()
    file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
    file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
endif()
list(FILTER SRC_FILES EXCLUDE REGEX "unitt_.*\\.(cu|cpp)$")
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

foreach(target ${PROJECT_NAME}_32 ${PROJECT_NAME}_64)
    target_include_directories(${target} PUBLIC ${HEADER_DIR})
endforeach()

add_test(NAME ${PROJECT_NAME}_32 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_32> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
add_test(NAME ${PROJECT_NAME}_64 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_64> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
