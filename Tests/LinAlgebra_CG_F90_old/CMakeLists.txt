project(unitt_LinAlgebra_CG_F90)

# FP32
add_executable(${PROJECT_NAME}_32 unitt_32.F90)
target_link_libraries(${PROJECT_NAME}_32 TensorUtils IterSolvers)

# FP64
# add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES} ${INC_FILES})
# target_link_libraries(${PROJECT_NAME}_64 TensorUtils IterSolvers)

add_test(NAME ${PROJECT_NAME}_32 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_32> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})