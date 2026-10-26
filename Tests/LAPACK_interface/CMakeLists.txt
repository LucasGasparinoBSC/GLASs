project(unitt_LAPACK_interface)

file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(${PROJECT_NAME}_32 unitt_32.cpp ${INC_FILES})
target_link_libraries(${PROJECT_NAME}_32 TensorUtils Comm_Utils)

add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${INC_FILES})
target_link_libraries(${PROJECT_NAME}_64 TensorUtils Comm_Utils)

add_test(NAME ${PROJECT_NAME}_32 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_32> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
add_test(NAME ${PROJECT_NAME}_64 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_64> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})