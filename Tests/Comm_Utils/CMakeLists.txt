project(unitt_Comm_Utils)

file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR})

include_directories(SYSTEM ${Comm_Utils_INCLUDE_DIRS} ${TensorUtils_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}_32 unitt_32.cpp ${SRC_FILES} ${INC_FILES})
target_link_libraries(${PROJECT_NAME}_32 TensorUtils Comm_Utils)

add_test(NAME ${PROJECT_NAME} COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_32> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
