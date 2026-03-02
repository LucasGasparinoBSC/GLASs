project(unitt_FlexiblePCG)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cu ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cuh ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
list(FILTER SRC_FILES EXCLUDE REGEX "unitt_.*\\.(cu|cpp)$")
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR})

# FP32
add_executable(${PROJECT_NAME}_32 unitt_32.cpp ${SRC_FILES} ${INC_FILES})
target_link_libraries(${PROJECT_NAME}_32 PUBLIC IterSolvers)
target_include_directories(${PROJECT_NAME}_32 PUBLIC ${HEADER_DIR})

#foreach(target ${PROJECT_NAME}_32 ${PROJECT_NAME}_64)
#    target_include_directories(${target} PUBLIC ${HEADER_DIR})
#endforeach()

add_test(NAME ${PROJECT_NAME}_32 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_32> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})