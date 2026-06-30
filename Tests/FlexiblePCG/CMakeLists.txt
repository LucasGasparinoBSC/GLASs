project(unitt_FlexiblePCG)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB_RECURSE SRC_FILES_DEVICE ${CMAKE_CURRENT_LIST_DIR}/*.cu)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cuh ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR})
if(USE_AMD_GPU)
    set_source_files_properties(${SRC_FILES_DEVICE} PROPERTIES LANGUAGE HIP)
endif()
list(FILTER SRC_FILES EXCLUDE REGEX "unitt_.*\\.(cu|cpp)$")
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR})

# FP32
add_executable(${PROJECT_NAME}_32 unitt_32.cpp ${SRC_FILES} ${SRC_FILES_DEVICE} ${INC_FILES})
set_target_properties(${PROJECT_NAME}_32 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME}_32 PUBLIC IterSolvers)
target_include_directories(${PROJECT_NAME}_32 PUBLIC ${HEADER_DIR})
add_test(NAME ${PROJECT_NAME}_32 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_32> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

# FP64
add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES} ${SRC_FILES_DEVICE} ${INC_FILES})
set_target_properties(${PROJECT_NAME}_64 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME}_64 PUBLIC IterSolvers)
target_include_directories(${PROJECT_NAME}_64 PUBLIC ${HEADER_DIR})
add_test(NAME ${PROJECT_NAME}_64 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_64> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

# Copy unitt_32_sol.dat to where the test is built
configure_file(${CMAKE_CURRENT_LIST_DIR}/unitt_32_sol.dat ${CMAKE_CURRENT_BINARY_DIR}/unitt_32_sol.dat COPYONLY)