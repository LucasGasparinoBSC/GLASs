project(unitt_GMRES)

set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR})

if(USE_GPU)
    file(GLOB_RECURSE SRC_FILES_DEVICE ${CMAKE_CURRENT_LIST_DIR}/*.cu)
    file(GLOB_RECURSE INC_FILES_DEVICE ${CMAKE_CURRENT_LIST_DIR}/*.cuh)
    if(USE_AMD_GPU)
        set_source_files_properties(${SRC_FILES_DEVICE} PROPERTIES LANGUAGE HIP)
    endif()
    # Generate a shared library with the HIP sources
    add_library(DeviceSide SHARED ${SRC_FILES_DEVICE} ${INC_FILES_DEVICE})
    target_link_libraries(DeviceSide PUBLIC DeviceUtils)
endif()

file(GLOB_RECURSE SRC_FILES_HOST ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB_RECURSE INC_FILES_HOST ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
list(FILTER SRC_FILES_HOST EXCLUDE REGEX "unitt_.*\\.(cu|cpp)$")

# FP32 executable
add_executable(${PROJECT_NAME}_32 unitt_32.cpp ${SRC_FILES_HOST} ${INC_FILES_HOST})
target_include_directories(${PROJECT_NAME}_32 PUBLIC ${HEADER_DIR})
if(USE_GPU)
    target_link_libraries(${PROJECT_NAME}_32 PUBLIC DeviceSide IterSolvers)
else()
    target_link_libraries(${PROJECT_NAME}_32 PUBLIC IterSolvers)
endif()

add_test(NAME ${PROJECT_NAME}_32 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_32> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

# FP64 executable:
add_executable(${PROJECT_NAME}_64 unitt_64.cpp ${SRC_FILES_HOST} ${INC_FILES_HOST})
target_include_directories(${PROJECT_NAME}_64 PUBLIC ${HEADER_DIR})
if(USE_GPU)
    target_link_libraries(${PROJECT_NAME}_64 PUBLIC DeviceSide IterSolvers)
else()
    target_link_libraries(${PROJECT_NAME}_64 PUBLIC IterSolvers)
endif()
add_test(NAME ${PROJECT_NAME}_64 
         COMMAND mpirun -np 2 $<TARGET_FILE:${PROJECT_NAME}_64> 
         WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})