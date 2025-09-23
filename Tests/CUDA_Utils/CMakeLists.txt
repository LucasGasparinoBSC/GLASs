project(unitt_CUDA_Utils)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

include_directories(SYSTEM ${CUDA_Utils_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} CUDA_Utils)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})