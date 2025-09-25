project(unitt_CUDA_Utils)

include_directories(SYSTEM ${CUDA_Utils_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}_16 unitt_16.cpp)
target_link_libraries(${PROJECT_NAME}_16 CUDA_Utils)
add_test(NAME ${PROJECT_NAME}_16 COMMAND ${PROJECT_NAME}_16)

add_executable(${PROJECT_NAME}_32 unitt_32.cpp)
target_link_libraries(${PROJECT_NAME}_32 CUDA_Utils)
add_test(NAME ${PROJECT_NAME}_32 COMMAND ${PROJECT_NAME}_32)

add_executable(${PROJECT_NAME}_64 unitt_64.cpp)
target_link_libraries(${PROJECT_NAME}_64 CUDA_Utils)
add_test(NAME ${PROJECT_NAME}_64 COMMAND ${PROJECT_NAME}_64)