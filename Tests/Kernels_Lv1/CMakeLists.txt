project(unitt_Kernels_Lv1)

add_executable(${PROJECT_NAME}_32 unitt_32.cpp)
target_link_libraries(${PROJECT_NAME}_32 Kernels_Lv1 DeviceMemory)
add_test(NAME ${PROJECT_NAME}_32 COMMAND ${PROJECT_NAME}_32)

add_executable(${PROJECT_NAME}_64 unitt_64.cpp)
target_link_libraries(${PROJECT_NAME}_64 Kernels_Lv1 DeviceMemory)
add_test(NAME ${PROJECT_NAME}_64 COMMAND ${PROJECT_NAME}_64)