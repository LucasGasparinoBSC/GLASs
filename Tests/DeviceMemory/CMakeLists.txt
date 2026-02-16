project(unitt_DeviceMemory)

add_executable(${PROJECT_NAME}_16 unitt_16.cpp)
target_link_libraries(${PROJECT_NAME}_16 DeviceMemory)
add_test(NAME ${PROJECT_NAME}_16 COMMAND ${PROJECT_NAME}_16)

add_executable(${PROJECT_NAME}_32 unitt_32.cpp)
target_link_libraries(${PROJECT_NAME}_32 DeviceMemory)
add_test(NAME ${PROJECT_NAME}_32 COMMAND ${PROJECT_NAME}_32)

add_executable(${PROJECT_NAME}_64 unitt_64.cpp)
target_link_libraries(${PROJECT_NAME}_64 DeviceMemory)
add_test(NAME ${PROJECT_NAME}_64 COMMAND ${PROJECT_NAME}_64)