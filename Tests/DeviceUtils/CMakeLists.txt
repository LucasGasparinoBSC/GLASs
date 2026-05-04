project(unitt_DeviceUtils)

add_executable(${PROJECT_NAME}_32 unitt_32.cpp)
#set_target_properties(${PROJECT_NAME}_32 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME}_32 DeviceUtils DeviceMemory Kernels_Lv1)
add_test(NAME ${PROJECT_NAME}_32 COMMAND mpirun -np 1 $<TARGET_FILE:${PROJECT_NAME}_32> WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})