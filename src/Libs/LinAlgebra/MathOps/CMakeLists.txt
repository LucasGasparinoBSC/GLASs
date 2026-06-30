project(MathOps)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/includes)

add_library(${PROJECT_NAME} ${LIBRARY_TYPE} ${SRC_FILES} ${INC_FILES})
if (USE_GPU)
    target_link_libraries(${PROJECT_NAME} PUBLIC EntryPoint CUDA_Utils Comm_Utils TensorUtils)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC EntryPoint Comm_Utils TensorUtils)
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${HEADER_DIR}>
        $<INSTALL_INTERFACE:include/MathOps>
)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS ${PROJECT_NAME}
    EXPORT GLASSTargets
    DESTINATION lib
)

install(FILES ${INC_FILES} DESTINATION include/MathOps)