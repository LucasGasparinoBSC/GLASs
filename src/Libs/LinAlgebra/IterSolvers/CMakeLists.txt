project(IterSolvers LANGUAGES CXX Fortran)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.F90)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/includes)
set(LIB_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/modules)

add_library(${PROJECT_NAME} ${LIBRARY_TYPE} ${SRC_FILES} ${INC_FILES})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${HEADER_DIR}
        ${LIB_MODULE_DIR}
)

if (USE_GPU)
    target_link_libraries(${PROJECT_NAME} PUBLIC CUDA_Utils TensorUtils)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC TensorUtils)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON Fortran_MODULE_DIRECTORY ${LIB_MODULE_DIR})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${INC_FILES} DESTINATION include/IterSolvers)