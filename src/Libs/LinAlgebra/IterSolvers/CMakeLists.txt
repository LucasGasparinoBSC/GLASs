project(IterSolvers)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/includes)

if (USE_GPU)
    include_directories(${HEADER_DIR} ${CUDA_Utils_INCLUDE_DIRS})
    add_library(${PROJECT_NAME} ${LIBRARY_TYPE} ${SRC_FILES} ${INC_FILES})
    target_link_libraries(${PROJECT_NAME} CUDA_Utils)
else()
    include_directories(${HEADER_DIR})
    add_library(${PROJECT_NAME} ${LIBRARY_TYPE} ${SRC_FILES} ${INC_FILES})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(${PROJECT_NAME}_INCLUDE_DIRS ${HEADER_DIR} CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${INC_FILES} DESTINATION include/IterSolvers)