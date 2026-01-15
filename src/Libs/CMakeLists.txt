if(USE_GPU)
    add_subdirectory(CUDA_Utils)
    add_subdirectory(DeviceMemory)
endif()

add_subdirectory(Comm_Utils)
add_subdirectory(TensorUtils)
add_subdirectory(LinAlgebra)
add_subdirectory(FEM2D)